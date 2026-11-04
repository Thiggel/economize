"""``economize.integrations.generate`` against ``transformers``' own ``generate`` (fp32, tiny random Qwen2)."""

from __future__ import annotations

import functools
from typing import Any

import pytest
import torch
from eco_fakes import FakeEfficientModel, RecordingStreamer, make_tokenizer
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

import economize
from economize import DecodeConfig, JumpConfig
from economize.errors import ConfigError
from economize.integrations.generate import generate
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, perturb


@pytest.fixture(scope="module")
def base() -> torch.nn.Module:
    return make_tiny_model()


def _prompt(seed: int = 1, batch: int = 1) -> torch.Tensor:
    return torch.randint(1, VOCAB, (batch, PROMPT_LEN), generator=torch.Generator().manual_seed(seed))


class Suppress(LogitsProcessor):
    def __init__(self, token: int) -> None:
        self.token = token

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores[:, self.token] = -float("inf")
        return scores


class StopAfter(StoppingCriteria):
    def __init__(self, length: int) -> None:
        self.length = length

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor | None, **kwargs: object) -> torch.Tensor:
        return torch.full((input_ids.shape[0],), input_ids.shape[1] >= self.length, dtype=torch.bool)


@pytest.mark.parametrize("chunk", [1, 3])
def test_greedy_equals_hf(base: torch.nn.Module, chunk: int) -> None:
    prompt = _prompt()
    expected = base.generate(prompt, max_new_tokens=20, do_sample=False)
    got = generate(FakeEfficientModel(base, chunk=chunk), prompt, max_new_tokens=20, do_sample=False)
    assert torch.equal(got, expected)


@pytest.mark.parametrize("chunk", [1, 4])
def test_eos_list_and_processors_equal_hf(base: torch.nn.Module, chunk: int) -> None:
    prompt = _prompt(2)
    free = base.generate(prompt, max_new_tokens=24, do_sample=False)[0, PROMPT_LEN:].tolist()
    eos = [free[9], 127]  # stops at the tenth token
    kwargs = dict(max_new_tokens=24, eos_token_id=eos, repetition_penalty=1.3, min_new_tokens=3, top_k=5,
                  logits_processor=LogitsProcessorList([Suppress(free[2])]))
    expected = base.generate(prompt, **kwargs)
    got = generate(FakeEfficientModel(base, chunk=chunk), prompt, **kwargs)
    assert torch.equal(got, expected)


@pytest.mark.parametrize("chunk", [1, 3])
def test_stopping_criteria_list(base: torch.nn.Module, chunk: int) -> None:
    prompt = _prompt(3)
    criteria = StoppingCriteriaList([StopAfter(PROMPT_LEN + 7)])
    expected = base.generate(prompt, max_new_tokens=20, stopping_criteria=criteria)
    got = generate(FakeEfficientModel(base, chunk=chunk), prompt, max_new_tokens=20, stopping_criteria=criteria)
    assert got.shape[1] == PROMPT_LEN + 7
    assert torch.equal(got, expected)


@pytest.mark.parametrize("chunk", [1, 3])
def test_streamer_receives_exactly_the_generated_tokens(base: torch.nn.Module, chunk: int) -> None:
    prompt = _prompt(4)
    free = base.generate(prompt, max_new_tokens=16, do_sample=False)[0, PROMPT_LEN:].tolist()
    for eos in (None, free[6]):  # stop by length (no stop call for the last token) and by EOS mid-chunk
        streamer = RecordingStreamer()
        out = generate(FakeEfficientModel(base, chunk=chunk), prompt, max_new_tokens=16, eos_token_id=eos,
                       streamer=streamer)
        assert torch.equal(streamer.puts[0].view(-1), prompt.view(-1))
        assert streamer.generated() == out[0, PROMPT_LEN:].tolist()
        assert streamer.ended == 1


def test_text_iterator_streamer(base: torch.nn.Module) -> None:
    from transformers import TextIteratorStreamer

    tokenizer = make_tokenizer()
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=10)
    out = generate(FakeEfficientModel(base, chunk=3), _prompt(5), max_new_tokens=10, streamer=streamer)
    assert "".join(streamer).strip() == tokenizer.decode(out[0, PROMPT_LEN:]).strip()


def test_scores_and_logits_match_hf(base: torch.nn.Module) -> None:
    prompt = _prompt(6)
    kwargs = dict(max_new_tokens=8, return_dict_in_generate=True, output_scores=True, output_logits=True,
                  repetition_penalty=1.2)
    expected = base.generate(prompt, **kwargs)
    got = generate(FakeEfficientModel(base), prompt, **kwargs)
    assert torch.equal(got.sequences, expected.sequences)
    assert len(got.scores) == len(expected.scores) == len(got.logits) == 8
    for mine, theirs in zip(got.scores + got.logits, expected.scores + expected.logits, strict=True):
        torch.testing.assert_close(mine, theirs, atol=1e-4, rtol=1e-4)


def test_left_padded_batch_equals_hf(base: torch.nn.Module) -> None:
    prompt = _prompt(7, batch=3)
    mask = torch.ones_like(prompt)
    mask[1, :4], mask[2, :2] = 0, 0
    prompt[1, :4], prompt[2, :2] = 0, 0
    free = base.generate(prompt, attention_mask=mask, max_new_tokens=12, do_sample=False)
    eos = int(free[1, PROMPT_LEN + 5])
    expected = base.generate(prompt, attention_mask=mask, max_new_tokens=12, eos_token_id=eos, pad_token_id=0)
    got = generate(FakeEfficientModel(base, chunk=2), prompt, attention_mask=mask, max_new_tokens=12,
                   eos_token_id=eos, pad_token_id=0)
    assert torch.equal(got, expected)


def test_sampling_without_drafter_reproduces_hf_samples(base: torch.nn.Module) -> None:
    prompt = _prompt(8)
    kwargs = dict(max_new_tokens=12, do_sample=True, temperature=0.8, top_k=20, top_p=0.9, min_p=0.01)
    torch.manual_seed(0)
    expected = base.generate(prompt, **kwargs)
    torch.manual_seed(0)
    got = generate(FakeEfficientModel(base), prompt, **kwargs)
    assert torch.equal(got, expected)


def test_num_return_sequences(base: torch.nn.Module) -> None:
    got = generate(FakeEfficientModel(base), _prompt(9), max_new_tokens=5, do_sample=True, num_return_sequences=3)
    assert got.shape == (3, PROMPT_LEN + 5)


def test_stop_strings_with_tokenizer(base: torch.nn.Module) -> None:
    tokenizer = make_tokenizer()
    prompt = _prompt(10)
    free = base.generate(prompt, max_new_tokens=12, do_sample=False)[0, PROMPT_LEN:].tolist()
    stop = tokenizer.decode([free[4]])
    expected = base.generate(prompt, max_new_tokens=12, stop_strings=[stop], tokenizer=tokenizer)
    got = generate(FakeEfficientModel(base, chunk=3), prompt, max_new_tokens=12, stop_strings=[stop],
                   tokenizer=tokenizer)
    assert torch.equal(got, expected)


class StopOncePerRow(StoppingCriteria):
    """lm-eval's stop-sequence criteria in miniature: one ``done`` flag per batch row, kept across calls."""

    def __init__(self, length: int, batch_size: int = 1) -> None:
        self.length, self.done = length, [False] * batch_size

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor | None, **kwargs: object) -> bool:
        self.done = [done or input_ids.shape[1] >= self.length for done in self.done]
        return all(self.done)


@pytest.mark.parametrize("chunk", [1, 3])
def test_row_stopping_criteria_stop_each_row(base: torch.nn.Module, chunk: int) -> None:
    prompt = _prompt(11, batch=3)
    lengths = [3, None, 6]
    own = [None if n is None else StoppingCriteriaList([StopOncePerRow(PROMPT_LEN + n)]) for n in lengths]
    shared = StoppingCriteriaList([StopAfter(PROMPT_LEN + 9)])
    got = generate(FakeEfficientModel(base, chunk=chunk), prompt, max_new_tokens=12, pad_token_id=0,
                   stopping_criteria=shared, row_stopping_criteria=own)
    assert got.shape == (3, PROMPT_LEN + 9)
    for row, n in enumerate(lengths):
        n = n or 9
        alone = base.generate(prompt[row : row + 1], max_new_tokens=n, do_sample=False)[0]
        assert torch.equal(got[row, : PROMPT_LEN + n], alone)
        assert not got[row, PROMPT_LEN + n :].any()  # padded after its stop
    with pytest.raises(ConfigError, match="2 entries for a batch of 3"):
        generate(FakeEfficientModel(base), prompt, max_new_tokens=4, row_stopping_criteria=own[:2])


def test_finished_rows_are_padded_with_the_first_eos_without_pad_token(base: torch.nn.Module) -> None:
    prompt = _prompt(12, batch=2)
    free = base.generate(prompt, max_new_tokens=10, do_sample=False, pad_token_id=0)
    eos = [int(free[0, PROMPT_LEN + 3]), 127]
    config = base.generation_config.__class__.from_dict({**base.generation_config.to_dict(), "pad_token_id": None})
    expected = base.generate(prompt, max_new_tokens=10, eos_token_id=eos, generation_config=config)
    got = generate(FakeEfficientModel(base, chunk=2), prompt, max_new_tokens=10, eos_token_id=eos,
                   generation_config=config)
    assert torch.equal(got, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_beams=2),
        dict(penalty_alpha=0.6, top_k=4),
        dict(output_attentions=True, return_dict_in_generate=True),
        dict(output_hidden_states=True, return_dict_in_generate=True),
        dict(guidance_scale=1.5),
        dict(prompt_lookup_num_tokens=3),
        dict(synced_gpus=True),
        dict(custom_generate=lambda *a, **k: None),
        dict(negative_prompt_ids=torch.ones(1, 3, dtype=torch.long)),
        dict(token_type_ids=torch.zeros(1, PROMPT_LEN, dtype=torch.long)),
    ],
    ids=lambda kw: next(iter(kw)),
)
def test_unsupported_arguments_raise(base: torch.nn.Module, kwargs: dict) -> None:
    with pytest.raises((ConfigError, ValueError)) as info:
        generate(FakeEfficientModel(base), _prompt(), max_new_tokens=4, **kwargs)
    if "token_type_ids" not in kwargs:
        assert isinstance(info.value, ConfigError)


def test_assistant_model_raises(base: torch.nn.Module) -> None:
    with pytest.raises(ConfigError, match="assisted generation"):
        generate(FakeEfficientModel(base), _prompt(), max_new_tokens=4, assistant_model=base)


def test_right_padding_and_streamer_batch_raise(base: torch.nn.Module) -> None:
    prompt = _prompt(batch=2)
    mask = torch.ones_like(prompt)
    mask[0, -3:] = 0
    with pytest.raises(ConfigError, match="left padding"):
        generate(FakeEfficientModel(base), prompt, attention_mask=mask, max_new_tokens=4)
    with pytest.raises(ConfigError, match="batch size 1"):
        generate(FakeEfficientModel(base), prompt, max_new_tokens=4, streamer=RecordingStreamer())


def test_sampling_with_greedy_verifier_raises(base: torch.nn.Module) -> None:
    with pytest.raises(ConfigError, match="rejection"):
        generate(FakeEfficientModel(base, drafting=True), _prompt(), max_new_tokens=4, do_sample=True)
    out = generate(FakeEfficientModel(base, drafting=True, verifier="accept_all"), _prompt(), max_new_tokens=4,
                   do_sample=True)
    assert out.shape == (1, PROMPT_LEN + 4)


def test_prefilled_cache_and_embeddings_raise(base: torch.nn.Module) -> None:
    prompt = _prompt()
    cache = base(prompt, use_cache=True).past_key_values
    with pytest.raises(ConfigError, match="empty cache"):
        generate(FakeEfficientModel(base), prompt, max_new_tokens=4, past_key_values=cache)
    embeds = base.get_input_embeddings()(prompt)
    with pytest.raises(ConfigError, match="inputs_embeds"):
        generate(FakeEfficientModel(base), inputs_embeds=embeds, max_new_tokens=4)


# ---- the real core (skipped while the v0.2 core is being rebuilt) ---------------------------------------------


@pytest.mark.parametrize("cycle_length", [1, 4])
def test_core_spec_decoding_equals_hf_greedy(jump_model, cycle_length: int) -> None:
    model = jump_model(cycle_length=cycle_length)
    prompt = _prompt(11)
    expected = model.base_model.generate(prompt, max_new_tokens=24, do_sample=False)
    streamer = RecordingStreamer()
    got = generate(model, prompt, max_new_tokens=24, do_sample=False, streamer=streamer)
    assert torch.equal(got, expected)
    assert streamer.generated() == got[0, PROMPT_LEN:].tolist() and streamer.ended == 1


def test_core_processors_apply_to_drafts(jump_model) -> None:
    model = jump_model()
    prompt = _prompt(12)
    kwargs = dict(max_new_tokens=20, repetition_penalty=1.5, no_repeat_ngram_size=2, eos_token_id=[5, 9])
    assert torch.equal(generate(model, prompt, **kwargs), model.base_model.generate(prompt, **kwargs))


# ---- batched generate and the scores pass ----
@pytest.fixture(scope="module")
def perturbed_jump_model() -> Any:
    model = economize.apply(make_tiny_model(), JumpConfig(target_layer=2, source_layer=4),
                         decoding=DecodeConfig(cycle_length=4))
    perturb(model.techniques["jump"], std=0.05, seed=2)
    return model


def _left_padded(lengths: tuple[int, ...], seed: int = 4) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    prompts = [torch.randint(1, VOCAB, (n,), generator=generator) for n in lengths]
    width = max(lengths)
    ids = torch.zeros(len(prompts), width, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for b, prompt in enumerate(prompts):
        ids[b, width - prompt.numel():] = prompt
        mask[b, width - prompt.numel():] = 1
    return ids, mask, prompts


@torch.no_grad()
def test_batched_generate_decodes_once_and_equals_row_by_row_decoding(perturbed_jump_model: Any,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """A left-padded batch goes through one ``decode_batch`` call; every row equals its own ``decode``, with an
    eos token that stops some rows early and a pad token after them."""
    ids, mask, prompts = _left_padded((9, 5, 12))
    singles = [perturbed_jump_model.decode(prompt, 14).tokens.view(-1) for prompt in prompts]
    eos = int(singles[1][4])
    calls: list[int] = []
    decode_batch = type(perturbed_jump_model).decode_batch

    def counting(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(len(args[0]))
        return decode_batch(self, *args, **kwargs)

    monkeypatch.setattr(type(perturbed_jump_model), "decode_batch", counting)
    out = generate(perturbed_jump_model, ids, attention_mask=mask, max_new_tokens=14, do_sample=False, eos_token_id=eos,
                   pad_token_id=0)
    assert calls == [3]
    for b, single in enumerate(singles):
        stop = (single == eos).nonzero()
        expected = single[: int(stop[0]) + 1] if stop.numel() else single
        row = out[b, ids.shape[1]:]
        assert row[: expected.numel()].tolist() == expected.tolist(), f"row {b}"
        assert bool((row[expected.numel():] == 0).all())


@torch.no_grad()
def test_scores_pass_runs_the_lm_head_on_the_generated_positions_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from eco_fakes import FakeEfficientModel

    base = make_tiny_model()
    ids, mask, _ = _left_padded((7, 10))
    reference = base.generate(ids, attention_mask=mask, max_new_tokens=6, do_sample=False, pad_token_id=0,
                              return_dict_in_generate=True, output_scores=True, output_logits=True)
    seen: list[Any] = []
    forward = type(base).forward

    @functools.wraps(forward)
    def recording(self: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("logits_to_keep"))
        return forward(self, *args, **kwargs)

    monkeypatch.setattr(type(base), "forward", recording)
    out = generate(FakeEfficientModel(base), ids, attention_mask=mask, max_new_tokens=6, do_sample=False,
                   pad_token_id=0, return_dict_in_generate=True, output_scores=True, output_logits=True)
    assert seen[-1] == 7  # the fake core's own decoding calls come first
    assert torch.equal(out.sequences, reference.sequences)
    for got, want in zip(out.scores, reference.scores, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
    for got, want in zip(out.logits, reference.logits, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


# ---- draft trees under generate ------------------------------------------------------------------------------------
def _eagle3_tree_model() -> economize.EfficientModel:
    from economize.techniques.eagle3 import Eagle3Config

    tree = DecodeConfig(cycle_length=5, draft_policy="dynamic_tree",
                        draft_policy_options={"depth": 4, "topk": 3, "total_tokens": 10})
    model = economize.apply(make_tiny_model(), Eagle3Config(), decoding=tree)
    for technique in model.techniques.values():
        perturb(technique, 0.05, seed=1)
    return model


@torch.no_grad()
def test_draft_tree_under_generate_equals_hf_greedy() -> None:
    model = _eagle3_tree_model()
    prompt = _prompt(13)
    expected = model.base_model.generate(prompt, max_new_tokens=20, do_sample=False)
    assert torch.equal(generate(model, prompt, max_new_tokens=20, do_sample=False), expected)
    kwargs = dict(max_new_tokens=20, do_sample=False, suppress_tokens=[expected[0, PROMPT_LEN].item()])
    assert torch.equal(generate(model, prompt, **kwargs), model.base_model.generate(prompt, **kwargs))


@torch.no_grad()
def test_draft_tree_under_generate_samples_with_warpers() -> None:
    model = _eagle3_tree_model()
    typical = DecodeConfig(cycle_length=5, verifier="typical", draft_policy="dynamic_tree",
                           draft_policy_options={"depth": 4, "topk": 3, "total_tokens": 10})
    torch.manual_seed(0)
    out = generate(model, _prompt(14, batch=2), max_new_tokens=16, do_sample=True, temperature=0.8, top_k=20,
                   top_p=0.9, decoding=typical)
    assert out.shape == (2, PROMPT_LEN + 16)


def test_history_free_selector_matches_the_row_selector() -> None:
    from transformers import TemperatureLogitsWarper, TopKLogitsWarper

    from economize.decoding.capabilities import BatchDistributions, BatchSelector
    from economize.integrations._generation_args import HistoryFreeSelector, ProcessedSelector, make_selector

    warpers = LogitsProcessorList([TemperatureLogitsWarper(0.7), TopKLogitsWarper(5)])
    sampler = make_selector(warpers, do_sample=True)
    assert isinstance(sampler, HistoryFreeSelector)
    assert isinstance(sampler, BatchSelector) and isinstance(sampler, BatchDistributions)
    logits = torch.randn(3, VOCAB, generator=torch.Generator().manual_seed(2))
    history = torch.tensor([4, 7])
    for row in range(3):
        assert torch.allclose(sampler.distributions(logits)[row], sampler.distribution(logits[row], history))
    greedy = make_selector(None, do_sample=False)
    assert torch.equal(greedy.select_many(logits), logits.argmax(-1))
    assert torch.equal(greedy.distributions(logits).argmax(-1), logits.argmax(-1))
    penalty = make_selector(LogitsProcessorList([Suppress(3)]), do_sample=False)
    assert type(penalty) is ProcessedSelector and not isinstance(penalty, BatchSelector)
