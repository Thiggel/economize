"""``model.generate`` routed through economize's decoding loop, against ``transformers``' own ``generate``."""

from __future__ import annotations

import pytest
import torch

import economize
from economize import ConfigError, DecodeConfig, JumpConfig
from helpers import make_identity_jump, perturb, reference_greedy

MAX_NEW = 16


@pytest.fixture
def rejecting(tiny_model) -> economize.EfficientModel:
    """A drafter whose drafts are all rejected."""
    model = economize.apply(tiny_model, JumpConfig(target_layer=2, source_layer=4))
    perturb(model.techniques["jump"], 0.05)
    return model


@pytest.fixture(params=["rejecting", "accepting"])
def model(request, rejecting, tiny_model) -> economize.EfficientModel:
    """Either the rejecting drafter or one whose drafts are all accepted."""
    return rejecting if request.param == "rejecting" else make_identity_jump(tiny_model)


def _new_tokens(sequences: torch.Tensor, prompt: torch.Tensor) -> list[int]:
    assert sequences.shape[0] == 1 and sequences[0, : prompt.numel()].tolist() == prompt.tolist()
    return sequences[0, prompt.numel() :].tolist()


def test_greedy_generate_is_lossless(model, prompt) -> None:
    out = model.generate(prompt[None], max_new_tokens=MAX_NEW, do_sample=False)
    assert _new_tokens(out, prompt) == reference_greedy(model.base_model, prompt, MAX_NEW).tolist()
    plain = model.generate(prompt[None], max_new_tokens=MAX_NEW, decoding=DecodeConfig(cycle_length=1))
    assert _new_tokens(plain, prompt) == _new_tokens(out, prompt)


def test_decoding_options_are_passed_through(rejecting, prompt) -> None:
    lossy = DecodeConfig(cycle_length=3, lossless=False)
    out = rejecting.generate(prompt[None], max_new_tokens=MAX_NEW, decoding=lossy)
    assert _new_tokens(out, prompt) == rejecting.decode(prompt, MAX_NEW, decoding=lossy).tokens.tolist()
    assert _new_tokens(out, prompt) != reference_greedy(rejecting.base_model, prompt, MAX_NEW).tolist()


def test_return_dict_in_generate(rejecting, prompt) -> None:
    out = rejecting.generate(prompt[None], max_new_tokens=5, return_dict_in_generate=True)
    assert _new_tokens(out.sequences, prompt) == reference_greedy(rejecting.base_model, prompt, 5).tolist()


def test_logits_processors_match_transformers(model, prompt) -> None:
    """A repetition penalty depends on the history; lossless verification must apply it at every drafted row."""
    options = dict(max_new_tokens=MAX_NEW, do_sample=False, repetition_penalty=1.8, no_repeat_ngram_size=2)
    expected = model.base_model.generate(prompt[None], **options)
    assert model.generate(prompt[None], **options).tolist() == expected.tolist()


def test_eos_stops_generation(model, prompt) -> None:
    greedy = reference_greedy(model.base_model, prompt, MAX_NEW).tolist()
    eos = greedy[5]
    out = model.generate(prompt[None], max_new_tokens=MAX_NEW, eos_token_id=eos)
    assert _new_tokens(out, prompt) == greedy[: greedy.index(eos) + 1]
    assert out.tolist() == model.base_model.generate(prompt[None], max_new_tokens=MAX_NEW, eos_token_id=eos).tolist()


def test_sampling_needs_lossy_decoding(rejecting, prompt) -> None:
    with pytest.raises(ConfigError, match="lossless=False"):
        rejecting.generate(prompt[None], max_new_tokens=4, do_sample=True)
    lossy = DecodeConfig(cycle_length=4, lossless=False)
    runs = []
    for _ in range(2):
        torch.manual_seed(7)
        runs.append(rejecting.generate(prompt[None], max_new_tokens=MAX_NEW, do_sample=True, top_k=0, decoding=lossy))
    assert runs[0].tolist() == runs[1].tolist() and runs[0].shape[1] == prompt.numel() + MAX_NEW


def test_batches_go_through_generate_or_decode_batch(rejecting, prompt) -> None:
    with pytest.raises(ValueError, match="decode_batch"):
        rejecting.decode(prompt[None].repeat(2, 1), 4)
    greedy = reference_greedy(rejecting.base_model, prompt, 4).tolist()
    out = rejecting.generate(prompt[None].repeat(2, 1), max_new_tokens=4)
    assert out[:, prompt.numel() :].tolist() == [greedy, greedy]
    results = rejecting.decode_batch([prompt, prompt[3:]], 4)
    assert results[0].tokens.tolist() == greedy
    assert results[1].tokens.tolist() == reference_greedy(rejecting.base_model, prompt[3:], 4).tolist()
    full = torch.ones(1, prompt.numel(), dtype=torch.long)
    assert rejecting.generate(prompt[None], attention_mask=full, max_new_tokens=4).shape[1] == prompt.numel() + 4


def test_base_model_generate_is_unchanged(rejecting, prompt) -> None:
    out = rejecting.base_model.generate(prompt[None], max_new_tokens=MAX_NEW, do_sample=False)
    assert _new_tokens(out, prompt) == reference_greedy(rejecting.base_model, prompt, MAX_NEW).tolist()
