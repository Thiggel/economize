"""The Gemma-4 assistant drafter: lossless on a tiny random Gemma-4, and its drafts are those of ``transformers``."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import economize
from economize import ConfigError, DecodeConfig, UnsupportedModelError
from economize.core.technique import TECHNIQUES
from economize.decoding import LayerFootprint
from economize.techniques.gemma4_assistant import (
    MODULE,
    Gemma4AssistantDraftConfig,
    Gemma4AssistantDrafter,
    Gemma4AssistantTechnique,
)
from helpers import FAMILY_MODELS, PROMPT_LEN, VOCAB, WINDOW, make_family_model, make_tiny_model, reference_greedy

ASSISTANT_LAYERS = 4
ASSISTANT_WIDTH = 32


def make_assistant(base: torch.nn.Module, seed: int = 3, **changes: Any) -> torch.nn.Module:
    """A random float32 assistant for ``base``: three sliding layers and one full layer, all reading the base's
    cache entries, with the base's head geometry."""
    from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM

    text = base.config.get_text_config()
    text_config = dict(
        model_type="gemma4_text", vocab_size=VOCAB, hidden_size=ASSISTANT_WIDTH, intermediate_size=48,
        num_hidden_layers=ASSISTANT_LAYERS, num_kv_shared_layers=ASSISTANT_LAYERS,
        layer_types=["sliding_attention"] * (ASSISTANT_LAYERS - 1) + ["full_attention"],
        sliding_window=WINDOW, hidden_size_per_layer_input=0, vocab_size_per_layer_input=0,
        max_position_embeddings=256, final_logit_softcapping=None,
        **{f: FAMILY_MODELS["gemma4_text"][1][f] for f in ("num_attention_heads", "num_key_value_heads", "head_dim",
                                                           "global_head_dim", "num_global_key_value_heads",
                                                           "attention_k_eq_v")},
    )
    text_config.update(changes.pop("text", {}))
    settings: dict[str, Any] = dict(backbone_hidden_size=text.hidden_size, use_ordered_embeddings=False)
    settings.update(changes)
    config = Gemma4AssistantConfig(text_config=text_config, **settings)
    torch.manual_seed(seed)
    assistant = Gemma4AssistantForCausalLM._from_config(config, attn_implementation="sdpa").float().eval()
    with torch.no_grad():  # larger weights than the default init, so the drafts depend on the inputs
        for parameter in assistant.parameters():
            parameter.normal_(0.0, 0.2) if parameter.dim() > 1 else None
    return assistant.requires_grad_(False)


@pytest.fixture(scope="module")
def base() -> torch.nn.Module:
    return make_family_model("gemma4_text")


@pytest.fixture(scope="module")
def assistant_dir(base: torch.nn.Module, tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("gemma4_assistant")
    make_assistant(base).save_pretrained(path)
    return str(path)


def _prompts(seed: int = 1, lengths: tuple[int, ...] = (PROMPT_LEN, 7, 20)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in lengths]


def _model(path: str, base: torch.nn.Module, **decoding: Any) -> economize.EfficientModel:
    return economize.apply(base, Gemma4AssistantDraftConfig(model=path),
                           decoding=DecodeConfig(**decoding) if decoding else None)


# ---- configuration -------------------------------------------------------------------------------------------------
def test_registration_and_config() -> None:
    assert TECHNIQUES.get("gemma4_assistant") is Gemma4AssistantTechnique
    config = Gemma4AssistantDraftConfig(revision="main", dtype="bfloat16")
    assert config.model == "google/gemma-4-31B-it-assistant"
    assert Gemma4AssistantDraftConfig.from_dict(config.to_dict()) == config
    for bad in (dict(model=""), dict(dtype="int8"), dict(attn_implementation="flash_attention_2")):
        with pytest.raises(ConfigError):
            Gemma4AssistantDraftConfig(**bad)


def test_the_assistant_stays_outside_the_module_tree(base: torch.nn.Module, assistant_dir: str) -> None:
    model = _model(assistant_dir, base)
    technique = model.techniques["gemma4_assistant"]
    assert technique.state_dict() == {} and not list(technique.parameters())
    # the last non-shared layer of each type stores the entries: layers 1-4 of the tiny model are s, s, f, s
    assert technique.context == {"full_attention": 3, "sliding_attention": 4}
    drafter = technique.drafter(None)
    assert isinstance(drafter, Gemma4AssistantDrafter)
    assert drafter.footprint == LayerFootprint(1, ASSISTANT_LAYERS, MODULE) and drafter.layer_span == (1, 0)
    technique.detach()
    with pytest.raises(RuntimeError):
        technique.assistant  # noqa: B018


def test_attach_checks_that_the_assistant_belongs_to_the_model(base: torch.nn.Module, tmp_path) -> None:
    wide = tmp_path / "wide"
    make_assistant(base, backbone_hidden_size=2 * base.config.hidden_size).save_pretrained(wide)
    heads = tmp_path / "heads"
    make_assistant(base, text=dict(num_attention_heads=2)).save_pretrained(heads)
    for path in (wide, heads):
        with pytest.raises(ConfigError):
            _model(str(path), base)
    with pytest.raises(UnsupportedModelError):
        _model(str(tmp_path / "wide"), make_tiny_model())


# ---- the drafts are those of transformers --------------------------------------------------------------------------
@torch.no_grad()
def test_a_cycle_drafts_what_transformers_drafts(base: torch.nn.Module, assistant_dir: str,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """The first cycle after the prompt, against ``SinglePositionMultiTokenCandidateGenerator``'s loop: the base's
    final normalized state and the bonus token, every step at the bonus position, argmax tokens and projected states
    fed back (after the prompt the shared entries are exactly the committed ones, as in ``transformers``)."""
    from transformers import AutoModelForCausalLM

    steps = 5
    assistant = AutoModelForCausalLM.from_pretrained(assistant_dir, dtype=torch.float32).eval()
    seen: list[torch.Tensor] = []
    step = Gemma4AssistantDrafter.step

    def recorded(self: Gemma4AssistantDrafter, *args: Any) -> torch.Tensor:
        seen.append(step(self, *args))
        return seen[-1]

    monkeypatch.setattr(Gemma4AssistantDrafter, "step", recorded)
    for prompt in _prompts(lengths=(PROMPT_LEN, 3, 20)):  # 3 < WINDOW < 12 < 20: both sides of the window
        out = base.model(prompt[None], output_hidden_states=True, return_shared_kv_states=True)
        token = base.lm_head(out.last_hidden_state[:, -1:]).argmax(-1)
        state, shared = out.hidden_states[-1][:, -1:], out.shared_kv_states
        position = torch.tensor([[prompt.numel()]])
        expected = []
        for _ in range(steps):
            draft = assistant(inputs_embeds=torch.cat([base.get_input_embeddings()(token), state], -1),
                              position_ids=position, shared_kv_states=shared)
            expected.append(draft.logits[0, -1])
            token, state = draft.logits.argmax(-1), draft.last_hidden_state

        seen.clear()
        _model(assistant_dir, base, cycle_length=steps + 1).decode(prompt, 2)
        assert len(seen) >= steps
        for got, want in zip(seen[:steps], expected):
            torch.testing.assert_close(got[0], want, atol=1e-4, rtol=1e-4)


# ---- lossless decoding ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cycle_length", [1, 3, 6])
def test_greedy_with_the_assistant_equals_plain_greedy(base: torch.nn.Module, assistant_dir: str,
                                                       cycle_length: int) -> None:
    model = _model(assistant_dir, base, cycle_length=cycle_length)
    prompts = _prompts()
    new = [30, 21, 26]
    results = model.decode_batch(prompts, new)
    for prompt, result, n in zip(prompts, results, new):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, n))
    if cycle_length > 1:
        assert sum(k for r in results for k, _ in r.acceptance) > 0


def test_dynamic_draft_length_is_lossless(base: torch.nn.Module, assistant_dir: str) -> None:
    model = _model(assistant_dir, base)
    config = DecodeConfig(cycle_length=6, min_confidence=0.6)
    prompts = _prompts(seed=2)
    for prompt, result in zip(prompts, model.decode_batch(prompts, 24, decoding=config)):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 24))


def test_cost_log_names_the_assistant(base: torch.nn.Module, assistant_dir: str) -> None:
    result = _model(assistant_dir, base, cycle_length=4).decode(_prompts()[0], 20)
    drafts = [e for e in result.cost.events if e.kind in ("draft", "draft_wasted")]
    assert drafts and len(drafts) == sum(k for k, _ in result.acceptance)
    assert all((e.tokens, e.first_layer, e.last_layer, e.module) == (1, 1, ASSISTANT_LAYERS, MODULE) for e in drafts)
    assert {e.kind for e in result.cost.events} <= {"exact", "draft", "draft_wasted"}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@torch.no_grad()
def test_graphed_assistant_steps_are_greedy_and_follow_the_batch(base: torch.nn.Module, assistant_dir: str) -> None:
    gpu = make_family_model("gemma4_text").cuda()
    model = _model(assistant_dir, gpu)
    prompts = [p.cuda() for p in _prompts()]
    decoding = DecodeConfig(cycle_length=4, cuda_graphs=True)
    for limits in ([24, 24, 24], [24, 3, 11], [60, 24, 24]):
        results = model.decode_batch(prompts, limits, decoding=decoding)
        for prompt, result, n in zip(prompts, results, limits):
            assert torch.equal(result.tokens.cpu(), reference_greedy(gpu, prompt, n).cpu())
    drafter = model.decoder(decoding).active_drafter
    assert isinstance(drafter, Gemma4AssistantDrafter) and drafter._step_fn.enabled
