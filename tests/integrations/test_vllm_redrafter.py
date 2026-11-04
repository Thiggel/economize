"""vLLM serving of ReDrafter drafters through economize's ReDrafter proposer (CPU, vLLM-style fake layers).

The vLLM draft model must draft exactly as :class:`~economize.techniques.redrafter.RedrafterTechnique`'s chain: the
context of the last verified token (the final norm with ``head="own"``), the recurrent state started at zero and
advanced on the drafted tokens, and the head on ``[h_t; s_k]`` with its own projection or the target's final norm and
LM head, top-k warped.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import load_file
from vllm_fakes import vllm_target

import economize
from economize.backends import UnsupportedBackendError
from economize.backends.vllm import export_drafter
from economize.backends.vllm.drafters import drafter_of
from economize.backends.vllm.redrafter.model import EconomizeRedrafterEagle3ForCausalLM
from economize.backends.vllm.serving import support_table, to_vllm
from economize.errors import ConfigError
from economize.techniques.redrafter import RedrafterConfig
from economize.techniques.redrafter.export import CONFIG_KEY, RedrafterDrafterSpec
from helpers import VOCAB, make_tiny_model

LAYERS = 4
DRAFTS = 3
#: (head, exit_dim, top_k): the plain head, an input projection, the shared readout, no warp.
CASES = [("own", None, 50), ("own", 96, 50), ("shared", None, 50), ("shared", 160, 0)]


def _redrafter(head: str = "own", exit_dim: int | None = None, top_k: int = 50, seed: int = 0,
               **config: Any) -> tuple[Any, Any]:
    base = make_tiny_model(seed=seed, num_layers=LAYERS)
    model = economize.apply(base, RedrafterConfig(beam_length=DRAFTS + 1, num_layers=2, head=head,
                                                  exit_dim=exit_dim, top_k=top_k, **config))
    generator = torch.Generator().manual_seed(seed + 7)
    with torch.no_grad():  # a trained-looking head (zero-init blocks would be the identity)
        for param in model.techniques["redrafter"].parameters():
            param.add_(0.1 * torch.randn(param.shape, generator=generator))
    return base, model


def _vllm_config(folder: Path) -> SimpleNamespace:
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(folder)
    draft = SimpleNamespace(hf_config=hf_config, architectures=list(hf_config.architectures))
    return SimpleNamespace(speculative_config=SimpleNamespace(draft_model_config=draft))


def _tied(folder: Path, target: SimpleNamespace) -> EconomizeRedrafterEagle3ForCausalLM:
    draft = EconomizeRedrafterEagle3ForCausalLM(vllm_config=_vllm_config(folder))
    loaded = draft.load_weights(load_file(str(folder / "model.safetensors")).items())
    assert loaded >= {name for name, _ in draft.named_parameters()}  # what vLLM's loader checks
    draft.tie_to_target(target)
    return draft


# ---- the export format ----------------------------------------------------------------------------------------------
def test_export_holds_the_recurrence_and_the_head(tmp_path: Path) -> None:
    _, model = _redrafter()
    technique = model.techniques["redrafter"]
    folder = export_drafter(model, tmp_path / "rd")
    config = json.loads((folder / "config.json").read_text())
    assert config["architectures"] == ["EconomizeRedrafterEagle3"]
    assert config["eagle_aux_hidden_state_layer_ids"] == [LAYERS]
    spec = RedrafterDrafterSpec.from_hf_config(config)
    assert spec == RedrafterDrafterSpec.of(technique)
    assert (spec.num_drafts, spec.head_layers, spec.width, spec.top_k) == (DRAFTS, 2, 128, 50)
    assert not spec.input_projection
    assert {"model.norm", "model.embed_tokens", "lm_head"} <= set(config[CONFIG_KEY]["shared_with_target"])
    tensors = load_file(str(folder / "model.safetensors"))
    expected = {f"redrafter.{name}": p for name, p in technique.state_dict().items()}
    assert tensors.keys() == expected.keys() and "redrafter.rnn_u.bias" in tensors
    assert all(torch.equal(tensors[name], expected[name].float()) for name in tensors)
    drafter = drafter_of(_vllm_config(folder))
    assert drafter is not None and drafter.technique == "redrafter"


def test_spec_round_trips_and_validates() -> None:
    spec = RedrafterDrafterSpec(num_layers=4, hidden_size=64, vocab_size=VOCAB, width=96, head_layers=1,
                                head="shared", param_dtype="float32", top_k=0, num_drafts=4)
    assert RedrafterDrafterSpec.from_dict(spec.to_dict()) == spec and spec.input_projection
    with pytest.raises(ConfigError, match="format version"):
        RedrafterDrafterSpec.from_dict(spec.to_dict() | {"format_version": 99})
    with pytest.raises(ConfigError, match="head must be"):
        RedrafterDrafterSpec.from_dict(spec.to_dict() | {"head": "tied"})
    with pytest.raises(ConfigError, match="num_drafts >= 1"):
        RedrafterDrafterSpec.from_dict(spec.to_dict() | {"num_drafts": 0})


def test_load_weights_refuses_foreign_tensors(tmp_path: Path) -> None:
    _, model = _redrafter()
    draft = EconomizeRedrafterEagle3ForCausalLM(vllm_config=_vllm_config(export_drafter(model, tmp_path / "rd")))
    with pytest.raises(ConfigError, match="not a ReDrafter drafter tensor"):
        draft.load_weights([("hydra.heads.0.proj.weight", torch.zeros(1))])
    with pytest.raises(ConfigError, match="not a tensor of the exported"):
        draft.load_weights([("redrafter.blocks.9.linear.weight", torch.zeros(1))])


def test_tensor_parallel_replicates_the_drafter(tmp_path: Path) -> None:
    _, model = _redrafter()
    folder = export_drafter(model, tmp_path / "rd")
    config = _vllm_config(folder)
    config.parallel_config = SimpleNamespace(tensor_parallel_size=2)
    draft = EconomizeRedrafterEagle3ForCausalLM(vllm_config=config)
    draft.load_weights(load_file(str(folder / "model.safetensors")).items())
    single = _tied(folder, vllm_target(_redrafter()[0]))
    assert dict(draft.named_parameters()).keys() == dict(single.named_parameters()).keys()
    assert all(torch.equal(p, dict(single.named_parameters())[name]) for name, p in draft.named_parameters())


# ---- the vLLM draft model -------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("head", "exit_dim", "top_k"), CASES)
def test_chain_drafts_as_the_technique(tmp_path: Path, head: str, exit_dim: int | None, top_k: int) -> None:
    base, model = _redrafter(head, exit_dim, top_k)
    technique = model.techniques["redrafter"]
    draft = _tied(export_drafter(model, tmp_path / "rd"), vllm_target(base))
    assert draft.attention_layer_names == [] and draft.spec.input_projection == (exit_dim is not None)

    generator = torch.Generator().manual_seed(3)
    rows = 5
    hidden = torch.randn(rows, technique.adapter.hidden_size, generator=generator)  # the target's last states
    tokens = torch.randint(0, VOCAB, (rows,), generator=generator)  # x_{t+1}
    with torch.no_grad():
        context, expected_context = draft.base(hidden), technique.base(hidden)
        torch.testing.assert_close(context, expected_context)
        state = expected_state = torch.zeros_like(context)
        for _ in range(DRAFTS):
            state = draft.recur(state, tokens)
            logits = draft.compute_logits(draft.head_hidden(context, state))
            expected, expected_state = technique.draft_step(expected_context, expected_state, tokens)
            assert logits.dtype == torch.float32
            torch.testing.assert_close(state, expected_state, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4)
            if top_k:
                assert int((logits > -1e4).sum(-1).max()) <= top_k
            tokens = logits.argmax(-1)


# ---- the serving plan -----------------------------------------------------------------------------------------------
@pytest.fixture
def tiny_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "tiny"
    make_tiny_model(seed=0, num_layers=LAYERS).save_pretrained(folder)
    return folder


def _loaded(folder: Path, **config: Any) -> Any:
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32).eval()
    return economize.apply(base, RedrafterConfig(beam_length=DRAFTS + 1, num_layers=1, **config))


def test_plan_serves_redrafter_on_the_plugin(tiny_dir: Path, tmp_path: Path) -> None:
    plan = to_vllm(_loaded(tiny_dir, beam_width=4), tmp_path / "out", num_speculative_tokens=DRAFTS)
    assert plan.techniques == {"redrafter": "plugin"}
    assert plan.speculative is not None and plan.speculative["method"] == "eagle3"
    assert any("beam search over 4 beams" in note for note in plan.notes["redrafter"])
    assert any("top-50 warp" in note for note in plan.notes["redrafter"])
    drafter = json.loads((tmp_path / "out" / "drafter" / "config.json").read_text())
    assert drafter["architectures"] == ["EconomizeRedrafterEagle3"] and CONFIG_KEY in drafter
    assert support_table(["redrafter"])[0][2] == "plugin"


def test_plan_refuses_more_draft_tokens_than_the_beam_length(tiny_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(UnsupportedBackendError, match=f"at most {DRAFTS} tokens"):
        to_vllm(_loaded(tiny_dir), tmp_path / "out", num_speculative_tokens=DRAFTS + 1)
