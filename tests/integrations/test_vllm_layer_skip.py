"""vLLM serving of layer-skip drafters through the jump plugin (CPU, vLLM-style fake layers).

A layer-skip drafter is exported as a jump drafter with ``m = u = 0``, the ``"embedding"`` predictor and the skip set
as holes in the path; the vLLM draft model must then draft exactly as
:class:`~economize.techniques.layer_skip.LayerSkipTechnique`, and write KV entries only in the layers it runs.
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
from economize.backends.vllm.jump.model import EconomizeJumpEagle3ForCausalLM
from economize.backends.vllm.serving import support_table, to_vllm
from economize.errors import ConfigError
from economize.techniques.jump.config import JumpConfig
from economize.techniques.jump.export import CONFIG_KEY, ExportedJumpDrafter, JumpDrafterSpec
from economize.techniques.kv_quant.fp8 import FP8KVConfig
from economize.techniques.layer_skip import LayerSkipConfig
from helpers import PROMPT_LEN, VOCAB, make_tiny_model

LAYERS = 6


def _layer_skip(skip: tuple[int, ...], seed: int = 0) -> tuple[Any, Any]:
    base = make_tiny_model(seed=seed, num_layers=LAYERS)
    return base, economize.apply(base, LayerSkipConfig(skip_layers=skip))


def _vllm_config(folder: Path) -> SimpleNamespace:
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(folder)
    draft = SimpleNamespace(hf_config=hf_config, architectures=list(hf_config.architectures))
    return SimpleNamespace(speculative_config=SimpleNamespace(draft_model_config=draft))


def _tied(folder: Path, target: SimpleNamespace) -> EconomizeJumpEagle3ForCausalLM:
    draft = EconomizeJumpEagle3ForCausalLM(vllm_config=_vllm_config(folder))
    loaded = draft.load_weights(load_file(str(folder / "model.safetensors")).items())
    assert loaded >= {name for name, _ in draft.named_parameters()}  # what vLLM's loader checks
    draft.tie_to_target(target)
    return draft


@torch.no_grad()
def _reference_step(technique: Any, ids: torch.Tensor) -> torch.Tensor:
    """The technique's draft logits for ``T`` one-token sequences: the run layers on the embedding, then the head."""
    adapter = technique.adapter
    hidden = adapter.embed(ids)[:, None]
    embeddings = adapter.rotary_emb(hidden, torch.zeros(len(ids), 1, dtype=torch.long))
    for layer in technique.run_layers:
        out = adapter.layer(layer)(hidden, attention_mask=None, position_embeddings=embeddings)
        hidden = out[0] if isinstance(out, tuple) else out
    return technique.logits(hidden[:, 0])


# ---- the export format ----------------------------------------------------------------------------------------------
def test_spec_with_holes_round_trips() -> None:
    config = JumpConfig(target_layer=0, source_layer=0, exit_layer=5, predictor="embedding", adaptation="frozen")
    spec = JumpDrafterSpec(config, LAYERS, 16, skip_layers=(4, 2, 4))
    assert spec.skip_layers == (2, 4) and spec.jump_layers == (1, 3, 5)
    block = spec.to_dict()
    assert block["format_version"] == 2 and block["skip_layers"] == [2, 4]
    assert JumpDrafterSpec.from_dict(block) == spec
    plain = JumpDrafterSpec(config, LAYERS, 16)
    assert plain.to_dict()["format_version"] == 1 and "skip_layers" not in plain.to_dict()
    assert plain.jump_layers == (1, 2, 3, 4, 5)


@pytest.mark.parametrize("skip", [(6,), (0,), (1, 2, 3, 4, 5)])
def test_spec_refuses_holes_outside_or_covering_the_path(skip: tuple[int, ...]) -> None:
    config = JumpConfig(target_layer=0, source_layer=0, exit_layer=5, predictor="embedding", adaptation="frozen")
    with pytest.raises(ConfigError, match="skip_layers"):
        JumpDrafterSpec(config, LAYERS, 16, skip_layers=skip)


def test_export_is_a_jump_drafter_with_holes(tmp_path: Path) -> None:
    _, model = _layer_skip((2, 4))
    folder = export_drafter(model, tmp_path / "ls")
    config = json.loads((folder / "config.json").read_text())
    block = config[CONFIG_KEY]
    assert (block["target_layer"], block["exit_layer"], block["skip_layers"]) == (0, LAYERS, [2, 4])
    assert block["jump_config"]["predictor"] == "embedding" and config["eagle_aux_hidden_state_layer_ids"] == [0]
    assert JumpDrafterSpec.from_hf_config(config).jump_layers == (1, 3, 5, 6)
    assert load_file(str(folder / "model.safetensors")) == {}  # r = L: no readout, nothing owned
    assert ExportedJumpDrafter.from_pretrained(folder).readout is None


def test_skipped_last_layers_export_an_identity_readout(tmp_path: Path) -> None:
    _, model = _layer_skip((3, 5, 6))
    folder = export_drafter(model, tmp_path / "ls")
    spec = JumpDrafterSpec.from_hf_config(json.loads((folder / "config.json").read_text()))
    assert (spec.exit_layer, spec.skip_layers, spec.jump_layers) == (4, (3,), (1, 2, 4))
    drafter = ExportedJumpDrafter.from_pretrained(folder)
    hidden = torch.randn(PROMPT_LEN, model.techniques["layer_skip"].adapter.hidden_size)
    with torch.no_grad():
        torch.testing.assert_close(drafter.readout_hidden(hidden), hidden, rtol=0, atol=0)


# ---- the vLLM draft model -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("skip", [(2, 4), (3, 5, 6), (1,)])
def test_vllm_draft_step_matches_layer_skip(tmp_path: Path, skip: tuple[int, ...]) -> None:
    base, model = _layer_skip(skip)
    technique = model.techniques["layer_skip"]
    target = vllm_target(base)
    weights = {name: p.clone() for name, p in target.model.named_parameters()}
    draft = _tied(export_drafter(model, tmp_path / "ls"), target)
    run = technique.run_layers
    assert all(a is target.model.layers[i - 1] for a, i in zip(draft.jump_path, run, strict=True))

    generator = torch.Generator().manual_seed(3)
    ids = torch.randint(0, VOCAB, (PROMPT_LEN,), generator=generator)
    aux = torch.randn(PROMPT_LEN, technique.adapter.hidden_size, generator=generator)  # ignored by the predictor
    with torch.no_grad():
        hidden, _ = draft.draft_forward(ids, torch.arange(len(ids)), aux)
        logits = draft.compute_logits(hidden)
    torch.testing.assert_close(logits, _reference_step(technique, ids), rtol=1e-4, atol=1e-4)
    written = [len(layer.self_attn.attn.kv_cache) for layer in target.model.layers]
    assert written == [int(i in run) for i in range(1, LAYERS + 1)]  # skipped layers write no KV, as in HF
    assert all(torch.equal(p, weights[name]) for name, p in target.model.named_parameters())


# ---- the serving plan -----------------------------------------------------------------------------------------------
@pytest.fixture
def tiny_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "tiny"
    make_tiny_model(seed=0, num_layers=LAYERS).save_pretrained(folder)
    return folder


def test_plan_serves_layer_skip_on_the_plugin(tiny_dir: Path, tmp_path: Path) -> None:
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(tiny_dir, dtype=torch.float32).eval()
    model = economize.apply(base, FP8KVConfig(), LayerSkipConfig(skip_layers=(2, 5)))
    plan = to_vllm(model, tmp_path / "out", num_speculative_tokens=3)
    assert plan.techniques == {"kv_fp8": "native", "layer_skip": "plugin"}
    assert plan.speculative is not None and plan.speculative["method"] == "eagle3"
    drafter = json.loads((tmp_path / "out" / "drafter" / "config.json").read_text())
    assert drafter["architectures"] == ["EconomizeJumpEagle3"] and drafter[CONFIG_KEY]["skip_layers"] == [2, 5]
    assert support_table(["layer_skip"])[0][2] == "plugin"


def test_plan_refuses_a_layer_skip_without_a_skip_set(tiny_dir: Path, tmp_path: Path) -> None:
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(tiny_dir, dtype=torch.float32).eval()
    model = economize.apply(base, LayerSkipConfig(skip_layers=(3,)))
    model.techniques["layer_skip"].skip_mask.zero_()
    with pytest.raises(UnsupportedBackendError, match="no skip set"):
        to_vllm(model, tmp_path / "out")
