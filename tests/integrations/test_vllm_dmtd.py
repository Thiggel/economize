"""vLLM serving of DMTD drafters through the jump plugin (CPU, vLLM-style fake layers).

A DMTD drafter with ``L_d`` decoding layers is exported as a jump drafter with ``m = u = L - L_d``, ``r = L`` and the
``"embedding"`` predictor; the vLLM draft model must then draft exactly as
:class:`~economize.techniques.dynamic_depth.DmtdTechnique`: the late layers on the next token's embedding, then the
final norm and the LM head.
"""

from __future__ import annotations

import importlib.util
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
from economize.backends.vllm.jump.model import EconomizeJumpEagle3ForCausalLM
from economize.backends.vllm.serving import check_servable, support_table, to_vllm
from economize.techniques.dynamic_depth import DmtdConfig
from economize.techniques.dynamic_depth.dmtd_export import jump_config
from economize.techniques.jump import JumpConfig
from economize.techniques.jump.export import CONFIG_KEY, JumpDrafterSpec
from economize.techniques.quant.config import QuantConfig
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, perturb

LAYERS = 6
DECODING = 2
M = LAYERS - DECODING


def _dmtd(adaptation: str = "full", seed: int = 0, cycle_length: int = 3) -> tuple[Any, Any]:
    base = make_tiny_model(seed=seed, num_layers=LAYERS)
    model = economize.apply(base, DmtdConfig(decoding_layers=DECODING, cycle_length=cycle_length,
                                             adaptation=adaptation))
    perturb(model.techniques["dmtd"], std=0.05, seed=seed + 1)
    return base, model


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
    """DMTD's draft logits for ``T`` one-token sequences: layers ``m+1..L`` on the embedding, then the head."""
    adapter = technique.adapter
    hidden = adapter.embed(ids)[:, None]
    embeddings = adapter.rotary_emb(hidden, torch.zeros(len(ids), 1, dtype=torch.long))
    overrides = technique.layer_overrides()
    with technique.jump_path():
        for layer in range(M + 1, LAYERS + 1):
            out = overrides.get(layer, adapter.layer(layer))(hidden, attention_mask=None,
                                                             position_embeddings=embeddings)
            hidden = out[0] if isinstance(out, tuple) else out
    return technique.logits(hidden[:, 0])


# ---- the export format ----------------------------------------------------------------------------------------------
def test_jump_config_drops_the_cycle_fields() -> None:
    _, model = _dmtd()
    config = jump_config(model.techniques["dmtd"].config)
    assert type(config) is JumpConfig
    assert (config.target_layer, config.source_layer, config.exit_layer) == (M, M, None)
    assert config.predictor == "embedding" and config.adaptation == "full"


@pytest.mark.parametrize("adaptation", ["frozen", "lora", "full"])
def test_export_is_a_jump_drafter_at_the_late_layers(tmp_path: Path, adaptation: str) -> None:
    _, model = _dmtd(adaptation)
    technique = model.techniques["dmtd"]
    folder = export_drafter(model, tmp_path / "dmtd")
    config = json.loads((folder / "config.json").read_text())
    assert config["architectures"] == ["EconomizeJumpEagle3"]
    block = config[CONFIG_KEY]
    assert (block["target_layer"], block["exit_layer"], block["jump_config"]["source_layer"]) == (M, LAYERS, M)
    assert not {"decoding_layers", "cycle_length"} & set(block["jump_config"])
    assert block["jump_config"]["predictor"] == "embedding" and config["eagle_aux_hidden_state_layer_ids"] == [M]
    spec = JumpDrafterSpec.from_hf_config(config)
    assert list(spec.jump_layers) == list(range(M + 1, LAYERS + 1))
    tensors = load_file(str(folder / "model.safetensors"))
    assert set(tensors) == {"jump." + name for name in technique.state_dict()}
    drafter = drafter_of(_vllm_config(folder))
    assert drafter is not None and drafter.technique == "jump"


# ---- the vLLM draft model -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("adaptation", ["frozen", "lora", "full"])
def test_vllm_draft_step_matches_dmtd(tmp_path: Path, adaptation: str) -> None:
    base, model = _dmtd(adaptation)
    technique = model.techniques["dmtd"]
    target = vllm_target(base)
    weights = {name: p.clone() for name, p in target.model.named_parameters()}
    draft = _tied(export_drafter(model, tmp_path / adaptation), target)
    late = target.model.layers[M:]
    if adaptation == "frozen":
        assert all(a is b for a, b in zip(draft.jump_path, late, strict=True))
    else:
        assert all(a is not b for a, b in zip(draft.jump_path, late, strict=True))

    generator = torch.Generator().manual_seed(3)
    ids = torch.randint(0, VOCAB, (PROMPT_LEN,), generator=generator)
    aux = torch.randn(PROMPT_LEN, technique.adapter.hidden_size, generator=generator)  # ignored by the predictor
    with torch.no_grad():
        hidden, _ = draft.draft_forward(ids, torch.arange(len(ids)), aux)
        logits = draft.compute_logits(hidden)
    torch.testing.assert_close(logits, _reference_step(technique, ids), rtol=1e-4, atol=1e-4)
    assert [len(layer.self_attn.attn.kv_cache) for layer in target.model.layers] == [0] * M + [1] * DECODING
    assert all(torch.equal(p, weights[name]) for name, p in target.model.named_parameters())


# ---- the serving plan -----------------------------------------------------------------------------------------------
needs_ct = pytest.mark.skipif(importlib.util.find_spec("compressed_tensors") is None, reason="needs compressed-tensors")


@pytest.fixture
def tiny_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "tiny"
    make_tiny_model(seed=0, num_layers=LAYERS).save_pretrained(folder)
    return folder


def _base(folder: Path) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32).eval()


def test_plan_serves_dmtd_on_the_plugin(tiny_dir: Path, tmp_path: Path) -> None:
    model = economize.apply(_base(tiny_dir), DmtdConfig(decoding_layers=DECODING, cycle_length=4))
    plan = to_vllm(model, tmp_path / "out", num_speculative_tokens=3)
    assert plan.techniques == {"dmtd": "plugin"}
    assert plan.speculative is not None and plan.speculative["method"] == "eagle3"
    assert any("num_speculative_tokens=3" in note and "HF backend only" in note for note in plan.notes["dmtd"])
    drafter = json.loads((tmp_path / "out" / "drafter" / "config.json").read_text())
    assert drafter["architectures"] == ["EconomizeJumpEagle3"] and drafter[CONFIG_KEY]["target_layer"] == M
    assert support_table(["dmtd"])[0][2] == "plugin"


@needs_ct
@pytest.mark.parametrize("adaptation", ["frozen", "lora"])
def test_compressed_tensors_base(tiny_dir: Path, tmp_path: Path, adaptation: str) -> None:
    model = economize.apply(_base(tiny_dir), QuantConfig(format="int4", group_size=32, backend="compressed-tensors"),
                            DmtdConfig(decoding_layers=DECODING, adaptation=adaptation))
    if adaptation == "frozen":
        assert to_vllm(model, tmp_path / "out").techniques == {"quant": "native", "dmtd": "plugin"}
        return
    with pytest.raises(UnsupportedBackendError, match="adaptation='lora'"):
        check_servable(model)
