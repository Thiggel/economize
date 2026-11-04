"""vLLM serving of early-exit drafters through the jump plugin (CPU, vLLM-style fake layers).

An early exit after layer ``r`` is exported as a jump drafter with ``m = u = 0`` and the ``"embedding"`` predictor;
the vLLM draft model must then draft exactly as :class:`~economize.techniques.early_exit.EarlyExitTechnique`.
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
from economize.backends.vllm.jump.model import EconomizeJumpEagle3ForCausalLM
from economize.backends.vllm.serving import check_servable, to_vllm
from economize.techniques.early_exit import EarlyExitConfig
from economize.techniques.early_exit.export import jump_state_dict
from economize.techniques.jump.export import CONFIG_KEY, ExportedJumpDrafter, JumpDrafterSpec
from economize.techniques.kv_quant.fp8 import FP8KVConfig
from economize.techniques.quant.config import DraftQuantConfig, QuantConfig
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, perturb

EXIT = 3


def _early_exit(adaptation: str = "frozen", head: str = "readout", seed: int = 0) -> tuple[Any, Any]:
    base = make_tiny_model(seed=seed)
    model = economize.apply(base, EarlyExitConfig(exit_layer=EXIT, head=head, adaptation=adaptation))
    perturb(model.techniques["early_exit"], std=0.05, seed=seed + 1)
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
    """The technique's draft logits for ``T`` one-token sequences: layers ``1..r`` on the embedding, then the head."""
    adapter = technique.adapter
    hidden = adapter.embed(ids)[:, None]
    embeddings = adapter.rotary_emb(hidden, torch.zeros(len(ids), 1, dtype=torch.long))
    overrides = technique.path.overrides()
    with technique.path.active():
        for layer in range(1, technique.exit_layer + 1):
            out = overrides.get(layer, adapter.layer(layer))(hidden, attention_mask=None,
                                                             position_embeddings=embeddings)
            hidden = out[0] if isinstance(out, tuple) else out
    return technique.logits(hidden[:, 0])


def test_export_is_a_jump_drafter_at_layer_zero(tmp_path: Path) -> None:
    _, model = _early_exit()
    folder = export_drafter(model, tmp_path / "ee")
    config = json.loads((folder / "config.json").read_text())
    block = config[CONFIG_KEY]
    assert (block["target_layer"], block["exit_layer"], block["jump_config"]["source_layer"]) == (0, EXIT, 0)
    assert block["jump_config"]["predictor"] == "embedding" and config["eagle_aux_hidden_state_layer_ids"] == [0]
    spec = JumpDrafterSpec.from_hf_config(config)
    assert list(spec.jump_layers) == [1, 2, 3]
    tensors = load_file(str(folder / "model.safetensors"))
    technique = model.techniques["early_exit"]
    assert set(tensors) == {"jump." + name.removeprefix("path.") for name in technique.state_dict()}


def test_shared_head_exports_an_identity_readout(tmp_path: Path) -> None:
    _, model = _early_exit(head="shared")
    drafter = ExportedJumpDrafter.from_pretrained(export_drafter(model, tmp_path / "shared"))
    hidden = torch.randn(PROMPT_LEN, model.techniques["early_exit"].adapter.hidden_size)
    with torch.no_grad():
        torch.testing.assert_close(drafter.readout_hidden(hidden), hidden, rtol=0, atol=0)
        torch.testing.assert_close(drafter.predict([hidden], hidden * 2), hidden * 2, rtol=0, atol=0)


@pytest.mark.parametrize("head", ["readout", "shared"])
@pytest.mark.parametrize("adaptation", ["frozen", "lora", "full"])
def test_vllm_draft_step_matches_early_exit(tmp_path: Path, adaptation: str, head: str) -> None:
    base, model = _early_exit(adaptation, head)
    technique = model.techniques["early_exit"]
    target = vllm_target(base)
    weights = {name: p.clone() for name, p in target.model.named_parameters()}
    draft = _tied(export_drafter(model, tmp_path / f"{adaptation}_{head}"), target)
    if adaptation == "frozen":
        assert all(a is b for a, b in zip(draft.jump_path, target.model.layers[:EXIT]))
    else:
        assert all(a is not b for a, b in zip(draft.jump_path, target.model.layers[:EXIT]))

    generator = torch.Generator().manual_seed(3)
    ids = torch.randint(0, VOCAB, (PROMPT_LEN,), generator=generator)
    aux = torch.randn(PROMPT_LEN, technique.adapter.hidden_size, generator=generator)  # ignored by the predictor
    with torch.no_grad():
        hidden, _ = draft.draft_forward(ids, torch.arange(len(ids)), aux)
        logits = draft.compute_logits(hidden)
    torch.testing.assert_close(logits, _reference_step(technique, ids), rtol=1e-4, atol=1e-4)
    assert [len(layer.self_attn.attn.kv_cache) for layer in target.model.layers] == [1] * EXIT + [0] * 3
    assert all(torch.equal(p, weights[name]) for name, p in target.model.named_parameters())


# ---- the serving plan -----------------------------------------------------------------------------------------------
needs_ct = pytest.mark.skipif(importlib.util.find_spec("compressed_tensors") is None, reason="needs compressed-tensors")


@pytest.fixture
def tiny_dir(tmp_path: Path) -> Path:
    """A saved tiny model, so the base has a ``name_or_path`` to serve."""
    folder = tmp_path / "tiny"
    make_tiny_model(seed=0).save_pretrained(folder)
    return folder


def _base(folder: Path) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32).eval()


def test_plan_serves_early_exit_on_the_plugin(tiny_dir: Path, tmp_path: Path) -> None:
    """fp8 KV cache + early-exit drafter + its layer copies in int8 in one plan, served on the jump plugin."""
    model = economize.apply(_base(tiny_dir), FP8KVConfig(), EarlyExitConfig(exit_layer=EXIT, adaptation="full"),
                            DraftQuantConfig(format="int8"))
    plan = to_vllm(model, tmp_path / "out", num_speculative_tokens=3)
    assert plan.techniques == {"kv_fp8": "native", "early_exit": "plugin", "draft_quant": "densified"}
    assert plan.model == str(tiny_dir) and plan.engine["kv_cache_dtype"] == "fp8_e4m3"
    assert plan.speculative is not None and plan.speculative["method"] == "eagle3"
    assert plan.speculative["model"] == str(tmp_path / "out" / "drafter")
    drafter = json.loads((tmp_path / "out" / "drafter" / "config.json").read_text())
    assert drafter["architectures"] == ["EconomizeJumpEagle3"]
    tensors = load_file(str(tmp_path / "out" / "drafter" / "model.safetensors"))
    state = jump_state_dict(model.techniques["early_exit"])
    assert set(tensors) == {"jump." + name for name in state}
    for name, value in state.items():
        torch.testing.assert_close(tensors["jump." + name], value.float())


@needs_ct
@pytest.mark.parametrize("adaptation", ["frozen", "lora"])
def test_compressed_tensors_base(tiny_dir: Path, tmp_path: Path, adaptation: str) -> None:
    """A frozen path shares the quantized target layers; an adapted one would need dense copies of them."""
    model = economize.apply(_base(tiny_dir), QuantConfig(format="int4", group_size=32, backend="compressed-tensors"),
                            EarlyExitConfig(exit_layer=EXIT, adaptation=adaptation))
    if adaptation == "frozen":
        assert to_vllm(model, tmp_path / "out").techniques == {"quant": "native", "early_exit": "plugin"}
        return
    with pytest.raises(UnsupportedBackendError, match="adaptation='lora'"):
        check_servable(model)
