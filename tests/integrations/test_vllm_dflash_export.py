"""The ``"vllm/dflash"`` exporter, checked on CPU: the folder is the z-lab (or ``speculators``) DFlash layout vLLM's
DFlash proposer loads, and reading it back with economize's own DFlash readers gives the exported drafter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

import economize
from economize.backends import export_drafter, exporter_for
from economize.backends.vllm.dflash import DFlashVllmExporter
from economize.backends.vllm.dflash.export import layout_problem
from economize.core.serialization import read_checkpoint
from economize.errors import ConfigError
from economize.techniques.dflash import DFlashConfig
from helpers import make_tiny_model, perturb

SMALL = dict(num_layers=3, intermediate_size=48, num_heads=4, num_kv_heads=2, head_dim=16, rope_theta=1e4,
             mask_token_id=4, sliding_window=4, block_size=6, target_layers=(1, 3, 4))
MIXED = ("sliding_attention", "sliding_attention", "full_attention")
SLIDING = ("sliding_attention",) * 3


def _model(**options: Any) -> Any:
    model = economize.apply(make_tiny_model(seed=0), DFlashConfig(**{**SMALL, **options}))
    perturb(model.techniques["dflash"], 0.05, seed=1)
    return model


def _same_state(model: Any, folder: Path, format: str) -> None:
    checkpoint = read_checkpoint(folder, format=format)
    technique = model.techniques["dflash"]
    assert checkpoint.configs[0].target_layers == technique.target_layers
    loaded = economize.load(make_tiny_model(seed=0), folder, format=format)
    for name, tensor in technique.state_dict().items():
        assert torch.equal(tensor, loaded.techniques["dflash"].state_dict()[name]), name


def test_the_registered_exporter_is_picked() -> None:
    assert isinstance(exporter_for(_model(), "vllm"), DFlashVllmExporter)


@pytest.mark.parametrize("options", [dict(layer_types=MIXED), dict(layer_types=MIXED, sliding_causal=False),
                                     dict(layer_types=("full_attention",) * 3, rope_theta=5e5),
                                     dict(embed_scale=2.5, logit_softcap=3.0)])
def test_zlab_layout_reads_back(options: dict[str, Any], tmp_path: Path) -> None:
    model = _model(**options)
    folder = export_drafter(model, tmp_path / "dflash", backend="vllm")
    config = json.loads((folder / "config.json").read_text())
    technique = model.techniques["dflash"]
    assert config["architectures"] == ["DFlashDraftModel"] and config["model_type"] == "qwen3"
    assert config["dflash_config"]["target_layer_ids"] == [0, 2, 3]  # the reference counts block outputs from 0
    assert config["dflash_config"]["mask_token_id"] == 4 and config["block_size"] == 6
    assert config["layer_types"] == list(technique.layer_types)
    assert config["use_sliding_window"] == ("sliding_attention" in technique.layer_types)
    assert ("is_causal" in config) == (not technique.config.sliding_causal)
    assert "rotary.cos" not in load_file(str(folder / "model.safetensors"))
    read = read_checkpoint(folder, format="dflash").configs[0]
    assert read.types() == technique.config.types()
    for field in ("block_size", "mask_token_id", "num_layers", "sliding_window", "sliding_causal", "num_heads",
                  "num_kv_heads", "head_dim", "intermediate_size", "rope_theta", "embed_scale"):
        assert getattr(read, field) == getattr(technique.config, field), field
    _same_state(model, folder, "dflash")


def test_a_scaled_embedding_is_written_as_the_drafters_own(tmp_path: Path) -> None:
    model = _model(embed_scale=2.5)
    folder = export_drafter(model, tmp_path / "dflash", backend="vllm")
    tensors = load_file(str(folder / "model.safetensors"))
    embedding = model.techniques["dflash"].adapter.embed_tokens.weight
    assert torch.allclose(tensors["embed_tokens.weight"], 2.5 * embedding)
    assert "embed_tokens.weight" not in load_file(str(export_drafter(_model(), tmp_path / "plain",
                                                                     backend="vllm") / "model.safetensors"))


def test_transformers_keeps_the_sliding_window(tmp_path: Path) -> None:
    from transformers import AutoConfig

    folder = export_drafter(_model(layer_types=MIXED), tmp_path / "dflash", backend="vllm")
    config = AutoConfig.from_pretrained(folder)
    assert config.sliding_window == 4 and list(config.layer_types) == list(MIXED)
    assert getattr(config, "is_causal", None) is None


@pytest.mark.parametrize("options", [dict(layer_types=SLIDING), dict(sliding_causal=False, layer_types=MIXED)])
def test_speculators_layout_reads_back(options: dict[str, Any], tmp_path: Path) -> None:
    model = _model(**options)
    folder = DFlashVllmExporter().export(model, tmp_path / "spec", format="speculators", verifier="org/target",
                                         num_speculative_tokens=3)
    config = json.loads((folder / "config.json").read_text())
    assert config["speculators_config"]["verifier"]["name_or_path"] == "org/target"
    assert config["speculators_config"]["proposal_methods"][0]["speculative_tokens"] == 3
    assert config["aux_hidden_state_layer_ids"] == [1, 3, 4]
    assert config["sliding_window_non_causal"] == (not model.techniques["dflash"].config.sliding_causal)
    checkpoint = read_checkpoint(folder, format="speculators_dflash")
    assert checkpoint.base_model == "org/target"
    assert checkpoint.configs[0].sliding_causal == model.techniques["dflash"].config.sliding_causal
    _same_state(model, folder, "speculators_dflash")


def test_layouts_that_cannot_hold_the_drafter_are_refused(tmp_path: Path) -> None:
    exporter = DFlashVllmExporter()
    mixed = _model(layer_types=MIXED)
    assert layout_problem(mixed.techniques["dflash"], "dflash") is None
    for model, match in ((mixed, "one causality"), (_model(embed_scale=2.0, layer_types=SLIDING), "embedding scale")):
        with pytest.raises(ConfigError, match=match):
            exporter.export(model, tmp_path / "x", format="speculators", verifier="org/target")
    with pytest.raises(ConfigError, match="unknown export format"):
        exporter.export(mixed, tmp_path / "x", format="eagle")
    with pytest.raises(ConfigError, match="verifier"):
        exporter.export(_model(layer_types=SLIDING), tmp_path / "x", format="speculators")
