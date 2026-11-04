"""The ``"vllm/medusa"`` exporter, checked on CPU: the folder holds vLLM's Medusa parameter names, vLLM's config
class reads it, and vLLM's head arithmetic on the exported tensors gives economize's head logits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

import economize
from economize.backends import export_drafter, exporter_for
from economize.backends.vllm.medusa import MedusaVllmExporter
from economize.errors import ConfigError
from economize.techniques.medusa import MedusaConfig
from helpers import VOCAB, make_tiny_model, perturb

HEADS, LAYERS = 3, 2


def _model(head: str = "own") -> Any:
    model = economize.apply(make_tiny_model(seed=0), MedusaConfig(num_heads=HEADS, num_layers=LAYERS, head=head))
    perturb(model.techniques["medusa"], 0.05, seed=1)
    return model


def test_the_registered_exporter_is_picked() -> None:
    assert isinstance(exporter_for(_model(), "vllm"), MedusaVllmExporter)


def test_config_and_tensor_names(tmp_path: Path) -> None:
    folder = export_drafter(_model(), tmp_path / "medusa", backend="vllm")
    config = json.loads((folder / "config.json").read_text())
    assert config["model_type"] == "medusa" and config["architectures"] == ["MedusaModel"]
    assert (config["num_heads"], config["num_hidden_layers"], config["vocab_size"]) == (HEADS, LAYERS, VOCAB)
    assert config["medusa_fc_bias"] is True and config["hidden_size"] == 64
    names = set(load_file(str(folder / "model.safetensors")))
    expected = {f"blocks.{h}.layers.{i}.{p}" for h in range(HEADS) for i in range(LAYERS) for p in ("weight", "bias")}
    assert names == expected | {f"lm_heads.{h}.weight" for h in range(HEADS)}


def test_vllm_head_arithmetic_matches_head_logits(tmp_path: Path) -> None:
    model = _model()
    technique = model.techniques["medusa"]
    tensors = load_file(str(export_drafter(model, tmp_path / "medusa", backend="vllm") / "model.safetensors"))
    hidden = torch.randn(5, 64, generator=torch.Generator().manual_seed(2))
    x = technique.adapter.final_norm(hidden)  # vLLM hands Medusa the target's post-norm last hidden state
    logits = []
    for h in range(HEADS):  # vllm.model_executor.models.medusa.ResidualBlock, then lm_heads[h]
        y = x
        for i in range(LAYERS):
            weight, bias = tensors[f"blocks.{h}.layers.{i}.weight"], tensors[f"blocks.{h}.layers.{i}.bias"]
            y = y + F.silu(F.linear(y, weight, bias))
        logits.append(y @ tensors[f"lm_heads.{h}.weight"].T)
    torch.testing.assert_close(torch.stack(logits, -2), technique.head_logits(hidden), atol=1e-5, rtol=1e-5)


def test_vllm_config_class_reads_the_folder(tmp_path: Path) -> None:
    pytest.importorskip("vllm")
    from vllm.transformers_utils.configs.medusa import MedusaConfig as VllmMedusaConfig

    config = VllmMedusaConfig.from_pretrained(str(export_drafter(_model(), tmp_path / "medusa", backend="vllm")))
    assert (config.num_heads, config.num_hidden_layers, config.truncated_vocab_size) == (HEADS, LAYERS, VOCAB)
    assert config.medusa_fc_bias is True


def test_shared_heads_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="head='own'"):
        export_drafter(_model("shared"), tmp_path / "medusa", backend="vllm")
