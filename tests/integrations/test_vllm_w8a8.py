"""W8A8 (``QuantConfig.activations``) in the vLLM and SGLang serving plans: which configs vLLM 0.30 has a kernel for,
and the compressed-tensors checkpoint the plan writes. Checked on CPU without vLLM."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import economize
from economize.backends import UnsupportedBackendError
from economize.backends.vllm import serving
from economize.backends.vllm.serving import Support, to_vllm
from economize.techniques.quant.config import QuantConfig
from helpers import VOCAB, make_tiny_model

needs_ct = pytest.mark.skipif(importlib.util.find_spec("compressed_tensors") is None, reason="needs compressed-tensors")
TOKENS = torch.randint(0, VOCAB, (2, 16), generator=torch.Generator().manual_seed(0))


@pytest.fixture
def tiny_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "tiny"
    make_tiny_model(seed=0).save_pretrained(folder)
    return folder


def _base(folder: Path) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32).eval()


def _config(**kwargs: Any) -> SimpleNamespace:
    fields = {"backend": "compressed-tensors", "format": "int8", "group_size": None, "symmetric": True,
              "activations": "int8", "fp8_activations": False}
    return SimpleNamespace(**{**fields, **kwargs})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"backend": "fake"}, "served dense without activation quantization"),
        ({"format": "int4"}, "no GPU kernel for format='int4' with activations='int8'"),
        ({"format": "int4", "activations": "int4"}, "no GPU kernel"),
        ({"group_size": 128}, "group_size=None"),
        ({"symmetric": False}, "symmetric=True"),
        ({"format": "fp8", "activations": "int8"}, "no GPU kernel"),
    ],
)
def test_vllm_refuses_w8a8_without_a_kernel(kwargs: dict, match: str) -> None:
    adapter = serving.base.adapter_for("quant")
    with pytest.raises(UnsupportedBackendError, match=match):
        adapter.check(SimpleNamespace(config=_config(**kwargs)), serving.PlanBuilder(None, None))


@pytest.mark.parametrize(("fmt", "activations", "backend"), [("int8", "int8", "compressed-tensors"),
                                                             ("int8", "int8", "torchao"),
                                                             ("fp8", "fp8", "compressed-tensors")])
def test_vllm_accepts_the_w8a8_kernels(fmt: str, activations: str, backend: str) -> None:
    adapter = serving.base.adapter_for("quant")
    technique = SimpleNamespace(config=_config(format=fmt, activations=activations, backend=backend))
    adapter.check(technique, serving.PlanBuilder(None, None))
    assert adapter.level(technique) is Support.NATIVE


def test_sglang_checks_w8a8_like_vllm() -> None:
    from economize.backends.sglang import serving as sglang_serving

    adapter = sglang_serving.base.adapter_for("quant")
    adapter.check(SimpleNamespace(config=_config()), None)
    with pytest.raises(UnsupportedBackendError, match="group_size=None"):
        adapter.check(SimpleNamespace(config=_config(group_size=64)), None)


@needs_ct
def test_smoothquant_plan_writes_a_w8a8_checkpoint(tiny_dir: Path, tmp_path: Path) -> None:
    config = QuantConfig(method="smoothquant", format="int8", group_size=None, symmetric=True, activations="int8",
                         backend="compressed-tensors")
    model = economize.apply(_base(tiny_dir), config, calibration=[TOKENS])
    plan = to_vllm(model, tmp_path / "out")
    assert plan.techniques == {"quant": "native"}
    assert any("CompressedTensorsW8A8Int8" in note for note in plan.notes["quant"])
    quantization = json.loads((Path(plan.model) / "config.json").read_text())["quantization_config"]
    assert quantization["format"] == "int-quantized"
    for group in quantization["config_groups"].values():
        assert group["input_activations"]["strategy"] == "token" and group["input_activations"]["dynamic"]
