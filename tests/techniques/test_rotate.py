"""``rotate``: QuaRot's value and online Hadamard rotations, their exactness, the outliers they remove before a
weight quantization, the combinations they refuse, and their vLLM / SGLang exports.

The oracles are compressed-tensors (``deterministic_hadamard_matrix``, and its own ``apply_transform_config`` run on
the checkpoint economize writes for vLLM) and the dense reference ``x @ H``.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import economize
from economize.backends import UnsupportedBackendError
from economize.errors import ConfigError, IncompatibleTechniquesError
from economize.techniques.kv_quant.fp8 import FP8KVConfig
from economize.techniques.lowrank import LowRankConfig
from economize.techniques.prune import PruneConfig
from economize.techniques.quant import QuantConfig
from economize.techniques.rotate import (
    HADAMARD_KERNELS,
    RotateConfig,
    block_hadamard,
    hadamard_block,
    hadamard_transform,
    input_hadamard_of,
    rotate_model,
    sylvester,
)
from economize.techniques.sparsity import TealConfig
from helpers import VOCAB, family_skip_reason, make_family_model, make_tiny_model

HAS_CT = importlib.util.find_spec("compressed_tensors") is not None
IDS = torch.randint(0, VOCAB, (2, 24), generator=torch.Generator().manual_seed(1))


def _logits(model: nn.Module, ids: torch.Tensor = IDS) -> torch.Tensor:
    return economize.unwrap.base_model_of(model)(ids).logits


# ---- Hadamard matrices and kernels -------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 8, 64])
def test_sylvester_is_symmetric_orthogonal(n: int) -> None:
    matrix = sylvester(n, dtype=torch.float64)
    torch.testing.assert_close(matrix, matrix.T)
    torch.testing.assert_close(matrix @ matrix, torch.eye(n, dtype=torch.float64))
    assert set(sylvester(n, normalized=False).unique().tolist()) <= {-1.0, 1.0}


@pytest.mark.skipif(not HAS_CT, reason="needs compressed-tensors")
@pytest.mark.parametrize("n", [2, 16, 128])
def test_sylvester_matches_compressed_tensors(n: int) -> None:
    from compressed_tensors.transform.utils.hadamard import deterministic_hadamard_matrix

    expected = deterministic_hadamard_matrix(n, torch.float64, torch.device("cpu"))
    torch.testing.assert_close(sylvester(n, normalized=False, dtype=torch.float64), expected)


def test_sylvester_refuses_other_sizes() -> None:
    with pytest.raises(ConfigError, match="power-of-two"):
        sylvester(12)


@pytest.mark.parametrize(("n", "cap", "block"), [(96, 0, 32), (128, 0, 128), (21504, 0, 1024), (512, 64, 64),
                                                 (7, 0, 1)])
def test_hadamard_block(n: int, cap: int, block: int) -> None:
    assert hadamard_block(n, cap) == block


def test_hadamard_block_refuses_a_bad_cap() -> None:
    with pytest.raises(ConfigError, match="power of two"):
        hadamard_block(64, 48)


@pytest.mark.parametrize(("width", "block"), [(64, 64), (96, 32), (48, 16)])
def test_torch_kernel_matches_the_dense_matrix(width: int, block: int) -> None:
    x = torch.randn(3, 5, width)
    expected = (x.double() @ block_hadamard(width, block, dtype=torch.float64)).float()
    torch.testing.assert_close(hadamard_transform(x, block, "torch"), expected)  # computed in float32
    torch.testing.assert_close(hadamard_transform(x, block), expected)  # auto picks torch on CPU


def test_transform_refuses_a_block_that_does_not_divide() -> None:
    with pytest.raises(ConfigError, match="does not divide"):
        hadamard_transform(torch.randn(2, 48), 32)


@pytest.mark.gpu
@pytest.mark.skipif(importlib.util.find_spec("fast_hadamard_transform") is None,
                    reason="needs fast-hadamard-transform")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_dao_kernel_matches_the_torch_kernel() -> None:
    x = torch.randn(4, 3 * 1024, device="cuda", dtype=torch.bfloat16)
    kernel = HADAMARD_KERNELS.get("fast_hadamard_transform")
    torch.testing.assert_close(kernel(x, 1024).float(), hadamard_transform(x.float(), 1024, "torch"),
                               atol=2e-2, rtol=2e-2)


@pytest.mark.gpu
@pytest.mark.skipif(importlib.util.find_spec("vllm") is None, reason="needs vLLM's compiled ops")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("block", [64, 1024, 32768])
def test_hadacore_kernel_matches_the_torch_kernel(block: int) -> None:
    x = torch.randn(4, 3 * block, device="cuda", dtype=torch.bfloat16)
    original = x.clone()
    kernel = HADAMARD_KERNELS.get("hadacore")
    assert kernel.available(x, block)
    torch.testing.assert_close(kernel(x, block).float(), hadamard_transform(x.float(), block, "torch"),
                               atol=2e-2, rtol=2e-2)
    assert torch.equal(x, original)


def test_hadacore_kernel_declines_what_it_gets_wrong() -> None:
    kernel = HADAMARD_KERNELS.get("hadacore")
    assert not kernel.available(torch.randn(4, 1024), 1024)
    if torch.cuda.is_available():
        assert not kernel.available(torch.randn(4, 32, device="cuda", dtype=torch.bfloat16), 32)
        assert not kernel.available(torch.randn(4, 1024, device="cuda"), 1024)


# ---- config ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"method": "nope"}, "rotation method"),
    ({"block": 48}, "power of two"),
    ({"block": -1}, "block"),
    ({"seed": 1.5}, "seed"),
    ({"kernel": "nope"}, "Hadamard kernel"),
    ({"online": (), "values": False}, "rotates nothing"),
])
def test_config_validation(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        RotateConfig(**kwargs)


# ---- exactness ------------------------------------------------------------------------------------------------------


def test_rotation_leaves_the_function_unchanged(adapter_of) -> None:
    model = make_tiny_model(num_layers=2).eval()
    reference = model(IDS).logits
    summary = rotate_model(adapter_of(model), RotateConfig())
    assert summary["values"] == [1, 2] and set(summary["online"]) == {"1.mlp.down_proj", "2.mlp.down_proj"}
    assert input_hadamard_of(model.model.layers[0].mlp.down_proj) == summary["online"]["1.mlp.down_proj"]
    torch.testing.assert_close(model(IDS).logits, reference, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("family", ["gemma4_text", "gemma3_text", "llama"])
def test_rotation_is_exact_across_families(family: str, adapter_of) -> None:
    reason = family_skip_reason(family)
    if reason:
        pytest.skip(reason)
    model = make_family_model(family).eval()
    ids = IDS % adapter_of(model).vocab_size
    reference = model(ids).logits
    summary = rotate_model(adapter_of(model), RotateConfig())
    assert summary["online"]
    torch.testing.assert_close(model(ids).logits, reference, atol=1e-4, rtol=1e-4)
    if family == "gemma4_text":  # KV sharing and k_eq_v leave some layers' values unrotated
        assert any("reads the values" in why or "v_proj" in why for why in summary["skipped"].values())


def test_value_rotation_changes_the_weights(adapter_of) -> None:
    model = make_tiny_model(num_layers=1).eval()
    before = model.model.layers[0].self_attn.v_proj.weight.clone()
    rotate_model(adapter_of(model), RotateConfig(online=()))
    assert not torch.allclose(model.model.layers[0].self_attn.v_proj.weight, before)
    assert input_hadamard_of(model.model.layers[0].mlp.down_proj) is None


def test_a_layer_cannot_be_rotated_twice(adapter_of) -> None:
    model = make_tiny_model(num_layers=1).eval()
    rotate_model(adapter_of(model), RotateConfig())
    with pytest.raises(ConfigError, match="already rotated"):
        rotate_model(adapter_of(model), RotateConfig(values=False))


def test_a_failed_rotation_restores_every_weight(adapter_of, monkeypatch: pytest.MonkeyPatch) -> None:
    from economize.techniques.rotate import model as rotate_module

    model = make_tiny_model(num_layers=2).eval()
    before = copy.deepcopy(model.state_dict())
    calls = {"n": 0}
    original = rotate_module.hadamard_transform

    def failing(x: torch.Tensor, block: int, kernel: str = "auto") -> torch.Tensor:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return original(x, block, kernel)

    monkeypatch.setattr(rotate_module, "hadamard_transform", failing)
    with pytest.raises(RuntimeError, match="boom"):
        rotate_model(adapter_of(model), RotateConfig())
    after = model.state_dict()
    assert all(torch.equal(value, after[key]) for key, value in before.items())
    assert input_hadamard_of(model.model.layers[0].mlp.down_proj) is None


def test_a_deep_copy_keeps_the_online_rotation(adapter_of) -> None:
    model = make_tiny_model(num_layers=1).eval()
    reference = model(IDS).logits
    rotate_model(adapter_of(model), RotateConfig())
    torch.testing.assert_close(copy.deepcopy(model)(IDS).logits, reference, atol=1e-4, rtol=1e-4)


# ---- the outliers it removes ----------------------------------------------------------------------------------------


def _outlier_model() -> nn.Module:
    """A tiny model with a few large ``down_proj`` input channels (the same function: ``up_proj`` is scaled down)."""
    model = make_tiny_model(num_layers=2).eval()
    channels = torch.tensor([3, 17, 40])
    for layer in model.model.layers:
        layer.mlp.down_proj.weight[:, channels] *= 30
        layer.mlp.up_proj.weight[channels] /= 30
    return model


def test_rotate_lowers_the_error_of_round_to_nearest() -> None:
    base = _outlier_model()
    reference = base(IDS).logits
    quant = QuantConfig(format="int4", group_size=None)
    errors = {}
    for name, configs in {"rtn": (quant,), "rotate+rtn": (RotateConfig(), quant)}.items():
        model = economize.apply(copy.deepcopy(base), *configs)
        errors[name] = (_logits(model) - reference).pow(2).mean().item()
    assert errors["rotate+rtn"] < 0.6 * errors["rtn"], errors


# ---- combinations ---------------------------------------------------------------------------------------------------


def test_rotate_alone_is_refused() -> None:
    with pytest.raises(IncompatibleTechniquesError, match="rotate needs"):
        economize.apply(make_tiny_model(num_layers=1), RotateConfig())


@pytest.mark.parametrize("other", [
    PruneConfig(sparsity=0.5),
    LowRankConfig(),
    TealConfig(sparsity=0.5),
    QuantConfig(method="awq", format="int4", group_size=32),
    QuantConfig(method="smoothquant", format="int8", group_size=None, symmetric=True, activations="int8"),
    FP8KVConfig(scale="tensor"),
])
def test_refused_combinations(other: Any) -> None:
    from economize.core.technique import TECHNIQUES

    rotate = TECHNIQUES.get("rotate")
    other_class = TECHNIQUES.get(other.technique)
    assert rotate.conflict(RotateConfig(), other_class, other) is not None or (
        other_class.conflict(other, rotate, RotateConfig()) is not None)


@pytest.mark.parametrize("other", [
    QuantConfig(format="int4", group_size=32),
    QuantConfig(method="gptq", format="int4", group_size=32),
    QuantConfig(method="hqq", format="int4", group_size=32),
    FP8KVConfig(),
])
def test_allowed_combinations(other: Any) -> None:
    from economize.core.technique import TECHNIQUES

    rotate = TECHNIQUES.get("rotate")
    other_class = TECHNIQUES.get(other.technique)
    assert rotate.conflict(RotateConfig(), other_class, other) is None
    assert other_class.conflict(other, rotate, RotateConfig()) is None


def test_awq_is_allowed_without_online_rotations() -> None:
    from economize.core.technique import TECHNIQUES

    awq = QuantConfig(method="awq", format="int4", group_size=32)
    assert TECHNIQUES.get("rotate").conflict(RotateConfig(online=()), TECHNIQUES.get("quant"), awq) is None


# ---- vLLM and SGLang ------------------------------------------------------------------------------------------------


@pytest.fixture
def tiny_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "tiny"
    make_tiny_model(seed=0).save_pretrained(folder)
    return folder


def _base(folder: Path) -> nn.Module:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32).eval()


@pytest.mark.skipif(not HAS_CT, reason="needs compressed-tensors")
def test_vllm_export_declares_the_online_transforms(tiny_dir: Path, tmp_path: Path) -> None:
    """The checkpoint holds the ``+-1`` matrices and a ``transform_config`` vLLM reads, and compressed-tensors' own
    transforms applied to the reloaded checkpoint reproduce economize's model."""
    from compressed_tensors.transform import TransformConfig, apply_transform_config
    from safetensors.torch import load_file

    from economize.backends.vllm.serving import to_vllm

    model = economize.apply(_base(tiny_dir), RotateConfig(),
                            QuantConfig(format="int4", group_size=32, backend="compressed-tensors"))
    plan = to_vllm(model, tmp_path / "out")
    assert plan.techniques["rotate"] == "native"
    folder = Path(plan.model)
    transforms = json.loads((folder / "config.json").read_text())["quantization_config"]["transform_config"]
    config = TransformConfig.model_validate(transforms)
    (scheme,) = config.config_groups.values()
    assert scheme.type == "hadamard" and scheme.head_dim == 128 and len(scheme.apply[0].targets) == 6
    state: dict[str, torch.Tensor] = {}
    for path in folder.glob("*.safetensors"):
        state.update(load_file(path))
    key = "model.layers.0.mlp.down_proj.economize_h128_input.weight"
    torch.testing.assert_close(state[key], sylvester(128, normalized=False))
    reloaded = _base(folder)
    apply_transform_config(reloaded, config)
    torch.testing.assert_close(reloaded(IDS).logits, _logits(model), atol=1e-4, rtol=1e-4)


def test_vllm_export_folds_dense_rotations_back(tiny_dir: Path, tmp_path: Path) -> None:
    from economize.backends.vllm.serving import to_vllm

    model = economize.apply(_base(tiny_dir), RotateConfig(), QuantConfig(format="int4", group_size=32))
    plan = to_vllm(model, tmp_path / "out")
    config = json.loads((Path(plan.model) / "config.json").read_text())
    assert "transform_config" not in (config.get("quantization_config") or {})
    torch.testing.assert_close(_base(Path(plan.model))(IDS).logits, _logits(model), atol=1e-4, rtol=1e-4)


def test_transform_targets_match_vllm_names() -> None:
    import re

    from economize.backends.vllm.serving.rotate import _target

    pattern = _target("model.language_model.layers.3.mlp.down_proj")[3:]
    assert re.match(pattern, "model.layers.3.mlp.down_proj")
    assert re.match(pattern, "language_model.model.layers.3.mlp.down_proj")
    assert not re.match(pattern, "model.layers.13.mlp.down_proj")


@pytest.mark.skipif(not HAS_CT, reason="needs compressed-tensors")
def test_sglang_refuses_online_rotations_of_quantized_maps(tiny_dir: Path) -> None:
    from economize.backends.sglang.serving import check_servable

    quant = QuantConfig(format="int4", group_size=32, backend="compressed-tensors")
    with pytest.raises(UnsupportedBackendError, match="online Hadamard"):
        check_servable(economize.apply(_base(tiny_dir), RotateConfig(), quant))
    assert check_servable(economize.apply(_base(tiny_dir), RotateConfig(online=()), quant))["rotate"].value == "native"
