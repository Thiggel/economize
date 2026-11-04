"""HQQ (half-quadratic quantization): the official ``hqq`` package as oracle, the config gates of its real-valued
zero points, and the kernel backends that run it."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from economize.errors import ConfigError
from economize.techniques.quant import QuantConfig, hqq_quantize, make_grid, quantize_model, quantize_rtn

HAS_CT = importlib.util.find_spec("compressed_tensors") is not None


def _weight(seed: int = 0) -> torch.Tensor:
    weight = torch.randn(48, 128, generator=torch.Generator().manual_seed(seed)) * 0.05
    weight[3, 5] = 0.9  # an outlier, where the l_p fit moves the zero point the most
    return weight


@pytest.mark.parametrize("group_size", [32, 64, None])
@pytest.mark.parametrize("round_zero", [False, True])
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_hqq_matches_the_official_package(group_size: int | None, round_zero: bool, bits: int) -> None:
    """Codes, zero points and dequantized weights equal ``hqq.core.quantize.Quantizer`` (its fp32 CPU path)."""
    pytest.importorskip("hqq")
    from hqq.core.quantize import Quantizer

    weight = _weight(bits)
    codes, meta = Quantizer.quantize(
        weight, nbits=bits, group_size=group_size, axis=1, optimize=True, round_zero=round_zero, bitpack=False,
        device="cpu",
    )
    reference = Quantizer.dequantize(codes, meta).reshape(weight.shape)
    ours = hqq_quantize(weight, make_grid(f"int{bits}"), group_size, round_zero=round_zero)
    assert ours.zero is not None
    assert torch.equal(ours.codes.reshape(-1).float(), codes.reshape(-1).float())
    assert torch.equal(ours.zero.reshape(-1), meta["zero"].reshape(-1).float())
    torch.testing.assert_close(ours.dequantize(), reference, atol=0, rtol=0)


def _reference_hqq(weight: torch.Tensor, bits: int, size: int, iters: int, p: float, beta: float) -> torch.Tensor:
    """The proximal loop of Badri and Shaji (2023), written out per group, for when ``hqq`` is not installed."""
    maxq = 2**bits - 1
    out = torch.empty_like(weight)
    for row in range(weight.shape[0]):
        for start in range(0, weight.shape[1], size):
            w = weight[row, start : start + size].double()
            s = maxq / (w.max() - w.min())
            z = -w.min() * s
            best = float("inf")
            for _ in range(iters):
                q = torch.round(w * s + z).clamp(0, maxq)
                r = (q - z) / s
                e = w - r
                e = torch.sign(e) * torch.relu(e.abs() - e.abs().pow(p - 1) / beta)
                z = (q - (w - e) * s).mean()
                current = (w - r).abs().mean().item()
                if current >= best:
                    break
                best = current
            q = torch.round(w * s + z).clamp(0, maxq)
            out[row, start : start + size] = ((q - z) / s).to(weight.dtype)
    return out


def test_hqq_matches_the_paper_reference_per_group() -> None:
    """A paper-style reference agrees on single-group weights (hqq stops on the error of the whole tensor, the
    reference on that of the group, so the two loops coincide only when the tensor is one group)."""
    for row in _weight(1)[:4]:
        weight = row[None]
        ours = hqq_quantize(weight, make_grid("int4"), None).dequantize()
        torch.testing.assert_close(ours, _reference_hqq(weight, 4, 128, 20, 0.7, 10.0), atol=1e-5, rtol=0)


def test_hqq_fits_the_weight_better_than_round_to_nearest() -> None:
    """The l_p fit lowers the mean absolute error of the min-max grid, the quantity it decreases step by step."""
    weight = _weight(2)
    grid = make_grid("int3")
    rtn = (quantize_rtn(weight, grid, 32).dequantize() - weight).abs().mean()
    hqq = (hqq_quantize(weight, grid, 32).dequantize() - weight).abs().mean()
    assert hqq < rtn


def test_zero_iterations_is_the_min_max_grid() -> None:
    """Without proximal steps HQQ is round to nearest on the group's own range (0 need not lie in it) with a real
    zero point."""
    weight = _weight(3)
    groups = weight.reshape(-1, 32)
    low, high = groups.min(1, keepdim=True).values, groups.max(1, keepdim=True).values
    s = 15 / (high - low)
    expected = (torch.round(groups * s - low * s).clamp(0, 15) + low * s) / s
    hqq = hqq_quantize(weight, make_grid("int4"), 32, iters=0)
    torch.testing.assert_close(hqq.dequantize(), expected.reshape(weight.shape), atol=1e-6, rtol=0)


def test_integer_zero_rounds_the_zero_points_onto_the_grid() -> None:
    q = hqq_quantize(_weight(4), make_grid("int4"), 32, integer_zero=True)
    assert q.zero is not None
    assert torch.equal(q.zero, q.zero.round()) and bool(((q.zero >= 0) & (q.zero <= 15)).all())
    assert not torch.equal(hqq_quantize(_weight(4), make_grid("int4"), 32).zero, q.zero)


def test_indivisible_groups_raise() -> None:
    with pytest.raises(ConfigError, match="does not divide"):
        hqq_quantize(_weight(), make_grid("int4"), 48)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"symmetric": True}, "symmetric=False"),
        ({"format": "fp8", "group_size": None}, "integer formats"),
        ({"hqq_lp_norm": 1.5}, "hqq_lp_norm"),
        ({"hqq_beta": 0.0}, "hqq_beta"),
        ({"backend": "compressed-tensors"}, "real-valued zero points"),
        ({"backend": "torchao", "format": "int8", "group_size": 32}, "real-valued zero points"),
        ({"method": "rtn", "hqq_iters": 5}, "hqq_iters"),
    ],
)
def test_hqq_config_validation(kwargs: dict, match: str) -> None:
    if kwargs.get("backend") == "compressed-tensors":
        pytest.importorskip("compressed_tensors")
        import economize.integrations.compressed_tensors  # noqa: F401  (registers the backend)
    with pytest.raises(ConfigError, match=match):
        QuantConfig(**{"method": "hqq", **kwargs})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "torchao", "group_size": 32},
        {"backend": "fake"},
        {"backend": "compressed-tensors", "hqq_integer_zero": True},
    ],
)
def test_hqq_backends_that_store_its_zero_points(kwargs: dict) -> None:
    if kwargs["backend"] == "compressed-tensors":
        pytest.importorskip("compressed_tensors")
        import economize.integrations.compressed_tensors  # noqa: F401
    QuantConfig(method="hqq", **kwargs)


def test_hqq_model_writes_the_hqq_weights(tiny2, adapter_of, held_out, nll) -> None:
    import copy

    from economize.techniques.prune.calibration import linear_modules

    original = copy.deepcopy(tiny2)
    dense = nll(tiny2, held_out)
    adapter = adapter_of(tiny2)
    summary = quantize_model(adapter, QuantConfig(method="hqq", format="int4", group_size=32))
    assert summary["modules"] == 14
    before = linear_modules(adapter, original.model.layers[0])
    for name, module in linear_modules(adapter, adapter.layer(1)).items():
        assert torch.equal(module.weight, hqq_quantize(before[name].weight, make_grid("int4"), 32).dequantize()), name
    assert abs(nll(tiny2, held_out) - dense) < 1.0


@pytest.mark.skipif(not HAS_CT, reason="needs compressed-tensors")
def test_integer_zero_hqq_exports_to_compressed_tensors(tmp_path: Path) -> None:
    from transformers import AutoModelForCausalLM

    from economize.arch import adapter_for
    from economize.integrations.compressed_tensors import export_compressed_tensors
    from helpers import make_tiny_model

    model = make_tiny_model(num_layers=2)
    config = QuantConfig(method="hqq", group_size=32, backend="compressed-tensors", hqq_integer_zero=True)
    quantize_model(adapter_for(model), config)
    export_compressed_tensors(model, tmp_path)
    assert json.loads((tmp_path / "config.json").read_text())["quantization_config"]["format"] == "pack-quantized"
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32).eval()
    ids = torch.arange(1, 20)[None]
    with torch.no_grad():
        torch.testing.assert_close(reloaded(ids).logits, model(ids).logits, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not HAS_CT, reason="needs compressed-tensors")
def test_float_zero_codes_refuse_the_compressed_tensors_export() -> None:
    from economize.integrations.compressed_tensors import _local_state

    q = hqq_quantize(_weight(), make_grid("int4"), 32)
    with pytest.raises(ConfigError, match="hqq_integer_zero"):
        _local_state(q, torch.float32)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0), reason="sm80+")
def test_torchao_tinygemm_runs_hqq_float_zeros(tiny_factory, adapter_of, held_out) -> None:
    """tinygemm int4 stores HQQ's real zero points as they are: its logits match the fake backend's."""
    pytest.importorskip("torchao")
    outputs = []
    for backend in ("fake", "torchao"):
        model = tiny_factory(num_layers=2).to("cuda", torch.bfloat16)
        quantize_model(adapter_of(model), QuantConfig(method="hqq", group_size=32, backend=backend))
        outputs.append(model(held_out.cuda()).logits.float())
    assert type(model.model.layers[0].mlp.up_proj.weight).__module__.startswith("torchao")
    torch.testing.assert_close(outputs[1], outputs[0], atol=5e-2, rtol=5e-2)
