"""The ``"triton"`` quantization backend (a weight-only int8 / fp8 GEMM that decodes the codes in registers).

The configuration checks run everywhere. Everything that launches the kernel is skipped without triton and a GPU of
compute capability 8.0 or newer.
"""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from economize.errors import ConfigError
from economize.techniques.quant import (
    QUANT_BACKENDS,
    QuantConfig,
    make_grid,
    quantize_model,
    quantize_rtn,
    read_quantized_weight,
)
from economize.techniques.quant.auto import layout_candidates
from economize.techniques.quant.triton_gemm import MAX_KERNEL_ROWS, TritonWeight, triton_available, triton_weight

requires_triton = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0) and triton_available()),
    reason="needs triton and a GPU of compute capability 8.0 or newer",
)

# (format, symmetric, group_size): every layout the kernel stores
LAYOUTS = [
    ("int8", True, None),
    ("int8", False, None),
    ("int8", True, 128),
    ("int8", False, 128),
    ("int8", False, 32),
    ("fp8", True, None),
    ("fp8", True, 64),
]


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).abs().max() / expected.float().abs().max()).item()


# ---- configuration (no kernels) -------------------------------------------------------------------------------------


def test_is_registered_and_chosen_by_auto_before_torchao() -> None:
    assert "triton" in QUANT_BACKENDS.names()
    for fmt, symmetric, group in LAYOUTS:
        config = QuantConfig(backend="triton", format=fmt, symmetric=symmetric, group_size=group)
        fitting, _ = layout_candidates(config)
        assert "triton" in fitting
        if "torchao" in fitting:
            assert fitting.index("triton") < fitting.index("torchao")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(format="int4", group_size=128), "stores int8 and fp8"),
        (dict(format="int8", group_size=48), "group_size"),
        (dict(format="int8", group_size=None, symmetric=True, activations="int8"), "activations"),
    ],
)
def test_config_rejects_layouts_without_a_kernel(kwargs: dict, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        QuantConfig(backend="triton", **kwargs)


# ---- kernels --------------------------------------------------------------------------------------------------------


def _codes(n: int, k: int, fmt: str, symmetric: bool, group: int | None, gen: torch.Generator):
    weight = torch.randn(n, k, device="cuda", generator=gen) * 0.05 + 0.01  # an offset makes zero points differ
    return quantize_rtn(weight, make_grid(fmt, symmetric), group)


@requires_triton
@pytest.mark.parametrize(("fmt", "symmetric", "group"), LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_linear_matches_fake_quantization(fmt: str, symmetric: bool, group: int | None, dtype: torch.dtype) -> None:
    gen = torch.Generator(device="cuda").manual_seed(0)
    n, k = (320, 1024) if group else (200, 376)  # unaligned shapes for per-channel scales
    q = _codes(n, k, fmt, symmetric, group, gen)
    stored = triton_weight(q, dtype)
    assert isinstance(stored, TritonWeight) and stored.shape == (n, k) and stored.dtype == dtype
    dense = q.dequantize(torch.float32)
    torch.testing.assert_close(stored.dequantize(), dense.to(dtype), atol=0, rtol=0)
    bias = torch.randn(n, device="cuda", dtype=dtype, generator=gen)
    for m in (1, 5, 16, 33, MAX_KERNEL_ROWS + 1):  # the last one takes the dense path
        x = torch.randn(2, m, k, device="cuda", dtype=dtype, generator=gen)
        out = F.linear(x, stored, bias)
        assert out.shape == (2, m, n) and out.dtype == dtype
        reference = F.linear(x.float(), dense, bias.float())
        assert _relative_error(out, reference) < 1e-2, (m, _relative_error(out, reference))


@requires_triton
@pytest.mark.parametrize(("fmt", "symmetric", "group"), LAYOUTS)
def test_codes_read_back_exactly(fmt: str, symmetric: bool, group: int | None) -> None:
    gen = torch.Generator(device="cuda").manual_seed(1)
    q = _codes(256, 512, fmt, symmetric, group, gen)
    back = read_quantized_weight(triton_weight(q, torch.bfloat16))
    assert back is not None and back.group_size == q.group_size and back.grid.symmetric == q.grid.symmetric
    torch.testing.assert_close(back.codes.view(torch.uint8), q.codes.view(torch.uint8), atol=0, rtol=0)
    torch.testing.assert_close(back.scale, q.scale.float(), atol=0, rtol=0)
    if q.zero is not None:
        torch.testing.assert_close(back.zero, q.zero.float(), atol=0, rtol=0)


@requires_triton
def test_views_keep_the_triton_kernel() -> None:
    q = _codes(256, 256, "int8", False, 128, torch.Generator(device="cuda").manual_seed(2))
    stored = triton_weight(q, torch.float32)
    for view in (stored.detach(), stored.clone(), stored.to(torch.bfloat16)):
        assert type(view) is TritonWeight
    x = torch.randn(3, 256, device="cuda")
    out = F.linear(x, stored)
    assert out.dtype == torch.float32 and _relative_error(out, F.linear(x, q.dequantize())) < 1e-2


@requires_triton
def test_cuda_graph_replay_matches_eager() -> None:
    q = _codes(1024, 512, "fp8", True, None, torch.Generator(device="cuda").manual_seed(3))
    stored = triton_weight(q, torch.bfloat16)
    static_x = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            F.linear(static_x, stored)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = F.linear(static_x, stored)
    x = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
    static_x.copy_(x)
    graph.replay()
    torch.testing.assert_close(static_out, F.linear(x, stored), atol=0, rtol=0)


@requires_triton
# the tiny model is 64 wide, so int8 uses groups of 32
@pytest.mark.parametrize(("fmt", "symmetric", "group"), [("int8", False, 32), ("fp8", True, None)])
def test_model_matches_fake_quantization_and_auto_skips_torchao(
    tiny_factory, adapter_of, held_out, fmt: str, symmetric: bool, group: int | None, caplog
) -> None:
    def quantized(backend: str) -> nn.Module:
        model = tiny_factory(num_layers=2).to("cuda", torch.bfloat16)
        config = QuantConfig(backend=backend, format=fmt, symmetric=symmetric, group_size=group)
        quantize_model(adapter_of(model), config)
        return model

    tokens = held_out.cuda()
    fake = quantized("fake")
    noise = _relative_error(fake(tokens).logits, fake.float()(tokens).logits)
    reference = fake.float()(tokens).logits
    with caplog.at_level(logging.WARNING):
        model = quantized("triton")
    assert "dequantizes the weight on every eager call" not in caplog.text
    layer = model.model.layers[0]
    for linear in (layer.mlp.down_proj, layer.self_attn.k_proj):
        assert isinstance(linear.weight, TritonWeight)
    assert _relative_error(model(tokens).logits, reference) < 2 * noise + 1e-2, noise
