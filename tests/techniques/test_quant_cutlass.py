"""The ``"cutlass"`` quantization backend (vLLM's CUTLASS int8 x int8 GEMM for W8A8).

The configuration checks run everywhere. Everything that calls the kernels is skipped unless vLLM's CUTLASS int8 ops
load and the GPU has compute capability 7.5 or newer.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from economize.errors import ConfigError
from economize.techniques.prune import token_batches
from economize.techniques.quant import (
    ACTIVATION_SCHEMES,
    QUANT_BACKENDS,
    DraftQuantConfig,
    QuantConfig,
    make_grid,
    quantize_model,
    quantize_rtn,
    read_quantized_weight,
)
from economize.techniques.quant.activations import activation_scheme_of
from economize.techniques.quant.cutlass import CutlassInt8Weight, cutlass_available, cutlass_int8_weight

requires_cutlass = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 5) and cutlass_available()),
    reason="needs vLLM's CUTLASS int8 kernels and a GPU of compute capability 7.5 or newer",
)
W8A8 = dict(format="int8", group_size=None, symmetric=True, activations="int8")


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).abs().max() / expected.float().abs().max()).item()


# ---- configuration (no kernels) -------------------------------------------------------------------------------------


def test_is_registered() -> None:
    assert "cutlass" in QUANT_BACKENDS.names()
    assert QuantConfig(backend="cutlass", **W8A8).backend == "cutlass"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(W8A8, activations=None), "activations='int8'"),
        (dict(W8A8, format="int4"), "format 'int8'"),
        (dict(W8A8, group_size=128), "group_size"),
        (dict(W8A8, symmetric=False), "no zero points"),
    ],
)
def test_config_rejects_layouts_without_a_kernel(kwargs: dict, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        QuantConfig(backend="cutlass", **kwargs)


def test_draft_config_is_refused() -> None:
    # drafter quantization keeps float activations, which the int8 x int8 GEMM cannot take
    with pytest.raises(ConfigError, match="activations='int8'"):
        DraftQuantConfig(group_size=None, backend="cutlass")


# ---- kernels --------------------------------------------------------------------------------------------------------


def _reference(x: torch.Tensor, dense: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """The fake W8A8 product: per-token int8 activations times the dequantized weight, in fp32."""
    quantized = ACTIVATION_SCHEMES.get("int8").fake(x.float())
    return F.linear(quantized, dense, None if bias is None else bias.float())


@requires_cutlass
@pytest.mark.parametrize(("n", "k"), [(512, 1024), (200, 376)])
def test_linear_matches_fake_quantization(n: int, k: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(0)
    q = quantize_rtn(torch.randn(n, k, device="cuda", generator=gen) * 0.05, make_grid("int8", True), None)
    stored = cutlass_int8_weight(q, torch.bfloat16)
    assert isinstance(stored, CutlassInt8Weight) and stored.shape == (n, k) and stored.dtype == torch.bfloat16
    dense = q.dequantize(torch.float32)
    assert _relative_error(stored.dequantize(), dense) < 1e-2
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=gen)
    for m in (1, 7, 33, 130):
        x = torch.randn(2, m, k, device="cuda", dtype=torch.bfloat16, generator=gen)
        out = F.linear(x, stored, bias)
        assert out.shape == (2, m, n) and out.dtype == torch.bfloat16
        assert _relative_error(out, _reference(x, dense, bias)) < 1e-2, (n, k, m)


@requires_cutlass
def test_codes_read_back_exactly() -> None:
    q = quantize_rtn(torch.randn(200, 376, device="cuda") * 0.05, make_grid("int8", True), None)
    back = read_quantized_weight(cutlass_int8_weight(q, torch.bfloat16))
    assert back is not None and back.group_size == 376
    torch.testing.assert_close(back.codes, q.codes, atol=0, rtol=0)
    torch.testing.assert_close(back.scale, q.scale.float(), atol=0, rtol=0)


@requires_cutlass
def test_views_keep_the_cutlass_kernel() -> None:
    q = quantize_rtn(torch.randn(256, 256, device="cuda") * 0.05, make_grid("int8", True), None)
    stored = cutlass_int8_weight(q, torch.float32)
    for view in (stored.detach(), stored.clone(), stored.to(torch.bfloat16)):
        assert type(view) is CutlassInt8Weight
    x = torch.randn(3, 256, device="cuda")
    out = F.linear(x, stored)
    assert out.dtype == torch.float32 and _relative_error(out, _reference(x, q.dequantize(), None)) < 1e-2


@requires_cutlass
def test_cuda_graph_replay_matches_eager() -> None:
    # the reason for this backend: torchao's int8 tensor reads its activation scale back to the host
    q = quantize_rtn(torch.randn(1024, 512, device="cuda") * 0.05, make_grid("int8", True), None)
    stored = cutlass_int8_weight(q, torch.bfloat16)
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


@requires_cutlass
@pytest.mark.parametrize("method", ["rtn", "smoothquant"])
def test_model_matches_fake_quantization(tiny_factory, adapter_of, held_out, calib, method: str) -> None:
    def quantized(backend: str) -> nn.Module:
        model = tiny_factory(num_layers=2).to("cuda", torch.bfloat16)
        data = token_batches(calib.cuda()) if method == "smoothquant" else None
        quantize_model(adapter_of(model), QuantConfig(backend=backend, method=method, **W8A8), data)
        return model

    tokens = held_out.cuda()
    fake = quantized("fake")
    fake_bf16 = fake(tokens).logits
    reference = fake.float()(tokens).logits
    model = quantized("cutlass")
    layer = model.model.layers[0]
    for linear in (layer.mlp.down_proj, layer.mlp.up_proj, layer.self_attn.k_proj):
        assert isinstance(linear.weight, CutlassInt8Weight)
        assert activation_scheme_of(linear) is ACTIVATION_SCHEMES.get("int8")
    noise = _relative_error(fake_bf16, reference)
    assert _relative_error(model(tokens).logits, reference) < 2 * noise + 1e-2, noise
