"""The ``"marlin"`` quantization backend (vLLM's Marlin weight-only GEMM).

The packing helpers and the configuration checks run everywhere. Everything that calls the kernels is skipped
unless vLLM's custom ops load and a CUDA GPU with compute capability 8.0 or newer is present.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from economize.errors import ConfigError
from economize.techniques.quant import (
    DraftQuantConfig,
    QuantConfig,
    make_grid,
    quantize_draft,
    quantize_model,
    quantize_rtn,
    restore_draft,
)
from economize.techniques.quant.marlin import (
    MarlinWeight,
    _gptq_pack,
    _pack,
    marlin_available,
    marlin_padded_nk,
    marlin_weight,
)

HAS_SM80 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0)
requires_marlin = pytest.mark.skipif(
    not (HAS_SM80 and marlin_available()), reason="needs vLLM's Marlin kernels and a CUDA GPU with sm80+"
)

#: (format, symmetric, group_size): every layout the backend stores.
LAYOUTS = [
    ("int8", True, None),
    ("int8", True, 128),
    ("int4", True, 128),
    ("int4", False, 128),
    ("int4", False, 64),
    ("int4", True, None),
    ("int4", False, None),
    ("fp8", True, None),
    ("fp8", True, 128),
]


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).abs().max() / expected.float().abs().max()).item()


# ---- layout helpers and configuration (no kernels) ----------------------------------------------------------------


def test_pack_puts_value_i_at_bits_i_times_width() -> None:
    values = torch.tensor([[1, 2, 3, 15, 0, 7, 8, 9]])
    word = _pack(values, 4).item() & 0xFFFFFFFF
    assert [(word >> (4 * i)) & 0xF for i in range(8)] == [1, 2, 3, 15, 0, 7, 8, 9]
    assert _pack(torch.tensor([[255, 255, 255, 255]]), 8).item() == -1  # wraps to int32


def test_gptq_pack_of_bytes_matches_the_generic_packer() -> None:
    codes = torch.randint(0, 256, (24, 64), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(_gptq_pack(codes, 8), _pack(codes, 8).T.contiguous())
    assert _gptq_pack(codes % 16, 4).shape == (64 // 8, 24)


@pytest.mark.parametrize(
    ("n", "k", "group", "expected"),
    [(4096, 5376, -1, (4096, 5376)), (64, 64, -1, (64, 128)), (200, 384, 128, (256, 384)), (32, 64, 32, (64, 128))],
)
def test_padding_reaches_a_thread_tile_family(n: int, k: int, group: int, expected: tuple[int, int]) -> None:
    padded = marlin_padded_nk(n, k, group)
    assert padded == expected
    pn, pk = padded
    assert (pn % 64 == 0 and pk % 128 == 0) or (pn % 128 == 0 and pk % 64 == 0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(format="int8", group_size=None, symmetric=False), "symmetric"),
        (dict(format="int4", group_size=256), "group_size"),
        (dict(format="fp8", group_size=64), "128"),
        (dict(format="int3", group_size=128), "int4, int8 and fp8"),
    ],
)
def test_config_rejects_layouts_without_a_kernel(kwargs: dict, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        QuantConfig(backend="marlin", **kwargs)


def test_draft_config_accepts_marlin() -> None:
    assert DraftQuantConfig(group_size=None, backend="marlin").backend == "marlin"


# ---- kernels ------------------------------------------------------------------------------------------------------


@requires_marlin
@pytest.mark.parametrize(("fmt", "symmetric", "group"), LAYOUTS)
@pytest.mark.parametrize(("n", "k"), [(512, 1024), (200, 384)])
def test_linear_matches_fake_quantization(fmt: str, symmetric: bool, group: int | None, n: int, k: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randn(n, k, device="cuda", generator=gen) * 0.05
    q = quantize_rtn(weight, make_grid(fmt, symmetric), group)
    stored = marlin_weight(q, torch.bfloat16)
    assert isinstance(stored, MarlinWeight) and stored.shape == (n, k) and stored.dtype == torch.bfloat16
    dense = q.dequantize(torch.float32)
    # the dequantized weight read back through the kernel equals the reference up to bf16 rounding of the scales
    assert _relative_error(stored.dequantize(), dense) < 1e-2
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=gen)
    for m in (1, 7, 33):
        x = torch.randn(2, m, k, device="cuda", dtype=torch.bfloat16, generator=gen)
        out = F.linear(x, stored, bias)
        assert out.shape == (2, m, n) and out.dtype == torch.bfloat16
        ref = F.linear(x.float(), dense, bias.float())
        assert _relative_error(out, ref) < 2e-2, (fmt, symmetric, group, m)


@requires_marlin
def test_float32_modules_compute_in_bf16_and_return_float32() -> None:
    linear = nn.Linear(256, 128, device="cuda")
    q = quantize_rtn(linear.weight.data, make_grid("int8", True), None)
    stored = marlin_weight(q, torch.float32)
    x = torch.randn(3, 256, device="cuda")
    out = F.linear(x, stored)
    assert stored.dtype == torch.float32 and out.dtype == torch.float32
    assert _relative_error(out, x @ q.dequantize().T) < 2e-2
    assert stored.to(torch.bfloat16).dtype == torch.bfloat16  # changes the reported dtype only


@requires_marlin
def test_cuda_graph_replay_matches_eager() -> None:
    weight = torch.randn(1024, 512, device="cuda") * 0.05
    stored = marlin_weight(quantize_rtn(weight, make_grid("fp8"), None), torch.bfloat16)
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


@requires_marlin
@pytest.mark.parametrize(
    "kwargs",
    [
        {"format": "int8", "symmetric": True, "group_size": None},
        {"format": "int4", "group_size": 32},
        {"format": "int4", "symmetric": True, "group_size": 64},
        {"format": "fp8", "group_size": None},
    ],
)
def test_model_matches_fake_quantization(tiny_factory, adapter_of, held_out, kwargs) -> None:
    def quantized(backend: str) -> nn.Module:
        model = tiny_factory(num_layers=2).to("cuda", torch.bfloat16)
        quantize_model(adapter_of(model), QuantConfig(backend=backend, **kwargs))
        return model

    tokens = held_out.cuda()
    fake = quantized("fake")
    fake_bf16 = fake(tokens).logits
    reference = fake.float()(tokens).logits  # the same quantized weights with fp32 activations
    model = quantized("marlin")
    assert isinstance(model.model.layers[0].mlp.up_proj.weight, MarlinWeight)
    assert isinstance(model.model.layers[0].self_attn.k_proj.weight, MarlinWeight)  # padded from 32 x 64
    # bf16 activations alone move the logits by ``noise`` (about 3e-2 for int8 per channel after two layers); the
    # kernel's own rounding is of the same order, so the bound scales with the floor instead of a fixed constant
    noise = _relative_error(fake_bf16, reference)
    assert _relative_error(model(tokens).logits, reference) < 2 * noise + 1e-2, noise


@requires_marlin
def test_draft_quantization_and_restore() -> None:
    torch.manual_seed(0)
    modules = {"a": nn.Linear(256, 512, device="cuda"), "b": nn.Linear(512, 128, device="cuda")}
    originals = {name: m.weight.detach().clone() for name, m in modules.items()}
    x = torch.randn(5, 256, device="cuda")
    fake = {name: nn.Linear(m.in_features, m.out_features, device="cuda") for name, m in modules.items()}
    for name, m in fake.items():
        m.load_state_dict(modules[name].state_dict())
    quantize_draft(fake, DraftQuantConfig(group_size=None, backend="fake"))
    snapshot = quantize_draft(modules, DraftQuantConfig(group_size=None, backend="marlin"))
    assert all(isinstance(m.weight, MarlinWeight) for m in modules.values())
    out = modules["b"](modules["a"](x))
    ref = fake["b"](fake["a"](x))
    assert out.dtype == torch.float32 and _relative_error(out, ref) < 3e-2
    restore_draft(snapshot)
    for name, m in modules.items():
        assert type(m.weight) is nn.Parameter
        torch.testing.assert_close(m.weight.data, originals[name], atol=0, rtol=0)


@requires_marlin
def test_unsupported_operations_raise() -> None:
    stored = marlin_weight(quantize_rtn(torch.randn(128, 128, device="cuda"), make_grid("int8", True)), torch.bfloat16)
    with pytest.raises(NotImplementedError, match="dequantize"):
        stored + 1
