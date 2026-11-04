"""The ``"machete"`` quantization backend (vLLM's Machete weight-only GEMM on Hopper).

The padding helper and the configuration checks run everywhere. Everything that calls the kernels is skipped unless
vLLM's Machete ops load and the GPU has compute capability 9.0 (H100 / H200).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from economize.errors import ConfigError
from economize.techniques.quant import (
    QUANT_BACKENDS,
    DraftQuantConfig,
    QuantConfig,
    make_grid,
    quantize_draft,
    quantize_model,
    quantize_rtn,
    restore_draft,
)
from economize.techniques.quant.machete import (
    MacheteWeight,
    machete_available,
    machete_layout_error,
    machete_padded_nk,
    machete_weight,
)

HAS_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
requires_machete = pytest.mark.skipif(
    not (HAS_SM90 and machete_available()), reason="needs vLLM's Machete kernels and a Hopper GPU (sm90)"
)

#: (format, symmetric, group_size): every layout the backend stores.
LAYOUTS = [
    ("int4", True, 128),
    ("int4", True, 64),
    ("int4", True, None),
    ("int4", False, 128),
    ("int4", False, 64),
    ("int8", True, None),
    ("int8", True, 128),
]


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).abs().max() / expected.float().abs().max()).item()


# ---- layout helper and configuration (no kernels) ------------------------------------------------------------------


def test_is_registered() -> None:
    assert "machete" in QUANT_BACKENDS.names()


@pytest.mark.parametrize(
    ("n", "k", "group", "expected"),
    [(5120, 25600, -1, (5120, 25600)), (200, 384, 128, (256, 384)), (64, 96, 64, (128, 128)), (32, 64, -1, (128, 64))],
)
def test_padding_reaches_the_prepacked_block(n: int, k: int, group: int, expected: tuple[int, int]) -> None:
    padded = machete_padded_nk(n, k, group)
    assert padded == expected
    assert padded[0] % 128 == 0 and padded[1] % 64 == 0 and (group < 0 or padded[1] % group == 0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(format="int4", group_size=32), "group_size"),
        (dict(format="int4", group_size=256), "group_size"),
        (dict(format="fp8", group_size=128), "int4 and int8"),
        (dict(format="int3", group_size=128), "int4 and int8"),
        (dict(format="int8", group_size=128, symmetric=False), "no exact zero points"),
        (dict(format="int8", group_size=None, symmetric=False), "no exact zero points"),
    ],
)
def test_config_rejects_layouts_without_a_kernel(kwargs: dict, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        QuantConfig(backend="machete", **kwargs)


def test_affine_int8_is_refused_and_affine_int4_is_kept() -> None:
    # bf16 rounds the kernel's shift -scale * zero by up to half a step for 8-bit zero points, exactly for 4-bit ones
    for group in (None, 64, 128):
        reason = machete_layout_error("int8", group, symmetric=False)
        assert reason is not None and "symmetric=True" in reason
        assert machete_layout_error("int8", group, symmetric=True) is None
        assert machete_layout_error("int4", group, symmetric=False) is None


@requires_machete
def test_machete_weight_refuses_affine_int8_codes() -> None:
    weight = torch.randn(256, 512, device="cuda") * 0.05
    q = quantize_rtn(weight, make_grid("int8", False), 128)
    with pytest.raises(ConfigError, match="no exact zero points"):
        machete_weight(q, torch.bfloat16)


def test_draft_config_accepts_machete() -> None:
    assert DraftQuantConfig(group_size=None, backend="machete").backend == "machete"


# ---- kernels ------------------------------------------------------------------------------------------------------


@requires_machete
@pytest.mark.parametrize(("fmt", "symmetric", "group"), LAYOUTS)
@pytest.mark.parametrize(("n", "k"), [(512, 1024), (200, 384)])
def test_linear_matches_fake_quantization(fmt: str, symmetric: bool, group: int | None, n: int, k: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randn(n, k, device="cuda", generator=gen) * 0.05
    q = quantize_rtn(weight, make_grid(fmt, symmetric), group)
    stored = machete_weight(q, torch.bfloat16)
    assert isinstance(stored, MacheteWeight) and stored.shape == (n, k) and stored.dtype == torch.bfloat16
    dense = q.dequantize(torch.float32)
    # the weight read back through the kernel equals the reference up to bf16 rounding of scales and zero terms
    assert _relative_error(stored.dequantize(), dense) < 1e-2
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=gen)
    for m in (1, 7, 33, 130):
        x = torch.randn(2, m, k, device="cuda", dtype=torch.bfloat16, generator=gen)
        out = F.linear(x, stored, bias)
        assert out.shape == (2, m, n) and out.dtype == torch.bfloat16
        ref = F.linear(x.float(), dense, bias.float())
        assert _relative_error(out, ref) < 2e-2, (fmt, symmetric, group, m)


@requires_machete
def test_views_keep_the_machete_kernel() -> None:
    q = quantize_rtn(torch.randn(256, 256, device="cuda") * 0.05, make_grid("int4", True), 128)
    stored = machete_weight(q, torch.float32)
    for view in (stored.detach(), stored.clone(), stored.to(torch.bfloat16)):
        assert type(view) is MacheteWeight
    x = torch.randn(3, 256, device="cuda")
    out = F.linear(x, stored)
    assert out.dtype == torch.float32 and _relative_error(out, x @ q.dequantize().T) < 2e-2


@requires_machete
def test_cuda_graph_replay_matches_eager() -> None:
    weight = torch.randn(1024, 512, device="cuda") * 0.05
    stored = machete_weight(quantize_rtn(weight, make_grid("int4", True), 128), torch.bfloat16)
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


@requires_machete
@pytest.mark.parametrize(
    "kwargs",
    [
        {"format": "int4", "group_size": 128},
        {"format": "int4", "symmetric": True, "group_size": 64},
        {"format": "int8", "symmetric": True, "group_size": None},
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
    reference = fake.float()(tokens).logits
    model = quantized("machete")
    layer = model.model.layers[0]
    assert isinstance(layer.mlp.down_proj.weight, MacheteWeight)
    if kwargs["group_size"] in (None, 64):  # inputs of 64 are skipped at group 128 (not divisible)
        assert isinstance(layer.mlp.up_proj.weight, MacheteWeight)
        assert isinstance(layer.self_attn.k_proj.weight, MacheteWeight)  # padded from 32 x 64
    noise = _relative_error(fake_bf16, reference)
    assert _relative_error(model(tokens).logits, reference) < 2 * noise + 1e-2, noise


@requires_machete
def test_draft_quantization_and_restore() -> None:
    torch.manual_seed(0)
    modules = {"a": nn.Linear(256, 512, device="cuda"), "b": nn.Linear(512, 128, device="cuda")}
    originals = {name: m.weight.detach().clone() for name, m in modules.items()}
    x = torch.randn(5, 256, device="cuda")
    fake = {name: nn.Linear(m.in_features, m.out_features, device="cuda") for name, m in modules.items()}
    for name, m in fake.items():
        m.load_state_dict(modules[name].state_dict())
    quantize_draft(fake, DraftQuantConfig(group_size=None, backend="fake"))
    snapshot = quantize_draft(modules, DraftQuantConfig(group_size=None, backend="machete"))
    assert all(isinstance(m.weight, MacheteWeight) for m in modules.values())
    out = modules["b"](modules["a"](x))
    ref = fake["b"](fake["a"](x))
    assert out.dtype == torch.float32 and _relative_error(out, ref) < 3e-2
    restore_draft(snapshot)
    for name, m in modules.items():
        assert type(m.weight) is nn.Parameter
        torch.testing.assert_close(m.weight.data, originals[name], atol=0, rtol=0)
