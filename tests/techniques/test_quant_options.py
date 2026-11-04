"""Per-method and per-backend options of the weight quantizers.

* A config field that the chosen method or backend does not read must keep its default (``check_options``).
* GPTQ's ``static_groups=False`` (the reference's plain ``act_order``) returns a column-to-group map; only backends
  with ``supports_g_idx`` accept it.
* ``keep_zeros`` can be forced either way.
* torchao tensor formats are a registry, and Marlin's layout rules are one shared check.
* What a kernel accepts for a format is a ``KernelConstraints`` registered with the format; the backends check it
  generically, so a new format brings its own rules without a backend change.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from economize.errors import ConfigError, UnsupportedModelError
from economize.techniques._weights import check_options
from economize.techniques.prune import token_batches
from economize.techniques.quant import (
    MARLIN_FORMATS,
    QUANT_BACKENDS,
    TORCHAO_CONSTRAINTS,
    TORCHAO_TENSORS,
    DraftQuantConfig,
    KernelConstraints,
    MarlinFormat,
    QParams,
    QuantConfig,
    QuantSpec,
    gptq_quantize,
    make_grid,
    marlin_layout_error,
    quantize_model,
    register_torchao_format,
)
from economize.techniques.quant.constraints import check_devices


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "rtn", "blocksize": 64},
        {"method": "rtn", "percdamp": 0.05},
        {"method": "rtn", "keep_zeros": True},
        {"method": "awq", "act_order": True},
        {"method": "gptq", "awq_grid": 10},
        {"method": "gptq", "awq_clip": False},
        {"method": "awq", "static_groups": False},
    ],
)
def test_knobs_the_method_does_not_read_are_rejected(kwargs: dict) -> None:
    name = next(k for k in kwargs if k != "method")
    with pytest.raises(ConfigError, match=name):
        QuantConfig(group_size=32, **kwargs)


def test_knobs_the_method_reads_are_accepted() -> None:
    QuantConfig(method="gptq", group_size=32, act_order=True, blocksize=64, percdamp=0.05, keep_zeros=False)
    QuantConfig(method="awq", group_size=32, awq_grid=10, awq_clip=False)
    QuantConfig(method="rtn", blocksize=128, percdamp=0.01)  # defaults are always fine


def test_check_options_names_what_the_owner_reads() -> None:
    @dataclass
    class Config:
        a: int = 0
        b: int = 0

    check_options(Config(a=1), "x", {"a"}, ("a", "b"))
    with pytest.raises(ConfigError, match=r"b=2 has no effect with x .* reads a"):
        check_options(Config(b=2), "x", {"a"}, ("a", "b"))


def test_both_configs_satisfy_the_quant_spec() -> None:
    assert isinstance(QuantConfig(), QuantSpec) and isinstance(DraftQuantConfig(), QuantSpec)
    assert DraftQuantConfig().fp8_activations is False


# ---- GPTQ groups and zeros ----------------------------------------------------------------------------------------


def test_dynamic_act_order_groups_return_a_column_to_group_map(correlated_inputs) -> None:
    weight = torch.randn(24, 64, generator=torch.Generator().manual_seed(0))
    inputs = correlated_inputs(64, count=1024)
    hessian = inputs.T @ inputs
    grid = make_grid("int4")
    static = gptq_quantize(weight, hessian, grid, 16, act_order=True, blocksize=16)
    dynamic = gptq_quantize(weight, hessian, grid, 16, act_order=True, blocksize=16, static_groups=False)
    assert static.g_idx is None and dynamic.g_idx is not None
    perm = torch.argsort(torch.diag(hessian), descending=True, stable=True)
    assert dynamic.g_idx.shape == (64,) and torch.equal(dynamic.g_idx[perm], torch.arange(64) // 16)
    # the map is what dequantize uses: every code decodes with its own group's parameters
    zero = None if dynamic.zero is None else dynamic.zero[:, dynamic.g_idx]
    manual = grid.decode(dynamic.codes, QParams(dynamic.scale[:, dynamic.g_idx], zero))
    torch.testing.assert_close(dynamic.dequantize(), manual)
    err = lambda q: (inputs @ (weight - q.dequantize()).T).pow(2).sum()
    assert err(dynamic) < 1.5 * err(static)
    without_groups = gptq_quantize(weight, hessian, grid, None, act_order=True, static_groups=False)
    assert without_groups.g_idx is None


def test_dynamic_groups_need_a_backend_with_g_idx() -> None:
    QuantConfig(method="gptq", group_size=32, act_order=True, static_groups=False)
    for backend in ("torchao", "marlin"):
        if backend in QUANT_BACKENDS and not QUANT_BACKENDS.get(backend).supports_g_idx:
            with pytest.raises(ConfigError, match=r"g_idx|static_groups"):
                QuantConfig(
                    method="gptq", format="int4", group_size=128, symmetric=True, act_order=True,
                    static_groups=False, backend=backend,
                )


def test_dynamic_groups_end_to_end_on_the_fake_backend(tiny2, adapter_of, calib) -> None:
    config = QuantConfig(method="gptq", group_size=32, act_order=True, static_groups=False)
    summary = quantize_model(adapter_of(tiny2), config, token_batches(calib))
    assert summary["modules"] > 0


def test_keep_zeros_can_be_forced() -> None:
    weight = torch.randn(16, 64, generator=torch.Generator().manual_seed(2))
    weight[0, :2] = 0  # 2 of 1024 entries: below the automatic threshold
    hessian = torch.randn(256, 64, generator=torch.Generator().manual_seed(3))
    hessian = hessian.T @ hessian
    grid = make_grid("int4")
    forced = gptq_quantize(weight, hessian, grid, 32, keep_zeros=True).dequantize()
    assert (forced[0, :2] == 0).all()
    auto = gptq_quantize(weight, hessian, grid, 32).dequantize()
    off = gptq_quantize(weight, hessian, grid, 32, keep_zeros=False).dequantize()
    torch.testing.assert_close(auto, off)


# ---- backends -----------------------------------------------------------------------------------------------------


def test_torchao_formats_are_a_registry() -> None:
    assert {"int8", "int4", "fp8"} <= set(TORCHAO_TENSORS.names())
    with pytest.raises(ConfigError, match="torchao"):
        QuantConfig(backend="torchao", format="int3")

    @TORCHAO_TENSORS.register("int3")
    def int3(q, dtype, config=None):  # type: ignore[no-untyped-def]
        return q.dequantize(dtype)

    try:
        QuantConfig(backend="torchao", format="int3", group_size=32)
    finally:
        TORCHAO_TENSORS.unregister("int3")


def test_a_torchao_format_registers_what_its_kernel_accepts() -> None:
    constraints = KernelConstraints("int3 kernel", group_sizes=(32,), symmetric="always", dtypes=(torch.bfloat16,))

    @register_torchao_format("int3", constraints)
    def int3(q, dtype, config=None):  # type: ignore[no-untyped-def]
        return q.dequantize(dtype)

    try:
        assert TORCHAO_CONSTRAINTS.get("int3") is constraints
        with pytest.raises(ConfigError, match=r"int3 kernel needs group_size in \(32,\), got 64"):
            QuantConfig(backend="torchao", format="int3", group_size=64)
        with pytest.raises(ConfigError, match="int3 kernel has no zero points: use symmetric=True"):
            QuantConfig(backend="torchao", format="int3", group_size=32, symmetric=False)
        config = QuantConfig(backend="torchao", format="int3", group_size=32, symmetric=True)
        with pytest.raises(UnsupportedModelError, match=r"int3 kernel needs bfloat16 weights; lin is torch\.float32"):
            check_devices(TORCHAO_CONSTRAINTS, {"lin": nn.Linear(4, 4)}, config)
        check_devices(TORCHAO_CONSTRAINTS, {"lin": nn.Linear(4, 4).to(torch.bfloat16)}, config)
    finally:
        TORCHAO_TENSORS.unregister("int3")
        TORCHAO_CONSTRAINTS.unregister("int3")


def test_torchao_int4_needs_a_gpu_and_bfloat16() -> None:
    config = QuantConfig(backend="torchao", format="int4", group_size=128, symmetric=True)
    with pytest.raises(UnsupportedModelError, match=r"tinygemm int4 needs a CUDA GPU with compute capability 8\.0"):
        check_devices(TORCHAO_CONSTRAINTS, {"lin": nn.Linear(4, 4).to(torch.bfloat16)}, config)


def test_a_marlin_format_is_one_registration() -> None:
    assert marlin_layout_error("int2", None, True) == (
        "the marlin backend stores int4, int8 and fp8, got format 'int2'")
    MARLIN_FORMATS.register("int2", MarlinFormat(KernelConstraints("Marlin int2", group_sizes=(None, 128)), "uint2b2"))
    try:
        assert marlin_layout_error("int2", 128, False) is None
        assert marlin_layout_error("int2", 64, True) == "Marlin int2 needs group_size None or 128, got 64"
        assert "int2, int4, int8 and fp8" in str(marlin_layout_error("int3", None, True))
    finally:
        MARLIN_FORMATS.unregister("int2")


def test_fp8_activations_need_fp8_weights() -> None:
    with pytest.raises(ConfigError, match="fp8_activations"):
        QuantConfig(backend="torchao", format="int8", symmetric=True, group_size=None, fp8_activations=True)


@pytest.mark.parametrize(
    ("format", "group_size", "symmetric", "match"),
    [
        ("int3", 128, True, "int4, int8 and fp8"),
        ("int4", 48, True, "group_size"),
        ("int8", None, False, "symmetric"),
        ("int4", 128, False, None),  # int4 has zero points
        ("int4", 128, True, None),
        ("fp8", 128, True, None),
    ],
)
def test_marlin_layout_rules_are_one_check(format: str, group_size: int, symmetric: bool, match: str | None) -> None:
    error = marlin_layout_error(format, group_size, symmetric)
    if match is None:
        assert error is None
    else:
        assert error is not None and match in error
        with pytest.raises(ConfigError, match=match):
            QuantConfig(backend="marlin", format=format, group_size=group_size, symmetric=symmetric)


def test_awq_scale_groups_name_their_unclipped_linears() -> None:
    from economize.techniques.quant import AWQ_SCALE_GROUPS, ScaleGroup

    qkv = AWQ_SCALE_GROUPS.get("llama")[0]
    assert qkv.unclipped() == ("self_attn.q_proj", "self_attn.k_proj")
    by_name = ScaleGroup("ln", ("attn.query", "attn.key", "attn.value"))
    assert by_name.unclipped() == ("attn.query", "attn.key")  # llm-awq's name test when the group says nothing
    assert ScaleGroup("ln", ("attn.wq", "attn.wk", "attn.wv"), no_clip=("attn.wq", "attn.wk")).unclipped() == (
        "attn.wq", "attn.wk",
    )
