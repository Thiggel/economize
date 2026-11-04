"""NVFP4 and MXFP4 weight formats: bit-exact agreement with compressed-tensors (the format vLLM loads), the shared
global scale of fused shards, the packed checkpoint, and which kernels and serving engines run them."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import economize
from economize import Saving, SupportStatus
from economize.backends import UnsupportedBackendError
from economize.errors import ConfigError
from economize.techniques.quant import (
    E2M1_VALUES,
    Mxfp4Grid,
    Nvfp4Grid,
    QuantConfig,
    make_grid,
    quantize_model,
    quantize_rtn,
)
from economize.techniques.quant.fp4 import e2m1_round, mx_scale, nvfp4_global_scale
from economize.techniques.quant.grid import bind_shards

HAS_CT = importlib.util.find_spec("compressed_tensors") is not None
needs_ct = pytest.mark.skipif(not HAS_CT, reason="needs compressed-tensors")
GROUPS = {"nvfp4": 16, "mxfp4": 32}


def _weight(seed: int) -> torch.Tensor:
    weight = torch.randn(48, 128, generator=torch.Generator().manual_seed(seed)) * 0.05
    weight[3, 5] = 0.9  # an outlier group
    weight[7, :32] = 0.0  # an all-zero group (NVFP4 replaces its zero scale by 0.125)
    weight[9, 32:64] = -1e-4  # negative values that round to zero keep their sign bit
    return weight


def _ct_args(fmt: str) -> Any:
    from compressed_tensors.quantization import QuantizationArgs

    if fmt == "nvfp4":
        return QuantizationArgs(num_bits=4, type="float", symmetric=True, strategy="tensor_group", group_size=16,
                                scale_dtype=torch.float8_e4m3fn, zp_dtype=torch.float8_e4m3fn)
    return QuantizationArgs(num_bits=4, type="float", symmetric=True, strategy="group", group_size=32,
                            scale_dtype=torch.uint8, zp_dtype=torch.uint8)


def _ct_reference(weight: torch.Tensor, fmt: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """compressed-tensors' qparams and fake quantization of ``weight`` (its weight observer's min/max path)."""
    from compressed_tensors.quantization.lifecycle.forward import fake_quantize
    from compressed_tensors.quantization.utils.helpers import calculate_qparams, generate_gparam

    args = _ct_args(fmt)
    global_scale = None
    if fmt == "nvfp4":
        global_scale = generate_gparam(weight.amin().clamp(max=0)[None], weight.amax().clamp(min=0)[None])
    blocks = weight.reshape(weight.shape[0], -1, args.group_size)
    scale, zero = calculate_qparams(blocks.amin(-1).clamp(max=0), blocks.amax(-1).clamp(min=0), args,
                                    global_scale=global_scale)
    return scale, fake_quantize(weight, scale, zero, args, global_scale=global_scale), global_scale


@needs_ct
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_fp4_matches_compressed_tensors(fmt: str, seed: int) -> None:
    """Scales, global scale and dequantized weights equal compressed-tensors' ``calculate_qparams`` /
    ``generate_gparam`` / ``fake_quantize`` bit for bit."""
    weight = _weight(seed)
    qweight = quantize_rtn(weight, make_grid(fmt), GROUPS[fmt])
    scale, reference, global_scale = _ct_reference(weight, fmt)
    assert torch.equal(qweight.dequantize(), reference)
    if fmt == "nvfp4":
        grid = qweight.grid
        assert isinstance(grid, Nvfp4Grid) and grid.global_scale == float(global_scale)  # type: ignore[arg-type]
        assert torch.equal(grid.local_scale(qweight.scale), scale.float())
    else:
        assert torch.equal(qweight.scale, scale.float())
    assert qweight.bits_per_weight == {"nvfp4": 4.5, "mxfp4": 4.25}[fmt]


@needs_ct
def test_e2m1_rounding_and_packing_follow_compressed_tensors() -> None:
    """Every midpoint (and its neighbours) rounds as ``cast_to_fp4``, and the codes pack as ``pack_fp4_to_uint8``."""
    from compressed_tensors.compressors.nvfp4.helpers import pack_fp4_to_uint8
    from compressed_tensors.quantization.utils.fp4_utils import cast_to_fp4

    points = torch.tensor([0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 7.0, 0.2499, 0.2501, 3.4999, 3.5001])
    x = torch.cat([points, -points])
    assert torch.equal(torch.tensor(E2M1_VALUES)[e2m1_round(x)] * x.sign(), cast_to_fp4(x))
    grid = Mxfp4Grid()
    codes = grid.encode(x[None, :].repeat(2, 1)[:, :28], SimpleNamespace(scale=torch.ones(1)))  # type: ignore[arg-type]
    values = grid.decode(codes, SimpleNamespace(scale=torch.ones(1)))  # type: ignore[arg-type]
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    assert torch.equal(packed, pack_fp4_to_uint8(values))


def test_scales_from_the_definitions() -> None:
    """MXFP4: ``2^(floor(log2 p) - 2)`` with ``p`` the maximum rounded to a power of two (up from mantissa 1.75);
    NVFP4: ``448 * 6 / amax``. Written from the OCP MX v1.0 spec and the NVFP4 recipe, independent of the package."""
    amax = torch.tensor([6.0, 7.0, 6.99, 0.75, 1.0, 0.0])
    assert mx_scale(amax).tolist() == [1.0, 2.0, 1.0, 0.125, 0.25, 2.0**-127]
    assert nvfp4_global_scale([torch.tensor([[0.5, -2.0]]), torch.tensor([[1.0]])]) == pytest.approx(448 * 6 / 2)
    assert nvfp4_global_scale([torch.zeros(2, 2)]) > 0


def test_fused_shards_share_the_global_scale() -> None:
    """q/k/v and gate/up of one layer bind one NVFP4 grid (vLLM fuses them into one GEMM with one global scale),
    every other map its own; MXFP4 needs no binding."""
    from helpers import make_tiny_model

    layer = make_tiny_model(num_layers=1).model.layers[0]
    modules = {name: module for name, module in layer.named_modules() if isinstance(module, torch.nn.Linear)}
    grids = bind_shards(modules, make_grid("nvfp4"))
    assert grids["self_attn.q_proj"] is grids["self_attn.k_proj"] is grids["self_attn.v_proj"]
    assert grids["mlp.gate_proj"] is grids["mlp.up_proj"]
    assert grids["self_attn.o_proj"] is not grids["self_attn.q_proj"]
    qkv = [modules[f"self_attn.{p}_proj"].weight for p in "qkv"]
    assert grids["self_attn.q_proj"].global_scale == nvfp4_global_scale(qkv)  # type: ignore[attr-defined]
    mx = make_grid("mxfp4")
    assert all(grid is mx for grid in bind_shards(modules, mx).values())


@pytest.mark.parametrize("method", ["rtn", "gptq"])
def test_methods_quantize_fused_shards_with_one_global_scale(method: str) -> None:
    from economize.arch import adapter_for
    from economize.techniques.quant.grid import QuantizedWeight
    from helpers import VOCAB, make_tiny_model

    model = make_tiny_model(num_layers=2)
    calib = [torch.randint(0, VOCAB, (2, 16), generator=torch.Generator().manual_seed(0))]
    quantize_model(adapter_for(model), QuantConfig(method=method, format="nvfp4", group_size=16,
                                                   backend="compressed-tensors"), calib)
    from economize.integrations.compressed_tensors import quantized_modules

    records: dict[str, QuantizedWeight] = quantized_modules(model)
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        scales = {records[prefix + f"self_attn.{p}_proj"].grid.global_scale for p in "qkv"}  # type: ignore[attr-defined]
        assert len(scales) == 1
        assert records[prefix + "mlp.gate_proj"].grid == records[prefix + "mlp.up_proj"].grid


@pytest.mark.parametrize(("fmt", "group_size"), [("nvfp4", 32), ("nvfp4", None), ("mxfp4", 16), ("mxfp4", 128)])
def test_fp4_refuses_other_group_sizes(fmt: str, group_size: int | None) -> None:
    with pytest.raises(ConfigError, match=f"set group_size={GROUPS[fmt]}"):
        QuantConfig(format=fmt, group_size=group_size)


@pytest.mark.parametrize("backend", ["marlin", "machete", "torchao"])
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_integer_kernels_refuse_fp4(fmt: str, backend: str) -> None:
    with pytest.raises(ConfigError):
        QuantConfig(format=fmt, group_size=GROUPS[fmt], backend=backend)


def test_hqq_refuses_fp4() -> None:
    with pytest.raises(ConfigError):
        QuantConfig(method="hqq", format="nvfp4", group_size=16)


@needs_ct
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_fp4_checkpoint_round_trips(tmp_path: Path, fmt: str) -> None:
    """The packed checkpoint decompresses (compressed-tensors' own compressor, in float32) to exactly the weights
    economize computes with, and ``transformers`` loads it."""
    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.quantization import QuantizationScheme
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    from economize.arch import adapter_for
    from economize.integrations.compressed_tensors import export_compressed_tensors, quantized_modules
    from helpers import make_tiny_model

    model = make_tiny_model(num_layers=2)
    quantize_model(adapter_for(model), QuantConfig(format=fmt, group_size=GROUPS[fmt], backend="compressed-tensors"))
    records = quantized_modules(model)
    export_compressed_tensors(model, tmp_path)
    quantization = json.loads((tmp_path / "config.json").read_text())["quantization_config"]
    assert quantization["format"] == f"{fmt}-pack-quantized"
    (group,) = quantization["config_groups"].values()
    assert group["weights"]["strategy"] == ("tensor_group" if fmt == "nvfp4" else "group")
    assert group["weights"]["group_size"] == GROUPS[fmt] and group["input_activations"] is None
    state = load_file(str(tmp_path / "model.safetensors"))
    compressor = BaseCompressor.get_value_from_registry(f"{fmt}-pack-quantized")
    scheme = QuantizationScheme(targets=["Linear"], weights=_ct_args(fmt), format=f"{fmt}-pack-quantized")
    for name, qweight in records.items():
        local = {key.removeprefix(f"{name}."): value for key, value in state.items() if key.startswith(f"{name}.")}
        assert torch.equal(local["weight_packed"], qweight.codes[:, 0::2] | (qweight.codes[:, 1::2] << 4))
        if fmt == "nvfp4":
            assert local["weight_scale"].dtype == torch.float8_e4m3fn
            assert float(local["weight_global_scale"]) == qweight.grid.global_scale  # type: ignore[attr-defined]
        else:
            assert local["weight_scale"].dtype == torch.uint8
        from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
        from compressed_tensors.quantization.lifecycle.forward import dequantize

        codes = unpack_fp4_from_uint8(local["weight_packed"], *qweight.codes.shape, dtype=torch.float32)
        scale = compressor._decompress_scale(local["weight_scale"], torch.float32)
        weight = dequantize(codes, scale, args=scheme.weights, global_scale=local.get("weight_global_scale"),
                            dtype=torch.float32)
        assert torch.equal(weight, qweight.dequantize()), name
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16).eval()
    reloaded(torch.arange(1, 9)[None])  # the first forward decompresses the packed weights (in bfloat16)
    theirs = dict(reloaded.named_modules())
    for name, qweight in records.items():
        torch.testing.assert_close(theirs[name].weight.float(), qweight.dequantize(), rtol=1e-2, atol=1e-6)


# ---- serving ------------------------------------------------------------------------------------------------------
def _config(fmt: str = "nvfp4", **kwargs: Any) -> SimpleNamespace:
    fields = {"backend": "compressed-tensors", "format": fmt, "group_size": GROUPS.get(fmt), "symmetric": True,
              "activations": None, "fp8_activations": False}
    return SimpleNamespace(config=SimpleNamespace(**{**fields, **kwargs}))


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_vllm_serves_fp4_on_marlin_and_declares_the_latency_simulated(fmt: str) -> None:
    from economize.backends.vllm import serving

    adapter: Any = serving.base.adapter_for("quant")
    technique = _config(fmt)
    adapter.check(technique, serving.PlanBuilder(None, None))
    assert adapter.level(technique) is serving.Support.NATIVE
    simulation = adapter.simulation_of(technique)
    assert simulation is not None and simulation.savings == Saving.LATENCY and "sm100" in simulation.reason
    assert adapter.simulation_of(_config("int4", group_size=128)) is None


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_sglang_refuses_fp4(fmt: str) -> None:
    from economize.backends.sglang import serving as sglang_serving

    adapter: Any = sglang_serving.base.adapter_for("quant")
    with pytest.raises(UnsupportedBackendError, match="no SGLang kernel"):
        adapter.check(_config(fmt), None)
    assert adapter.level(_config(fmt)) is sglang_serving.Support.UNSUPPORTED


@needs_ct
def test_backend_support_reports_fp4_paths() -> None:
    support = economize.backend_support(QuantConfig(format="nvfp4", group_size=16, backend="compressed-tensors"))
    assert support["transformers"].status is SupportStatus.SIMULATED
    assert support["vllm"].status is SupportStatus.SIMULATED
    assert support["vllm"].simulation is not None and support["vllm"].simulation.savings == Saving.LATENCY
    assert support["sglang"].status is SupportStatus.UNSUPPORTED
    int4 = economize.backend_support(QuantConfig(format="int4", group_size=128, backend="compressed-tensors"))
    assert int4["vllm"].status is SupportStatus.REAL


@needs_ct
def test_vllm_plan_names_the_marlin_fp4_kernel(tmp_path: Path) -> None:
    from economize.backends.vllm.serving import to_vllm
    from helpers import make_tiny_model

    folder = tmp_path / "tiny"
    make_tiny_model(seed=0).save_pretrained(folder)
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32).eval()
    model = economize.apply(base, QuantConfig(format="nvfp4", group_size=16, backend="compressed-tensors"))
    plan = to_vllm(model, tmp_path / "out")
    assert plan.techniques == {"quant": "native"}
    assert any("MarlinNvFp4LinearKernel" in note for note in plan.notes["quant"])
    quantization = json.loads((Path(plan.model) / "config.json").read_text())["quantization_config"]
    assert quantization["format"] == "nvfp4-pack-quantized"
