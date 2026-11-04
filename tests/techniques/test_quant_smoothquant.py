"""SmoothQuant and W8A8 activation quantization: the scales of the official ``smooth_ln_fcs``, the invariance of the
float model under smoothing, the activation schemes, the torchao ``torch._int_mm`` path and the compressed-tensors
export with ``input_activations``."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from economize.errors import ConfigError
from economize.techniques.prune import token_batches
from economize.techniques.prune.calibration import LayerwiseCalibration, linear_modules
from economize.techniques.quant import (
    ACTIVATION_SCHEMES,
    QUANT_BACKENDS,
    QuantConfig,
    make_grid,
    quantize_model,
    quantize_rtn,
    smoothquant_scales,
)
from economize.techniques.quant import activations as act
from economize.techniques.quant.smoothquant import AbsMaxObserver, smooth_layer

HAS_CT = importlib.util.find_spec("compressed_tensors") is not None
W8A8 = {"format": "int8", "group_size": None, "symmetric": True, "activations": "int8"}


def _official_smooth_ln_fcs(ln: nn.Module, fcs: list[nn.Linear], act_scales: torch.Tensor, alpha: float) -> None:
    """``smoothquant.smooth.smooth_ln_fcs`` (mit-han-lab/smoothquant), verbatim apart from the device moves."""
    weight_scales = torch.cat([fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5)
    ln.weight.div_(scales)
    if getattr(ln, "bias", None) is not None:
        ln.bias.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@pytest.mark.parametrize("alpha", [0.0, 0.5, 0.85, 1.0])
def test_smoothing_matches_the_official_smooth_ln_fcs(alpha: float) -> None:
    from economize.techniques.quant.awq import apply_scale

    generator = torch.Generator().manual_seed(0)
    ln = nn.LayerNorm(32)
    ln.weight.data = torch.rand(32, generator=generator) + 0.5
    ln.bias.data = torch.randn(32, generator=generator)
    fcs = [nn.Linear(32, n, bias=False) for n in (16, 8, 8)]
    for fc in fcs:
        fc.weight.data = torch.randn(fc.weight.shape, generator=generator)
    x = torch.randn(64, 32, generator=generator) * torch.linspace(0.1, 20, 32)
    observer = AbsMaxObserver(fcs[0])
    observer.update(x[:40])
    observer.update(x[40:])
    torch.testing.assert_close(observer.max, x.abs().amax(0))
    ours_ln, ours_fcs = copy.deepcopy(ln), copy.deepcopy(fcs)
    apply_scale(ours_ln, ours_fcs, smoothquant_scales(observer.max, [fc.weight for fc in ours_fcs], alpha))
    _official_smooth_ln_fcs(ln, fcs, x.abs().amax(0), alpha)
    torch.testing.assert_close(ours_ln.weight, ln.weight)
    torch.testing.assert_close(ours_ln.bias, ln.bias)
    for ours, reference in zip(ours_fcs, fcs, strict=True):
        torch.testing.assert_close(ours.weight, reference.weight)


def _smooth_only(model: nn.Module, adapter_of, calib: torch.Tensor, alpha: float) -> dict[int, list[str]]:
    adapter = adapter_of(model)
    smoothed = {}
    for step in LayerwiseCalibration(adapter, token_batches(calib), propagate="original"):
        scales = smooth_layer(step, linear_modules(adapter, step.layer), alpha, adapter)
        smoothed[step.index] = sorted(scales)
    return smoothed


def test_smoothing_leaves_the_float_model_unchanged(tiny2, adapter_of, calib, held_out) -> None:
    reference = tiny2(held_out).logits
    smoothed = _smooth_only(tiny2, adapter_of, calib, alpha=0.5)
    assert list(smoothed.values()) == [["input_layernorm", "post_attention_layernorm"]] * 2
    torch.testing.assert_close(tiny2(held_out).logits, reference, atol=1e-4, rtol=1e-4)


def test_smoothing_gemma4_uses_the_derived_norm_groups(adapter_of, calib, held_out) -> None:
    from helpers import family_skip_reason, make_family_model

    if reason := family_skip_reason("gemma4_text"):
        pytest.skip(reason)
    pytest.importorskip("transformers.models.gemma4")
    model = make_family_model("gemma4_text").eval()
    vocab = adapter_of(model).vocab_size
    ids, held = calib % vocab, held_out % vocab
    reference = model(held).logits
    smoothed = _smooth_only(model, adapter_of, ids[:4], alpha=0.5)
    assert all(smoothed.values()) and all("v_proj" not in names for names in smoothed.values())
    torch.testing.assert_close(model(held).logits, reference, atol=1e-4, rtol=1e-4)


def _outlier_model(tiny_factory) -> nn.Module:
    """A tiny model whose residual stream has four outlier channels (the case SmoothQuant is made for)."""
    model = tiny_factory(num_layers=2)
    for layer in model.model.layers:
        layer.input_layernorm.weight.data[:4] *= 30
        layer.post_attention_layernorm.weight.data[:4] *= 30
    return model


def test_smoothquant_w8a8_beats_round_to_nearest_w8a8(tiny_factory, adapter_of, calib, held_out) -> None:
    base = _outlier_model(tiny_factory)
    reference = base(held_out).logits
    errors = {}
    for method in ("rtn", "smoothquant"):
        model = copy.deepcopy(base)
        quantize_model(adapter_of(model), QuantConfig(method=method, **W8A8), token_batches(calib))
        errors[method] = (model(held_out).logits - reference).pow(2).mean().item()
    assert errors["smoothquant"] < 0.5 * errors["rtn"]


def test_smoothquant_quantizes_the_smoothed_weights(tiny2, adapter_of, calib) -> None:
    smoothed = copy.deepcopy(tiny2)
    _smooth_only(smoothed, adapter_of, calib, alpha=0.5)
    quantize_model(adapter_of(tiny2), QuantConfig(method="smoothquant", **W8A8), token_batches(calib))
    for index in (0, 1):
        expected = linear_modules(adapter_of(smoothed), smoothed.model.layers[index])
        for name, module in linear_modules(adapter_of(tiny2), tiny2.model.layers[index]).items():
            want = quantize_rtn(expected[name].weight.data, make_grid("int8", symmetric=True), None).dequantize()
            torch.testing.assert_close(module.weight, want, atol=1e-6, rtol=0, msg=name)
            assert act.activation_scheme_of(module) is ACTIVATION_SCHEMES.get("int8"), name


@pytest.mark.parametrize("name", ["int8", "int4", "fp8"])
def test_activation_schemes_quantize_per_token(name: str) -> None:
    scheme = ACTIVATION_SCHEMES.get(name)
    x = torch.randn(3, 5, 64, generator=torch.Generator().manual_seed(0)) * torch.logspace(-2, 1, 64)
    out = scheme.fake(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    if name == "fp8":
        scale = x.abs().amax(-1, keepdim=True) / 448
        torch.testing.assert_close(out, (x / scale).to(torch.float8_e4m3fn).float() * scale)
    else:
        qmax = 2 ** (scheme.bits - 1) - 1
        scale = x.abs().amax(-1, keepdim=True) / qmax
        codes = out / scale
        torch.testing.assert_close(codes, codes.round(), atol=1e-4, rtol=0)
        assert codes.abs().amax() <= qmax + 1e-4
    torch.testing.assert_close(scheme.fake(out), out, atol=1e-5, rtol=1e-5)  # idempotent


def test_fake_backend_hook_quantizes_the_inputs(tiny2, adapter_of, held_out) -> None:
    quantize_model(adapter_of(tiny2), QuantConfig(format="int8", group_size=None, symmetric=True, activations="int8"))
    module = tiny2.model.layers[0].mlp.down_proj
    x = torch.randn(4, module.in_features)
    expected = F.linear(ACTIVATION_SCHEMES.get("int8").fake(x), module.weight)
    torch.testing.assert_close(module(x), expected)
    act.uninstall(module)
    torch.testing.assert_close(module(x), F.linear(x, module.weight))
    assert act.activation_scheme_of(module) is None


def test_activation_hooks_are_removed_when_quantization_fails(tiny2, adapter_of, monkeypatch) -> None:
    backend = QUANT_BACKENDS.get("fake")
    original = backend.apply
    calls = []

    def failing(self, module, qweight, config):
        calls.append(module)
        original(self, module, qweight, config)
        if len(calls) == 9:
            raise RuntimeError("kernel conversion failed")

    monkeypatch.setattr(backend, "apply", failing)
    reference = [p.clone() for p in tiny2.parameters()]
    with pytest.raises(RuntimeError, match="kernel conversion"):
        quantize_model(adapter_of(tiny2), QuantConfig(format="int8", group_size=None, symmetric=True,
                                                      activations="int8"))
    for module in tiny2.modules():
        assert not module._forward_pre_hooks and act.activation_scheme_of(module) is None
    assert all(torch.equal(a, b) for a, b in zip(reference, tiny2.parameters(), strict=True))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": "smoothquant", "format": "int8", "group_size": None}, "activations='int8'"),
        ({"method": "smoothquant", **W8A8, "smooth_alpha": 1.5}, "smooth_alpha"),
        ({"activations": "int3"}, "activation scheme"),
        ({"method": "rtn", "smooth_alpha": 0.8}, "smooth_alpha"),
        ({"backend": "marlin", "activations": "int8"}, "activations"),
        ({"backend": "torchao", "format": "int8", "group_size": 128, "symmetric": True, "activations": "int8"},
         "group_size None"),
        ({"backend": "torchao", "format": "int8", "group_size": None, "symmetric": False, "activations": "int8"},
         "symmetric=True"),
        ({"backend": "torchao", "format": "int4", "group_size": 32, "activations": "int8"}, "activations=int8"),
    ],
)
def test_activation_config_validation(kwargs: dict, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        QuantConfig(**kwargs)


@pytest.mark.skipif(not HAS_CT, reason="needs compressed-tensors")
@pytest.mark.parametrize(("fmt", "activations"), [("int4", "int8"), ("int8", "fp8"), ("fp8", "int8")])
def test_compressed_tensors_refuses_unexportable_activation_pairs(fmt: str, activations: str) -> None:
    import economize.integrations.compressed_tensors  # noqa: F401

    with pytest.raises(ConfigError, match="compressed-tensors exports"):
        QuantConfig(format=fmt, group_size=None, symmetric=True, backend="compressed-tensors", activations=activations)


def test_torchao_w8a8_runs_int_mm(tiny_factory, adapter_of, calib, held_out, monkeypatch) -> None:
    """torchao's int8 tensor with per-row activation quantization multiplies int8 by int8 (``int_scaled_matmul``,
    ``torch._int_mm``); its error is that of the simulated W8A8 (torchao divides by 127.5, so codes move by a step)."""
    pytest.importorskip("torchao")
    from torchao.quantization.quantize_.workflows.int8 import int8_tensor

    calls = []
    real = int8_tensor.int_scaled_matmul

    def counting(*args, **kwargs):
        calls.append(args[0].dtype)
        return real(*args, **kwargs)

    monkeypatch.setattr(int8_tensor, "int_scaled_matmul", counting)
    base = _outlier_model(tiny_factory)
    reference = base(held_out).logits
    errors = {}
    for backend in ("fake", "torchao"):
        model = copy.deepcopy(base)
        config = QuantConfig(method="smoothquant", backend=backend, **W8A8)
        quantize_model(adapter_of(model), config, token_batches(calib))
        errors[backend] = (model(held_out).logits - reference).pow(2).mean().item()
    assert type(model.model.layers[0].mlp.up_proj.weight).__name__ == "Int8Tensor"
    assert act.activation_scheme_of(model.model.layers[0].mlp.up_proj).name == "int8"
    assert calls and set(calls) == {torch.int8}
    assert errors["torchao"] < 3 * errors["fake"] and errors["fake"] < 3 * errors["torchao"]


@pytest.mark.skipif(not HAS_CT, reason="needs compressed-tensors")
@pytest.mark.parametrize(("fmt", "activations", "storage"), [("int8", "int8", "int-quantized"),
                                                            ("fp8", "fp8", "float-quantized")])
def test_w8a8_exports_input_activations(tmp_path: Path, fmt: str, activations: str, storage: str) -> None:
    """The checkpoint declares dynamic per-token ``input_activations`` (vLLM's W8A8 schemes) and ``transformers``
    reloads it with the same weights. Its own activation emulation matches ours exactly for fp8, and for int8 once
    the per-token scale is ``max|x| / 127.5`` (compressed-tensors' symmetric range; vLLM's kernel divides by 127)."""
    from transformers import AutoModelForCausalLM

    from economize.arch import adapter_for
    from economize.integrations.compressed_tensors import export_compressed_tensors
    from helpers import VOCAB, make_tiny_model

    model = make_tiny_model(num_layers=2)
    calib = torch.randint(0, VOCAB, (4, 32), generator=torch.Generator().manual_seed(0))
    config = QuantConfig(method="smoothquant", format=fmt, group_size=None, symmetric=True,
                         backend="compressed-tensors", activations=activations)
    quantize_model(adapter_for(model), config, token_batches(calib))
    export_compressed_tensors(model, tmp_path)
    quantization = json.loads((tmp_path / "config.json").read_text())["quantization_config"]
    assert quantization["format"] == storage
    (group,) = quantization["config_groups"].values()
    inputs = group["input_activations"]
    assert (inputs["num_bits"], inputs["type"], inputs["strategy"], inputs["dynamic"], inputs["symmetric"]) == (
        8, "int" if fmt == "int8" else "float", "token", True, True)
    assert group["weights"]["strategy"] == "channel" and group["weights"]["symmetric"]
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32).eval()
    ids = torch.arange(1, 20)[None]
    logits = reloaded(ids).logits  # the first forward decompresses the int8 / fp8 weights
    ours, theirs = model.state_dict(), reloaded.state_dict()
    assert all(torch.equal(value, theirs[key]) for key, value in ours.items())
    if fmt == "int8":
        def ct_int8(_module: nn.Module, args: tuple) -> tuple:
            scale = args[0].abs().amax(-1, keepdim=True) / 127.5
            return (torch.round(args[0] / scale).clamp(-128, 127) * scale,)

        for module in model.modules():
            if act.activation_scheme_of(module) is not None:
                act.uninstall(module)
                module.register_forward_pre_hook(ct_int8)
    torch.testing.assert_close(logits, model(ids).logits, atol=1e-5, rtol=1e-5)
