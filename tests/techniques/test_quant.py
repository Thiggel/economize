"""Weight-only quantization (RTN, GPTQ, AWQ; int2-8 and fp8), its backends, and drafter-only quantization."""

from __future__ import annotations

import copy
import importlib.util

import pytest
import torch
from torch import nn

from economize.core.metadata import Family, Saving, TechniqueInfo
from economize.core.technique import Capability, Technique
from economize.errors import ConfigError, EconomizeError, IncompatibleTechniquesError, UnknownNameError
from economize.techniques.prune import token_batches
from economize.techniques.prune.calibration import LayerwiseCalibration, linear_modules
from economize.techniques.quant import (
    DraftQuantConfig,
    Fp8Grid,
    IntGrid,
    QuantConfig,
    awq_clip_thresholds,
    awq_scale_layer,
    draft_modules,
    gptq_quantize,
    make_grid,
    quantize_draft,
    quantize_model,
    quantize_rtn,
    quantized_copy,
    restore_draft,
)
from economize.techniques.quant.awq import ScaleGroup, derive_scale_groups
from economize.techniques.quant.methods import awq_scale_groups

requires_weights_contract = pytest.mark.skipif(
    not hasattr(Capability, "WEIGHTS"), reason="needs the economize v0.2 Technique contract (Capability.WEIGHTS)"
)
HAS_TORCHAO = importlib.util.find_spec("torchao") is not None


def _output_error(weight: torch.Tensor, quantized: torch.Tensor, inputs: torch.Tensor) -> float:
    return (inputs @ (weight - quantized).T).pow(2).sum().item()


def _weights(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


# ---- number formats and round to nearest ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("group_size", [None, 16])
def test_rtn_error_is_at_most_half_a_step(bits: int, symmetric: bool, group_size: int | None) -> None:
    weight = torch.randn(32, 64, generator=torch.Generator().manual_seed(bits)) * 0.1
    grid = IntGrid(bits, symmetric)
    q = quantize_rtn(weight, grid, group_size)
    groups = 64 // (group_size or 64)
    assert q.codes.dtype == torch.uint8 and int(q.codes.max()) <= grid.maxq
    assert q.scale.shape == (32, groups) and q.zero is not None and q.zero.shape == (32, groups)
    step = q.scale.repeat_interleave(group_size or 64, dim=1)
    # The row maximum of a symmetric grid sits exactly half a step outside it; allow float32 cancellation there.
    assert ((weight - q.dequantize()).abs() <= step / 2 * (1 + 1e-4)).all()
    if symmetric:
        assert (q.zero == (grid.maxq + 1) / 2).all()


def test_rtn_keeps_zero_exact_and_rejects_indivisible_groups() -> None:
    weight = torch.randn(8, 32, generator=torch.Generator().manual_seed(0))
    weight[:, ::3] = 0
    for symmetric in (False, True):
        dequantized = quantize_rtn(weight, IntGrid(4, symmetric), 16).dequantize()
        assert (dequantized[:, ::3] == 0).all()  # the grid always contains zero (pruned weights stay pruned)
    with pytest.raises(ConfigError, match="group size"):
        quantize_rtn(weight, IntGrid(4), 12)


def test_fp8_relative_error() -> None:
    weight = torch.randn(16, 64, generator=torch.Generator().manual_seed(0))
    q = quantize_rtn(weight, Fp8Grid())
    assert q.codes.dtype == torch.float8_e4m3fn and q.zero is None
    error = (weight - q.dequantize()).abs()
    # e4m3 keeps 3 mantissa bits: relative error <= 2^-4 for normal values, absolute <= 2^-10 * scale below 2^-6.
    assert (error <= weight.abs() / 16 + q.scale * 2**-10 + 1e-7).all()
    row_max = weight.abs().amax(1)
    assert torch.allclose(q.dequantize().abs().amax(1), row_max)  # the row maximum maps to 448 exactly


@pytest.mark.parametrize(
    ("format", "symmetric", "group_size", "bits"),
    [
        ("int4", False, 128, 4 + 32 / 128),
        ("int4", True, 128, 4 + 16 / 128),
        ("int3", False, 32, 3 + 32 / 32),
        ("int8", True, None, 8 + 16 / 256),
        ("fp8", True, None, 8 + 32 / 256),
    ],
)
def test_bits_per_weight(format: str, symmetric: bool, group_size: int | None, bits: float) -> None:
    q = quantize_rtn(torch.randn(4, 256), make_grid(format, symmetric), group_size)
    assert q.bits_per_weight == pytest.approx(bits)


def test_format_registry() -> None:
    assert make_grid("int4").maxq == 15 and make_grid("int8", symmetric=True).symmetric
    assert isinstance(make_grid("fp8"), Fp8Grid)
    with pytest.raises(UnknownNameError):
        make_grid("int5")
    with pytest.raises(ConfigError):
        IntGrid(9)


# ---- GPTQ and AWQ on single matrices -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("format", "group_size", "act_order"),
    [("int4", 32, False), ("int3", None, False), ("int4", 32, True), ("int3", 16, True), ("int8", None, False)],
)
def test_gptq_output_error_below_rtn(correlated_inputs, format: str, group_size: int | None, act_order: bool) -> None:
    weight = torch.randn(48, 128, generator=torch.Generator().manual_seed(1))
    inputs = correlated_inputs(128, count=2048)
    grid = make_grid(format)
    gptq = gptq_quantize(weight, inputs.T @ inputs, grid, group_size, act_order=act_order, blocksize=32)
    rtn = quantize_rtn(weight, grid, group_size)
    # About 0.55 of RTN without act_order and 0.35-0.41 with it, on every seed tried.
    assert _output_error(weight, gptq.dequantize(), inputs) < 0.75 * _output_error(weight, rtn.dequantize(), inputs)
    assert gptq.codes.dtype == torch.uint8 and int(gptq.codes.max()) <= grid.maxq


@pytest.mark.parametrize("group_size", [None, 32])
def test_gptq_with_identity_hessian_is_rtn(group_size: int | None) -> None:
    """Uncorrelated inputs leave GPTQ nothing to compensate: its codes equal round to nearest."""
    weight = torch.randn(24, 64, generator=torch.Generator().manual_seed(0))
    grid = make_grid("int4")
    gptq = gptq_quantize(weight, torch.eye(64), grid, group_size, blocksize=16)
    assert torch.equal(gptq.dequantize(), quantize_rtn(weight, grid, group_size).dequantize())


@pytest.mark.parametrize("group_size, act_order", [(None, False), (32, False), (32, True)])
def test_gptq_keeps_pruned_weights_at_zero(correlated_inputs, group_size: int | None, act_order: bool) -> None:
    """SparseGPT's joint mode: exact zeros stay zero, and the error is still below RTN on the pruned weight."""
    generator = torch.Generator().manual_seed(3)
    weight = torch.randn(32, 128, generator=generator)
    weight[torch.rand(weight.shape, generator=generator) < 0.5] = 0
    inputs = correlated_inputs(128, count=2048)
    grid = make_grid("int4")
    gptq = gptq_quantize(weight, inputs.T @ inputs, grid, group_size, act_order=act_order).dequantize()
    assert (gptq[weight == 0] == 0).all()
    rtn = quantize_rtn(weight, grid, group_size).dequantize()
    assert _output_error(weight, gptq, inputs) < 0.9 * _output_error(weight, rtn, inputs)


def test_awq_clip_thresholds_never_increase_the_group_error(correlated_inputs) -> None:
    weight = torch.randn(32, 64, generator=torch.Generator().manual_seed(0))
    inputs = correlated_inputs(64, count=512)
    grid, size = make_grid("int3"), 16
    thresholds = awq_clip_thresholds(weight, inputs, grid, size)
    group_max = weight.abs().reshape(32, -1, size).amax(-1)
    assert thresholds.shape == group_max.shape
    assert (thresholds <= group_max).all() and (thresholds >= 0.55 * group_max - 1e-6).all()
    assert (thresholds < group_max).any()

    def group_error(w: torch.Tensor) -> torch.Tensor:  # mean squared partial dot product error per (row, group)
        q = quantize_rtn(w, grid, size).dequantize()
        x = inputs.reshape(inputs.shape[0], -1, size)
        diff = torch.einsum("tgk,ogk->otg", x, (q - weight).reshape(32, -1, size))
        return diff.pow(2).mean(1)

    limit = thresholds.repeat_interleave(size, dim=1)
    clipped = torch.maximum(torch.minimum(weight, limit), -limit)
    assert (group_error(clipped) <= group_error(weight) * (1 + 1e-5)).all()


# ---- whole models --------------------------------------------------------------------------------------------------


def test_rtn_model_writes_round_to_nearest_weights(tiny2, adapter_of) -> None:
    original = copy.deepcopy(tiny2)
    adapter = adapter_of(tiny2)
    summary = quantize_model(adapter, QuantConfig(format="int4", group_size=32))
    assert summary["modules"] == 14 and summary["weights"] == 2 * 36864
    assert summary["bits_per_weight"] == pytest.approx(5.0)
    for index in (1, 2):
        before = linear_modules(adapter, original.model.layers[index - 1])
        for name, module in linear_modules(adapter, adapter.layer(index)).items():
            expected = quantize_rtn(before[name].weight, make_grid("int4"), 32).dequantize()
            assert torch.equal(module.weight, expected), name
    assert torch.equal(tiny2.lm_head.weight, original.lm_head.weight)  # the LM head stays in full precision


def test_selection_and_indivisible_maps(tiny2, adapter_of, caplog) -> None:
    before = _weights(tiny2)
    adapter = adapter_of(tiny2)
    summary = quantize_model(adapter, QuantConfig(group_size=32, layers=(2,), targets=("mlp.*",), exclude=("*down*",)))
    assert summary["modules"] == 2
    changed = {name for name, p in tiny2.named_parameters() if not torch.equal(p, before[name])}
    assert changed == {"model.layers.1.mlp.gate_proj.weight", "model.layers.1.mlp.up_proj.weight"}
    # group size 128 divides only down_proj's 128 inputs; the 64-input maps are skipped with a warning
    summary = quantize_model(adapter, QuantConfig(group_size=128))
    assert summary["modules"] == 2 and "skipping" in caplog.text


@pytest.mark.parametrize("format", ["int4", "int3"])
def test_calibrated_methods_reduce_error_on_calibration_data(tiny_factory, adapter_of, calib, format) -> None:
    """GPTQ and AWQ minimize the output error on the calibration inputs, so they beat RTN there (GPTQ by about
    30%, AWQ by 5-15%, on every seed tried). Held-out inputs of a random tiny model are too few to rank them."""
    reference = tiny_factory(num_layers=2)(calib, output_hidden_states=True)
    errors = {}
    for method in ("rtn", "gptq", "awq"):
        model = tiny_factory(num_layers=2)
        config = QuantConfig(format=format, method=method, group_size=32)
        quantize_model(adapter_of(model), config, token_batches(calib))
        output = model(calib, output_hidden_states=True)
        errors[method] = [
            (output.hidden_states[1] - reference.hidden_states[1]).pow(2).mean().item(),
            (output.logits - reference.logits).pow(2).mean().item(),
        ]
    for level in range(2):
        assert errors["gptq"][level] < 0.8 * errors["rtn"][level]
        assert errors["awq"][level] < errors["rtn"][level]


@pytest.mark.parametrize("method", ["rtn", "gptq", "awq"])
def test_quantization_keeps_the_pruning_mask(tiny2, adapter_of, calib, method: str) -> None:
    from economize.techniques.prune import PruneConfig, prune_model

    adapter = adapter_of(tiny2)
    prune_model(adapter, PruneConfig(method="wanda", sparsity=0.5), token_batches(calib))
    zeros = {name: p == 0 for name, p in tiny2.named_parameters() if p.dim() == 2 and "layers" in name}
    quantize_model(adapter, QuantConfig(format="int4", method=method, group_size=32), token_batches(calib))
    for name, p in tiny2.named_parameters():
        if name in zeros:
            assert (p[zeros[name]] == 0).all(), name


def test_awq_scales_leave_the_float_model_unchanged(tiny2, adapter_of, calib, held_out) -> None:
    """Scale folding (``1/s`` into the norm or previous linear map, ``s`` into the next maps) is exact in float."""
    reference = tiny2(held_out).logits
    adapter = adapter_of(tiny2)
    before = _weights(tiny2)
    grid = make_grid("int4")
    scaled = []
    for step in LayerwiseCalibration(adapter, token_batches(calib[:4]), propagate="original"):
        result = awq_scale_layer(
            step, awq_scale_groups(adapter), linear_modules(adapter, step.layer), grid, 32, n_grid=5, clip=False
        )
        scaled.append(result["scaled"])
        assert result["clipped"] == []
    # v_proj -> o_proj is skipped: with grouped-query attention v_proj's outputs are not o_proj's inputs one to one
    assert scaled[0] == ["input_layernorm", "post_attention_layernorm", "mlp.up_proj"]
    norm = tiny2.model.layers[0].input_layernorm.weight
    assert not torch.equal(norm, before["model.layers.0.input_layernorm.weight"])
    torch.testing.assert_close(tiny2(held_out).logits, reference, atol=1e-4, rtol=1e-4)


def test_awq_derives_scale_groups_when_none_are_registered() -> None:
    from types import SimpleNamespace

    assert awq_scale_groups(SimpleNamespace(config=SimpleNamespace(model_type="unknown_arch"))) is None


def _derived(adapter, calib) -> list[tuple[ScaleGroup, ...]]:
    return [derive_scale_groups(step) for step in LayerwiseCalibration(adapter, token_batches(calib[:2]))]


def test_derived_scale_groups_match_the_registered_llama_groups(tiny2, adapter_of, calib) -> None:
    registered = set(awq_scale_groups(adapter_of(tiny2)))
    for groups in _derived(adapter_of(tiny2), calib):
        # the registered v_proj -> o_proj group is skipped at run time under grouped-query attention; the probe
        # never proposes it
        assert {(g.prev, g.linears, g.inspect, g.layer_kwargs) for g in groups} == {
            (g.prev, g.linears, g.inspect, g.layer_kwargs) for g in registered if g.prev != "self_attn.v_proj"
        }


def test_derivation_leaves_the_weights_bit_for_bit(tiny2, adapter_of, calib) -> None:
    before = _weights(tiny2)
    _derived(adapter_of(tiny2), calib)
    assert all(torch.equal(p, before[name]) for name, p in tiny2.named_parameters())


@pytest.fixture
def gemma4():
    from helpers import family_skip_reason, make_family_model

    if reason := family_skip_reason("gemma4_text"):
        pytest.skip(reason)
    pytest.importorskip("transformers.models.gemma4")
    return make_family_model("gemma4_text").eval()


def test_gemma4_scale_groups_follow_each_layer_type(gemma4, adapter_of) -> None:
    """Sliding layers scale q, k and v; global layers share k as v (``attention_k_eq_v``); KV-shared layers own
    only q. The value norm blocks ``v_proj -> o_proj`` and the MLP reads the pre-feedforward norm."""
    adapter = adapter_of(gemma4)
    assert awq_scale_groups(adapter) is None
    ids = torch.randint(0, adapter.vocab_size, (2, 16), generator=torch.Generator().manual_seed(0))
    attn = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
    layers = []
    for groups in _derived(adapter, ids):
        by_prev = {g.prev: g for g in groups}
        assert set(by_prev) == {"input_layernorm", "pre_feedforward_layernorm", "mlp.up_proj"}
        assert by_prev["pre_feedforward_layernorm"].linears == ("mlp.gate_proj", "mlp.up_proj")
        assert by_prev["pre_feedforward_layernorm"].inspect == "mlp"
        assert by_prev["mlp.up_proj"].linears == ("mlp.down_proj",)
        layers.append(by_prev["input_layernorm"])
    config = gemma4.config
    shared_from = config.num_hidden_layers - config.num_kv_shared_layers
    for i, group in enumerate(layers):
        if i >= shared_from:
            assert group.linears == attn[:1] and group.inspect is None
        elif config.layer_types[i] == "full_attention":
            assert group.linears == attn[:2] and group.inspect == "self_attn" and group.layer_kwargs
        else:
            assert group.linears == attn and group.inspect == "self_attn" and group.layer_kwargs


def test_awq_quantizes_gemma4(gemma4, adapter_of, nll) -> None:
    adapter = adapter_of(gemma4)
    ids = torch.randint(0, adapter.vocab_size, (4, 24), generator=torch.Generator().manual_seed(0))
    held_out = torch.randint(0, adapter.vocab_size, (2, 24), generator=torch.Generator().manual_seed(1))
    reference = gemma4(held_out).logits
    float_copy = copy.deepcopy(gemma4)
    for step in LayerwiseCalibration(adapter_of(float_copy), token_batches(ids), propagate="original"):
        result = awq_scale_layer(step, None, linear_modules(adapter_of(float_copy), step.layer), make_grid("int4"),
                                 None, n_grid=5, clip=False)
        assert "input_layernorm" in result["scaled"]
    torch.testing.assert_close(float_copy(held_out).logits, reference, atol=1e-4, rtol=1e-4)
    dense = nll(gemma4, held_out)
    quantize_model(adapter, QuantConfig(method="awq", format="int4", group_size=None), token_batches(ids))
    logits = gemma4(held_out).logits
    assert torch.isfinite(logits).all() and not torch.equal(logits, reference)
    assert abs(nll(gemma4, held_out) - dense) < 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"format": "int8", "group_size": None, "symmetric": True},
        {"format": "fp8", "group_size": None},
        {"method": "gptq", "act_order": True, "group_size": 32},
        {"method": "gptq", "true_sequential": False, "group_size": 32},
        {"method": "awq", "format": "int3", "group_size": 32},
    ],
)
def test_quantized_model_end_to_end(tiny2, adapter_of, calib, held_out, nll, kwargs) -> None:
    dense_logits, dense_nll = tiny2(held_out).logits, nll(tiny2, held_out)
    quantize_model(adapter_of(tiny2), QuantConfig(**kwargs), token_batches(calib))
    logits, value = tiny2(held_out).logits, nll(tiny2, held_out)
    assert torch.isfinite(logits).all() and not torch.equal(logits, dense_logits)
    bound = 0.05 if kwargs.get("format") in ("int8", "fp8") else 1.0
    assert abs(value - dense_nll) < bound


def test_calibrated_methods_need_data(tiny2, adapter_of) -> None:
    with pytest.raises(ConfigError, match="calibration"):
        quantize_model(adapter_of(tiny2), QuantConfig(method="gptq"))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"format": "int5"}, "format"),
        ({"method": "squeezellm"}, "method"),
        ({"backend": "bitsandbytes"}, "backend"),
        ({"group_size": 0}, "group_size"),
        ({"act_order": True}, "act_order"),
        ({"percdamp": 1.5}, "percdamp"),
        ({"layers": (0,)}, "1-indexed"),
        ({"fp8_activations": True}, "fp8_activations"),
        ({"backend": "torchao", "format": "int3"}, "torchao"),
        ({"backend": "torchao", "format": "int4", "group_size": 48}, "tinygemm"),
        ({"backend": "torchao", "format": "int8", "group_size": None}, "symmetric"),
        ({"backend": "torchao", "format": "fp8", "fp8_activations": True, "group_size": 32}, "row-wise"),
    ],
)
def test_quant_config_validation(kwargs: dict, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        QuantConfig(**kwargs)


def test_quant_config_validate_checks_layers(tiny2, adapter_of) -> None:
    with pytest.raises(ConfigError):
        QuantConfig(layers=(3,)).validate(adapter_of(tiny2))


# ---- torchao backend -----------------------------------------------------------------------------------------------


@pytest.mark.skipif(HAS_TORCHAO, reason="torchao is installed")
def test_torchao_backend_reports_missing_package(tiny2, adapter_of) -> None:
    with pytest.raises(ConfigError, match="pip install"):
        config = QuantConfig(format="int8", symmetric=True, group_size=None, backend="torchao")
        quantize_model(adapter_of(tiny2), config)


@pytest.mark.parametrize(
    ("kwargs", "device", "slow"),
    [
        ({"format": "int8", "symmetric": True, "group_size": None}, "cuda", True),
        ({"format": "int8", "group_size": 32}, "cuda", True),
        ({"format": "fp8", "group_size": None}, "cuda", True),
        ({"format": "fp8", "group_size": None, "fp8_activations": True}, "cuda", False),
        ({"format": "int4", "group_size": 128}, "cuda", False),
        ({"format": "int8", "group_size": 32}, "cpu", False),
    ],
)
def test_torchao_warns_when_it_dequantizes_on_every_eager_call(kwargs, device, slow, caplog) -> None:
    """B8: torchao int8 and weight-only fp8 decoded at 0.50x / 0.14x of the dense model without torch.compile."""
    from types import SimpleNamespace

    from economize.techniques.quant.backends import warn_eager_dequant

    modules = {"up": SimpleNamespace(weight=SimpleNamespace(device=torch.device(device)))}
    config = QuantConfig(backend="torchao", **kwargs)
    with caplog.at_level("WARNING", logger="economize.techniques.quant.backends"):
        assert warn_eager_dequant(modules, config) is slow  # type: ignore[arg-type]
    assert ("marlin" in caplog.text) is slow


@pytest.mark.parametrize(
    "kwargs",
    [
        {"format": "int8", "symmetric": True, "group_size": None},
        {"format": "int8", "group_size": 32},
        {"format": "fp8", "group_size": None},
    ],
)
def test_torchao_matches_fake_quantization(tiny_factory, adapter_of, held_out, kwargs) -> None:
    pytest.importorskip("torchao")
    outputs = []
    for backend in ("fake", "torchao"):
        model = tiny_factory(num_layers=2)
        quantize_model(adapter_of(model), QuantConfig(backend=backend, **kwargs))
        outputs.append(model(held_out).logits)
    assert type(model.model.layers[0].mlp.up_proj.weight).__module__.startswith("torchao")
    torch.testing.assert_close(outputs[1], outputs[0], atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9), reason="fp8 matmuls need sm89+"
)
def test_torchao_fp8_activations_run_scaled_matmuls(tiny_factory, adapter_of, held_out) -> None:
    pytest.importorskip("torchao")
    outputs = []
    for fp8_activations in (False, True):
        model = tiny_factory(num_layers=2).to("cuda", torch.bfloat16)
        config = QuantConfig(format="fp8", group_size=None, backend="torchao", fp8_activations=fp8_activations)
        quantize_model(adapter_of(model), config)
        outputs.append(model(held_out.cuda()).logits.float())
    # W8A8 adds the activation rounding to the weight-only result
    assert torch.nn.functional.cosine_similarity(outputs[1].flatten(), outputs[0].flatten(), dim=0) > 0.99


def test_torchao_int4_needs_bf16_on_sm80(tiny2, adapter_of) -> None:
    pytest.importorskip("torchao")
    from economize.errors import UnsupportedModelError

    if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0):
        pytest.skip("the CPU check is moot on an sm80+ GPU")
    with pytest.raises(UnsupportedModelError, match="tinygemm"):
        quantize_model(adapter_of(tiny2), QuantConfig(format="int4", group_size=32, backend="torchao"))


# ---- drafter-only quantization -------------------------------------------------------------------------------------


class _StubDrafter(Technique):
    """A drafter with its own predictor, a LoRA factor and a linear map shared with the base model."""

    name = "stub"
    capabilities = Capability.DRAFTER
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True, trainable=("predictor",))

    def __init__(self, shared: nn.Linear) -> None:
        nn.Module.__init__(self)  # no config or adapter to validate
        self.config = None
        self.predictor = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
        self.lora_A = nn.Linear(64, 8, bias=False)
        self.shared = shared


@pytest.fixture
def drafter(tiny2) -> _StubDrafter:
    torch.manual_seed(0)
    return _StubDrafter(tiny2.model.layers[0].self_attn.q_proj)


def test_draft_modules_skip_shared_and_lora_maps(tiny2, drafter) -> None:
    config = DraftQuantConfig(group_size=32)
    assert set(draft_modules(drafter, config, tiny2.parameters())) == {"predictor.0", "predictor.2"}
    assert "shared" in draft_modules(drafter, config)  # without the base parameters it is a candidate
    drafter.draft_modules = lambda: {"head": drafter.predictor[2]}  # a drafter may choose its modules itself
    assert set(draft_modules(drafter, config, tiny2.parameters())) == {"head"}


def test_quantize_draft_and_restore_exactly(tiny2, drafter) -> None:
    config = DraftQuantConfig(format="int4", group_size=32, symmetric=False)
    before = _weights(drafter)
    trainable = {name: p.requires_grad for name, p in drafter.named_parameters()}
    assert trainable["predictor.0.weight"] and not trainable["shared.weight"]  # the tiny base model is frozen
    modules = draft_modules(drafter, config, tiny2.parameters())
    snapshot = quantize_draft(modules, config)
    assert snapshot.summary["modules"] == 2 and snapshot.summary["bits_per_weight"] == pytest.approx(5.0)
    for name, module in modules.items():
        expected = quantize_rtn(before[f"{name}.weight"], config.grid, 32).dequantize()
        assert torch.equal(module.weight, expected) and not module.weight.requires_grad
    assert torch.equal(drafter.shared.weight, before["shared.weight"])
    assert torch.equal(drafter.lora_A.weight, before["lora_A.weight"])
    restore_draft(snapshot)
    for name, p in drafter.named_parameters():
        assert torch.equal(p, before[name]) and p.requires_grad == trainable[name], name


def test_quantized_copy_leaves_the_shared_module_untouched(tiny2) -> None:
    layer = tiny2.model.layers[1]
    before = {name: p.clone() for name, p in layer.named_parameters()}
    clone, summary = quantized_copy(layer, DraftQuantConfig(group_size=32))
    assert summary["modules"] == 7
    assert all(torch.equal(p, before[name]) for name, p in layer.named_parameters())
    assert not torch.equal(clone.mlp.up_proj.weight, layer.mlp.up_proj.weight)


def test_draft_quant_config_validation() -> None:
    with pytest.raises(ConfigError):
        DraftQuantConfig(format="int5")
    with pytest.raises(ConfigError):
        DraftQuantConfig(backend="torchao", format="int3")


# ---- registered techniques -----------------------------------------------------------------------------------------


@requires_weights_contract
def test_quant_technique(tiny2, adapter_of, calib, attach_context) -> None:
    from economize.techniques.lowrank import LowRankTechnique
    from economize.techniques.prune import PruneTechnique
    from economize.techniques.quant import QuantTechnique

    adapter = adapter_of(tiny2)
    assert PruneTechnique.order < LowRankTechnique.order < QuantTechnique.order == 30
    gptq = QuantTechnique(QuantConfig(method="gptq", group_size=32), adapter)
    assert gptq.capabilities == Capability.WEIGHTS and not gptq.lossless
    with pytest.raises(ConfigError, match="calibration"):
        gptq.attach(tiny2, adapter, attach_context(None))
    gptq.attach(tiny2, adapter, attach_context(calib))
    assert gptq.summary["modules"] == 14 and gptq.summary["method"] == "gptq"
    with pytest.raises(EconomizeError):
        gptq.detach()
    rtn = QuantTechnique(QuantConfig(format="int8", symmetric=True, group_size=None), adapter)
    rtn.attach(tiny2, adapter, None)  # round to nearest needs no data
    per_layer = {64: 4096 * 2 + 2048 * 2 + 8192 * 2, 128: 8192}  # weights by input size (down_proj reads 128)
    expected = 8 + sum(16 / features * n for features, n in per_layer.items()) / 36864
    assert rtn.summary["bits_per_weight"] == pytest.approx(expected)


@requires_weights_contract
def test_draft_quant_technique(tiny2, adapter_of, drafter, attach_context, held_out) -> None:
    from economize.techniques.quant import DraftQuantTechnique

    adapter = adapter_of(tiny2)
    technique = DraftQuantTechnique(DraftQuantConfig(group_size=32), adapter)
    assert technique.lossless and technique.order > 30 and technique.capabilities == Capability.NONE
    with pytest.raises(IncompatibleTechniquesError):
        technique.check_compatible([])
    technique.check_compatible([drafter])
    base_logits, before = tiny2(held_out).logits, _weights(drafter)
    technique.attach(tiny2, adapter, attach_context(techniques=(drafter,)))
    assert technique.summary["modules"] == 2
    assert not torch.equal(drafter.predictor[0].weight, before["predictor.0.weight"])
    assert torch.equal(tiny2(held_out).logits, base_logits)  # the verified base model is untouched
    technique.detach()
    assert all(torch.equal(p, before[name]) for name, p in drafter.named_parameters())


@requires_weights_contract
def test_draft_quant_sees_through_wrapping_drafters(tiny2) -> None:
    import economize
    from economize.techniques.eagle3 import Eagle3Config
    from economize.techniques.frspec import FRSpecConfig
    from economize.techniques.hybrid import HybridConfig

    def quantized(drafter: object) -> int:
        model = economize.apply(tiny2, drafter, DraftQuantConfig(group_size=32))
        modules = model.techniques["draft_quant"].summary["modules"]
        model.detach()
        return modules

    plain = quantized(Eagle3Config())
    assert plain > 0
    assert quantized(FRSpecConfig(vocab_size=32, drafter=Eagle3Config(), token_ids=list(range(32)))) == plain
    assert quantized(HybridConfig(fallback=Eagle3Config())) == plain
