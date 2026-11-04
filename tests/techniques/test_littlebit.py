"""LittleBit binary low-rank compression: packing, bit budget, Dual-SVID, the Triton kernel, QAT and the technique."""

from __future__ import annotations

import itertools

import pytest
import torch
from torch import nn

from economize.core.technique import Capability
from economize.errors import ConfigError, EconomizeError
from economize.techniques.littlebit import (
    LITTLEBIT_BACKENDS,
    LITTLEBIT_METHODS,
    LatentLittleBitLinear,
    LittleBitConfig,
    LittleBitLinear,
    LittleBitTechnique,
    bits_of,
    littlebit_init,
    littlebit_model,
    load_factors,
    pack_signs,
    packed_width,
    rank_for_bpw,
    reference_forward,
    save_factors,
    smooth_sign,
    unpack_signs,
)
from economize.techniques.littlebit.kernel import KernelConfig, littlebit_matmul, triton_available
from economize.techniques.prune import token_batches

requires_triton = pytest.mark.skipif(not triton_available(), reason="needs Triton and a CUDA device")

# rank_multiple 32 at 1 bit would leave every 64-wide map of the tiny model dense
TINY = dict(bpw=4.0, rank_multiple=1)


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


# ---- packing and bit budget ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 31, 32, 33, 96, 100])
def test_pack_round_trip(n: int) -> None:
    x = torch.randn(3, 5, n, generator=torch.Generator().manual_seed(n))
    x[0, 0, 0] = 0.0  # sign(0) = +1
    words = pack_signs(x)
    assert words.dtype == torch.int32 and words.shape == (3, 5, packed_width(n))
    signs = unpack_signs(words, n)
    assert torch.equal(signs, torch.where(x >= 0, 1.0, -1.0))
    assert signs[0, 0, 0] == 1


def test_pack_uses_the_top_bit() -> None:
    assert pack_signs(torch.ones(32)).item() == -1  # all 32 bits set
    assert pack_signs(torch.full((32,), -1.0)).item() == 0


def test_rank_for_bpw_matches_the_paper() -> None:
    """Appendix D: 4096 x 4096 at 0.55 bits and 11008 x 4096 at 0.1 bits with residual give ranks 546 and 133."""
    assert rank_for_bpw(4096, 4096, 0.55) == 546
    assert rank_for_bpw(11008, 4096, 0.1) == 133
    for out_features, in_features, bpw in ((4096, 4096, 1.0), (21504, 5376, 0.55), (64, 128, 4.0)):
        rank = rank_for_bpw(out_features, in_features, bpw)
        assert bits_of(out_features, in_features, rank) <= bpw * out_features * in_features
        assert bits_of(out_features, in_features, rank + 1) > bpw * out_features * in_features
    assert rank_for_bpw(21504, 5376, 1.0, multiple=32) == 2112
    assert rank_for_bpw(64, 64, 1.0) == 0  # the scales alone exceed the budget
    with pytest.raises(ValueError):
        rank_for_bpw(64, 64, 0.0)


# ---- Dual-SVID -----------------------------------------------------------------------------------------------------


def test_dual_svid_error_decreases_with_rank() -> None:
    weight = torch.randn(64, 96, generator=torch.Generator().manual_seed(0))
    errors = [_relative(littlebit_init(weight, rank).dense_weight(), weight) for rank in (4, 8, 16, 32, 64)]
    assert all(a > b for a, b in itertools.pairwise(errors))
    assert errors[-1] < 0.65


def test_residual_path_reduces_the_error() -> None:
    weight = torch.randn(64, 96, generator=torch.Generator().manual_seed(1))
    for rank in (8, 32):
        single = littlebit_init(weight, rank, residual=False)
        double = littlebit_init(weight, rank)
        assert single.paths == 1 and double.paths == 2 and double.rank == rank
        assert _relative(double.dense_weight(), weight) < _relative(single.dense_weight(), weight)


def test_dual_svid_scales_are_nonnegative_and_signs_follow_the_svd() -> None:
    weight = torch.randn(48, 40, generator=torch.Generator().manual_seed(2))
    factors = littlebit_init(weight, 8)
    for scale in (factors.h, factors.g, factors.l):
        assert (scale >= 0).all()
    left, sigma, _ = torch.linalg.svd(weight)
    assert torch.equal(factors.u[0] >= 0, left[:, :8] * sigma[:8].sqrt() >= 0)


# ---- module and backends -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("residual", [False, True])
def test_module_matches_the_dense_reconstruction(bias: bool, residual: bool) -> None:
    generator = torch.Generator().manual_seed(3)
    weight = torch.randn(40, 72, generator=generator)
    b = torch.randn(40, generator=generator) if bias else None
    factors = littlebit_init(weight, 20, residual=residual)
    module = LittleBitLinear.from_factors(factors, b, weight)
    x = torch.randn(2, 5, 72, generator=generator)
    dense = x @ factors.dense_weight().T + (b if bias else 0)
    torch.testing.assert_close(module(x), dense, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(module.dense_weight(), factors.dense_weight())
    assert module.stored_bits() == bits_of(40, 72, 20, factors.paths)
    assert module.u_bits.shape == (factors.paths, 40, 1) and module.v_bits.shape == (factors.paths, 20, 3)


def test_backend_registry() -> None:
    assert {"torch", "triton"} <= set(LITTLEBIT_BACKENDS.names())
    module = LittleBitLinear(8, 4, 2)
    module.set_backend("triton")
    assert module.backend == "triton"
    x = torch.randn(3, 8)
    torch.testing.assert_close(module(x), reference_forward(module, x))  # CPU: falls back to the reference


@requires_triton
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [1, 2, 7, 16, 33, 64])
@pytest.mark.parametrize("paths", [1, 2])
def test_kernel_matches_the_reference(dtype: torch.dtype, rows: int, paths: int) -> None:
    generator = torch.Generator().manual_seed(rows)
    out_features, in_features, rank = 200, 352, 72  # neither a multiple of the blocks nor of 32
    weight = torch.randn(out_features, in_features, generator=generator) / in_features**0.5
    factors = littlebit_init(weight, rank, residual=paths == 2)
    like = weight.to("cuda", dtype)
    module = LittleBitLinear.from_factors(factors, torch.randn(out_features, generator=generator), like, "triton")
    x = torch.randn(rows, in_features, generator=generator).to("cuda", dtype)
    expected = reference_forward(module, x.float())  # float32 reference of the same stored factors
    for config in (None, KernelConfig(split_k=1), KernelConfig(split_k=4, down_block_w=2)):
        out = littlebit_matmul(x, module.v_bits, module.u_bits, module.g, module.l, module.h, module.bias, config)
        assert out.dtype == dtype and out.shape == (rows, out_features)
        assert _relative(out, expected) < 1e-2
    torch.testing.assert_close(module(x).float(), reference_forward(module, x).float(), atol=3e-2, rtol=3e-2)


@requires_triton
def test_kernel_at_a_31b_mlp_shape() -> None:
    """Gemma-4-31B's up projection (5376 -> 21504) at 1 bit (rank 2112), batch 1 and 16."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    rank, paths = 2112, 2
    module = LittleBitLinear(5376, 21504, rank, paths, False, "triton", torch.device("cuda"), torch.bfloat16)
    module.v_bits.copy_(torch.randint(-(2**31), 2**31 - 1, module.v_bits.shape, generator=generator, device="cuda"))
    module.u_bits.copy_(torch.randint(-(2**31), 2**31 - 1, module.u_bits.shape, generator=generator, device="cuda"))
    for name in ("g", "l", "h"):
        getattr(module, name).uniform_(0.5, 1.5, generator=generator)
    for rows in (1, 16):
        x = torch.randn(rows, 5376, generator=generator, device="cuda", dtype=torch.bfloat16)
        assert _relative(module(x), reference_forward(module, x.float())) < 1e-2


# ---- QAT -----------------------------------------------------------------------------------------------------------


def test_smooth_sign_gradient() -> None:
    with torch.enable_grad():
        x = torch.tensor([-0.5, 0.0, 0.001, 0.5], requires_grad=True)
        y = smooth_sign(x, 100.0)
        y.sum().backward()
    assert torch.equal(y.detach(), torch.tensor([-1.0, 1.0, 1.0, 1.0]))
    torch.testing.assert_close(x.grad, 100 * (1 - torch.tanh(100 * x.detach()).square()))


def test_latent_module_packs_to_the_same_map() -> None:
    generator = torch.Generator().manual_seed(4)
    weight = torch.randn(24, 40, generator=generator)
    latent = LatentLittleBitLinear(littlebit_init(weight, 12), torch.randn(24, generator=generator))
    x = torch.randn(6, 40, generator=generator)
    torch.testing.assert_close(latent.pack(weight)(x), latent(x), atol=1e-4, rtol=1e-4)


# ---- whole-model compression ---------------------------------------------------------------------------------------


def test_littlebit_model_summary_and_structure(tiny2, adapter_of) -> None:
    adapter = adapter_of(tiny2)
    summary = littlebit_model(adapter, LittleBitConfig(**TINY))
    assert summary["modules"] == 14 and summary["kept_dense"] == 0 and summary["method"] == "dual_svid"
    assert 3.0 < summary["bpw"] <= 4.0
    q_proj = tiny2.model.layers[0].self_attn.q_proj
    assert isinstance(q_proj, LittleBitLinear) and q_proj.rank == summary["ranks"]["1.self_attn.q_proj"] == 42
    assert q_proj.bias is not None  # Qwen2's q/k/v biases are kept
    assert isinstance(tiny2.lm_head, nn.Linear) and isinstance(tiny2.model.embed_tokens, nn.Embedding)


def test_maps_without_a_budget_stay_dense(tiny2, adapter_of) -> None:
    summary = littlebit_model(adapter_of(tiny2), LittleBitConfig(bpw=1.0))
    assert (summary["modules"], summary["kept_dense"]) == (0, 14)


def test_selection(tiny2, adapter_of) -> None:
    summary = littlebit_model(adapter_of(tiny2), LittleBitConfig(**TINY, layers=(2,), targets=("mlp.*",)))
    assert sorted(summary["ranks"]) == ["2.mlp.down_proj", "2.mlp.gate_proj", "2.mlp.up_proj"]
    assert not any(isinstance(m, LittleBitLinear) for m in tiny2.model.layers[0].modules())


def test_blockwise_training_reduces_the_layer_error(tiny_factory, adapter_of, calib, held_out) -> None:
    reference = tiny_factory(num_layers=2)
    errors = {}
    for method in ("dual_svid", "blockwise"):
        model = tiny_factory(num_layers=2)
        config = LittleBitConfig(**TINY, method=method, epochs=8, lr=1e-3)
        summary = littlebit_model(adapter_of(model), config, token_batches(calib))
        errors[method] = [(model(ids).logits - reference(ids).logits).pow(2).mean().item() for ids in (calib, held_out)]
        if method == "blockwise":
            assert set(summary["losses"]) == {1, 2}
            assert all(stats["final"] < stats["init"] for stats in summary["losses"].values())
    assert errors["blockwise"][0] < errors["dual_svid"][0]
    assert errors["blockwise"][1] < errors["dual_svid"][1]


def test_blockwise_restores_the_layer_on_failure(tiny2, adapter_of, calib, monkeypatch) -> None:
    import economize.techniques.littlebit.methods as methods

    def fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(methods, "train_block", fail)
    before = {k: v.clone() for k, v in tiny2.state_dict().items()}
    with pytest.raises(RuntimeError, match="boom"):
        littlebit_model(adapter_of(tiny2), LittleBitConfig(**TINY, method="blockwise"), token_batches(calib))
    assert not any(isinstance(m, (LittleBitLinear, LatentLittleBitLinear)) for m in tiny2.modules())
    assert all(torch.equal(before[k], v) for k, v in tiny2.state_dict().items())


def test_blockwise_needs_data(tiny2, adapter_of) -> None:
    with pytest.raises(ConfigError, match="calibration"):
        littlebit_model(adapter_of(tiny2), LittleBitConfig(**TINY, method="blockwise"))


def test_save_load_round_trip(tiny_factory, adapter_of, held_out, tmp_path) -> None:
    model = tiny_factory(num_layers=2)
    adapter = adapter_of(model)
    littlebit_model(adapter, LittleBitConfig(**TINY))
    path = tmp_path / "factors.safetensors"
    assert save_factors(adapter, path) == 14
    assert len(load_factors(path)) == 14
    expected = model(held_out).logits
    fresh = tiny_factory(num_layers=2)
    summary = littlebit_model(adapter_of(fresh), LittleBitConfig(factors=str(path), backend="torch"))
    assert summary["modules"] == 14 and summary["method"] == "factors"
    torch.testing.assert_close(fresh(held_out).logits, expected)


def test_load_rejects_a_mismatched_model(tiny_factory, adapter_of, tmp_path) -> None:
    model = tiny_factory(num_layers=2)
    littlebit_model(adapter_of(model), LittleBitConfig(**TINY))
    path = tmp_path / "factors.safetensors"
    save_factors(adapter_of(model), path)
    other = tiny_factory(num_layers=2)
    other.model.layers[0].mlp.up_proj = nn.Linear(64, 96)
    with pytest.raises(ConfigError, match="does not match"):
        littlebit_model(adapter_of(other), LittleBitConfig(factors=str(path)))
    assert not any(isinstance(m, LittleBitLinear) for m in other.modules())  # rolled back


@requires_triton
def test_tiny_model_triton_matches_the_reference_backend(tiny_factory, adapter_of, held_out) -> None:
    """The bf16 Triton model is as close to the float32 reference as the bf16 reference backend is."""
    model = tiny_factory(num_layers=2)
    littlebit_model(adapter_of(model), LittleBitConfig(**TINY, backend="torch"))
    ids = held_out.cuda()
    model = model.cuda()
    exact = model(ids).logits
    model = model.to(torch.bfloat16)
    rounded = model(ids).logits.float()
    for module in model.modules():
        if isinstance(module, LittleBitLinear):
            module.set_backend("triton")
    fast = model(ids).logits.float()
    assert _relative(fast, exact) < 1.5 * _relative(rounded, exact) + 1e-3
    assert (fast.argmax(-1) == exact.argmax(-1)).float().mean() > 0.9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "gptq"},
        {"backend": "cuda"},
        {"bpw": 0.0},
        {"bpw": 16.0},
        {"rank_multiple": 0},
        {"epochs": 0},
        {"lr": 0.0},
        {"smooth_k": -1.0},
        {"layers": (0,)},
        {"targets": ()},
    ],
)
def test_littlebit_config_validation(kwargs: dict) -> None:
    with pytest.raises(ConfigError):
        LittleBitConfig(**kwargs)


def test_method_registry() -> None:
    assert set(LITTLEBIT_METHODS.names()) >= {"dual_svid", "blockwise"}
    assert not LITTLEBIT_METHODS.get("dual_svid").needs_calibration
    assert LITTLEBIT_METHODS.get("blockwise").needs_calibration


# ---- registered technique ------------------------------------------------------------------------------------------


def test_littlebit_technique(tiny2, tiny_factory, adapter_of, calib, attach_context, tmp_path) -> None:
    adapter = adapter_of(tiny2)
    technique = LittleBitTechnique(LittleBitConfig(**TINY, method="blockwise", epochs=1), adapter)
    assert technique.capabilities == Capability.WEIGHTS and technique.order == 25 and not technique.lossless
    with pytest.raises(ConfigError, match="calibration"):
        technique.attach(tiny2, adapter, attach_context(None))
    technique.attach(tiny2, adapter, attach_context(calib))
    assert technique.summary["modules"] == 14
    with pytest.raises(EconomizeError):
        technique.detach()
    path = tmp_path / "factors.safetensors"
    save_factors(adapter, path)
    model = tiny_factory(num_layers=2)
    other = adapter_of(model)
    loaded = LittleBitTechnique(LittleBitConfig(method="blockwise", factors=str(path)), other)
    assert not loaded.needs_calibration()
    loaded.attach(model, other, None)
    assert loaded.summary["modules"] == 14
