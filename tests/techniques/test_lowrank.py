"""Low-rank weight compression: SVD-LLM truncation-aware whitening and the plain truncated-SVD baseline."""

from __future__ import annotations

import gc
import weakref

import pytest
import torch
from torch import nn

from economize.core.technique import Capability
from economize.errors import ConfigError, EconomizeError, UnknownNameError
from economize.techniques.lowrank import (
    FACTORIZERS,
    LowRankConfig,
    LowRankLinear,
    lowrank_model,
    rank_for_ratio,
    replace_module,
    svd_factors,
    svd_llm_factors,
    whitening,
)
from economize.techniques.prune import token_batches
from economize.techniques.prune.calibration import linear_modules
from economize.techniques.quant import QuantConfig, quantize_model

requires_weights_contract = pytest.mark.skipif(
    not hasattr(Capability, "WEIGHTS"), reason="needs the economize v0.2 Technique contract (Capability.WEIGHTS)"
)


def _output_error(weight: torch.Tensor, approx: torch.Tensor, inputs: torch.Tensor) -> float:
    return (inputs.double() @ (weight.double() - approx).T).pow(2).sum().item()


# ---- ranks and factorizations --------------------------------------------------------------------------------------


def test_rank_for_ratio() -> None:
    assert rank_for_ratio(64, 128, 0.2) == int(0.8 * 64 * 128 / 192) == 34
    assert rank_for_ratio(64, 128, 0.2, multiple=8) == 32
    assert rank_for_ratio(64, 64, 0.99, multiple=16) == 16  # never below one multiple
    for k_in, k_out, c in ((4096, 4096, 0.2), (11008, 4096, 0.4), (128, 64, 0.5)):
        rank = rank_for_ratio(k_out, k_in, c)
        assert rank * (k_in + k_out) <= (1 - c) * k_in * k_out < (rank + 1) * (k_in + k_out)
    for bad in (0.0, 1.0, -0.2):
        with pytest.raises(ConfigError):
            rank_for_ratio(64, 64, bad)


def test_svd_factors_attain_the_eckart_young_error() -> None:
    weight = torch.randn(48, 80, generator=torch.Generator().manual_seed(0))
    sigma = torch.linalg.svdvals(weight.double())
    for rank in (1, 10, 47):
        up, down = svd_factors(weight, rank)
        assert up.shape == (48, rank) and down.shape == (rank, 80)
        error = (weight.double() - up @ down).pow(2).sum().item()
        assert error == pytest.approx(sigma[rank:].pow(2).sum().item(), rel=1e-9)
    up, down = svd_factors(weight, 48)
    torch.testing.assert_close(up @ down, weight.double())


def test_whitening_is_a_cholesky_factor() -> None:
    inputs = torch.randn(256, 32, generator=torch.Generator().manual_seed(0)).double()
    gram = inputs.T @ inputs
    factor = whitening(gram)
    assert factor.dtype == torch.float64 and torch.equal(factor, factor.tril())
    torch.testing.assert_close(factor @ factor.T, gram)
    singular = inputs[:8].T @ inputs[:8]  # rank 8 < 32: shifted by -lambda_min + 1e-6 as in the reference
    factor = whitening(singular)
    assert torch.isfinite(factor).all()
    torch.testing.assert_close(factor @ factor.T, singular, atol=1e-4, rtol=1e-5)


def _reference_whitening(gram: torch.Tensor) -> torch.Tensor:
    """The out-of-place formula :func:`whitening` replaced."""
    gram = gram.double()
    gram = (gram + gram.T) / 2
    factor, info = torch.linalg.cholesky_ex(gram)
    if int(info) != 0:
        shift = -torch.linalg.eigvalsh(gram)[0] + 1e-6
        factor = torch.linalg.cholesky(gram + shift * torch.eye(gram.shape[0], dtype=gram.dtype))
    return factor


@pytest.mark.parametrize("count", [256, 8], ids=["definite", "singular"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_whitening_in_place_equals_the_out_of_place_formula(count: int, dtype: torch.dtype) -> None:
    """In place (blockwise symmetrization, diagonal shift) the factor is bit for bit the reference's; without
    ``overwrite`` the caller's matrix is left alone."""
    from economize.techniques.lowrank.factorize import _symmetrize_

    inputs = torch.randn(count, 37, generator=torch.Generator().manual_seed(0)).double()
    gram = (inputs.T @ inputs + 1e-3 * torch.randn(37, 37, generator=torch.Generator().manual_seed(1))).to(dtype)
    expected = _reference_whitening(gram)
    original = gram.clone()
    assert torch.equal(whitening(gram), expected) and torch.equal(gram, original)
    assert torch.equal(whitening(gram, overwrite=True), expected)
    assert torch.equal(gram, original) != (dtype == torch.float64)  # only a float64 matrix is reused
    for block in (1, 5, 36, 37, 64):
        matrix = original.double()
        _symmetrize_(matrix, block)
        assert torch.equal(matrix, (original.double() + original.double().T) / 2)


def test_svd_llm_compression_frees_each_gram_matrix(tiny2, adapter_of, calib, monkeypatch) -> None:
    """``lowrank_model`` lets the factorizer overwrite each map's Gram matrix, which is the observer's running sum
    turned into the mean in place, and frees it before the next map's factorization."""
    from economize.techniques.lowrank.factorize import SVDLLMFactorizer

    earlier: list[weakref.ref] = []
    factorize = SVDLLMFactorizer.factorize

    def spy(self, weight, gram, rank, *, overwrite_gram=False):
        assert overwrite_gram and all(ref() is None for ref in earlier)
        earlier.append(weakref.ref(gram))
        return factorize(self, weight, gram, rank, overwrite_gram=overwrite_gram)

    monkeypatch.setattr(SVDLLMFactorizer, "factorize", spy)
    summary = lowrank_model(adapter_of(tiny2), LowRankConfig(compression=0.2), token_batches(calib))
    assert len(earlier) == summary["modules"] == 14


def _spy_factorizer(monkeypatch, check) -> None:
    """Wrap ``SVDLLMFactorizer.factorize`` so that ``check(weight)`` runs before every factorization."""
    from economize.techniques.lowrank.factorize import SVDLLMFactorizer

    factorize = SVDLLMFactorizer.factorize

    def spy(self, weight, gram, rank, *, overwrite_gram=False):
        check(weight)
        return factorize(self, weight, gram, rank, overwrite_gram=overwrite_gram)

    monkeypatch.setattr(SVDLLMFactorizer, "factorize", spy)


def test_svd_llm_compression_releases_each_dense_map(tiny2, adapter_of, calib, monkeypatch) -> None:
    """Every replaced dense map has left the model before the next one is factorized, and once the compression
    succeeds nothing holds the dense maps or their weights any more (the rollback log is dropped with them)."""
    adapter = adapter_of(tiny2)
    dense = [m for i in (1, 2) for m in linear_modules(adapter, adapter.layer(i)).values()]
    by_storage = {m.weight.data_ptr(): id(m) for m in dense}
    refs = [weakref.ref(t) for m in dense for t in (m, m.weight)]
    done: list[int] = []

    def check(weight: torch.Tensor) -> None:
        live = {id(m) for m in tiny2.modules()}
        assert not live.intersection(done)
        done.append(by_storage[weight.data_ptr()])

    _spy_factorizer(monkeypatch, check)
    del dense
    summary = lowrank_model(adapter, LowRankConfig(compression=0.2), token_batches(calib))
    gc.collect()
    assert len(done) == summary["modules"] == 14
    assert all(ref() is None for ref in refs)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_svd_llm_compression_parks_each_dense_map_on_the_cpu(tiny_factory, adapter_of, calib, monkeypatch) -> None:
    """On a GPU every replaced dense map moves to the CPU before the next factorization, so device memory shrinks
    layer by layer towards the compressed size instead of growing by the factors (the b31k_lr20 / q31_lr20 OOM on
    Gemma-4-31B, 90 GiB allocated, on a code copy without ``WeightTransaction.offload_replaced``)."""
    model = tiny_factory(num_layers=4).cuda()
    adapter = adapter_of(model)
    dense = {m.weight.data_ptr(): m for i in range(1, 5) for m in linear_modules(adapter, adapter.layer(i)).values()}
    done: list[nn.Linear] = []

    def check(weight: torch.Tensor) -> None:
        assert weight.is_cuda and all(m.weight.device.type == "cpu" for m in done)
        done.append(dense[weight.data_ptr()])

    _spy_factorizer(monkeypatch, check)
    weights = sum(m.weight.numel() * m.weight.element_size() for m in dense.values())
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    summary = lowrank_model(adapter, LowRankConfig(compression=0.2), token_batches(calib.cuda()))
    torch.cuda.synchronize()
    assert len(done) == summary["modules"] == 28 and all(m.weight.device.type == "cpu" for m in done)
    assert before - torch.cuda.memory_allocated() >= 0.15 * weights  # the dense maps left, 0.8x their size came in


def test_gram_mean_in_place_equals_the_mean() -> None:
    from economize.core.calibration import GramObserver

    observer = GramObserver(nn.Linear(5, 3), dtype=torch.float64)
    observer.update(torch.randn(7, 5, generator=torch.Generator().manual_seed(0)))
    mean, running = observer.mean, observer.sum
    assert observer.mean_() is running and torch.equal(running, mean)


@pytest.mark.parametrize("rank", [4, 16, 40])
def test_svd_llm_truncation_loss_equals_dropped_singular_values(correlated_inputs, rank: int) -> None:
    """SVD-LLM's theorem: the output error of ``(W S)_k S^{-1}`` is the sum of the dropped ``sigma(W S)^2``."""
    weight = torch.randn(40, 64, generator=torch.Generator().manual_seed(1))
    inputs = correlated_inputs(64, count=1024).double()
    gram = inputs.T @ inputs
    up, down = svd_llm_factors(weight, gram, rank)
    sigma = torch.linalg.svdvals(weight.double() @ whitening(gram))
    assert _output_error(weight, up @ down, inputs) == pytest.approx(sigma[rank:].pow(2).sum().item(), rel=1e-6)


@pytest.mark.parametrize("rank", [4, 16, 32])
def test_svd_llm_output_error_below_svd(correlated_inputs, rank: int) -> None:
    weight = torch.randn(48, 64, generator=torch.Generator().manual_seed(2))
    inputs = correlated_inputs(64, count=2048).double()
    llm_up, llm_down = svd_llm_factors(weight, inputs.T @ inputs, rank)
    svd_up, svd_down = svd_factors(weight, rank)
    assert _output_error(weight, llm_up @ llm_down, inputs) < 0.8 * _output_error(weight, svd_up @ svd_down, inputs)
    # and the converse: plain SVD keeps the smaller weight-space error
    assert (weight - svd_up @ svd_down).norm() < (weight - llm_up @ llm_down).norm()


def test_factorizer_registry() -> None:
    assert not FACTORIZERS.get("svd").needs_calibration and FACTORIZERS.get("svd_llm").needs_calibration
    with pytest.raises(ConfigError, match="Gram"):
        FACTORIZERS.get("svd_llm")().factorize(torch.randn(4, 4), None, 2)
    with pytest.raises(UnknownNameError):
        FACTORIZERS.get("asvd")


# ---- the replacement module ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("bias", [True, False])
def test_low_rank_linear(bias: bool) -> None:
    generator = torch.Generator().manual_seed(0)
    dense = nn.Linear(24, 16, bias=bias)
    up, down = torch.randn(16, 5, generator=generator), torch.randn(5, 24, generator=generator)
    module = LowRankLinear.from_factors(up, down, dense.bias, dense.weight)
    assert (module.in_features, module.out_features, module.rank) == (24, 16, 5)
    assert module.down.bias is None and (module.up.bias is not None) == bias
    assert module.down.weight.dtype == dense.weight.dtype and module.up.weight.requires_grad
    x = torch.randn(3, 24, generator=generator)
    expected = x @ (up @ down).T + (dense.bias if bias else 0)
    torch.testing.assert_close(module(x), expected)
    torch.testing.assert_close(module.dense_weight(), up @ down)
    assert "rank=5" in repr(module)
    parent = nn.Sequential(nn.Sequential(dense))
    assert replace_module(parent, "0.0", module) is dense and parent[0][0] is module


# ---- whole models --------------------------------------------------------------------------------------------------


def test_lowrank_model_summary_and_structure(tiny2, adapter_of, calib) -> None:
    adapter = adapter_of(tiny2)
    shapes = {name: tuple(m.weight.shape) for name, m in linear_modules(adapter, adapter.layer(1)).items()}
    summary = lowrank_model(adapter, LowRankConfig(compression=0.2), token_batches(calib))
    assert summary["modules"] == 14 and summary["kept_dense"] == 0
    assert summary["params_before"] == 2 * 36864 and summary["params_after"] < 0.8 * summary["params_before"]
    assert 0.2 <= summary["compression"] < 0.22  # ranks round down, so slightly more is removed
    for index in (1, 2):
        layer = adapter.layer(index)
        for name, (out_features, in_features) in shapes.items():
            module = layer.get_submodule(name)
            assert isinstance(module, LowRankLinear)
            assert module.rank == summary["ranks"][f"{index}.{name}"] == rank_for_ratio(out_features, in_features, 0.2)
    assert tiny2.model.layers[0].self_attn.q_proj.up.bias is not None  # Qwen2's q/k/v biases are kept


def test_maps_that_would_not_shrink_stay_dense(tiny2, adapter_of) -> None:
    """With ``rank_multiple`` 32 the attention maps would need rank 32 (no saving) and stay dense."""
    summary = lowrank_model(adapter_of(tiny2), LowRankConfig(method="svd", compression=0.6, rank_multiple=32))
    assert (summary["modules"], summary["kept_dense"]) == (6, 8)
    assert isinstance(tiny2.model.layers[0].self_attn.q_proj, nn.Linear)
    assert isinstance(tiny2.model.layers[0].mlp.gate_proj, LowRankLinear)
    assert set(summary["ranks"].values()) == {32}


def test_selection(tiny2, adapter_of) -> None:
    summary = lowrank_model(adapter_of(tiny2), LowRankConfig(method="svd", layers=(2,), targets=("mlp.*",)))
    assert sorted(summary["ranks"]) == ["2.mlp.down_proj", "2.mlp.gate_proj", "2.mlp.up_proj"]
    assert not any(isinstance(m, LowRankLinear) for m in tiny2.model.layers[0].modules())
    assert isinstance(tiny2.model.layers[1].self_attn.q_proj, nn.Linear)


def test_svd_llm_model_error_below_svd(tiny_factory, adapter_of, calib, held_out) -> None:
    """At 20% compression SVD-LLM's logit error is about 0.67 of plain SVD's on the calibration inputs and lower on
    held-out inputs too (every seed tried)."""
    reference = tiny_factory(num_layers=2)
    errors = {}
    for method in ("svd", "svd_llm"):
        model = tiny_factory(num_layers=2)
        lowrank_model(adapter_of(model), LowRankConfig(method=method, compression=0.2), token_batches(calib))
        errors[method] = [(model(ids).logits - reference(ids).logits).pow(2).mean().item() for ids in (calib, held_out)]
    assert errors["svd_llm"][0] < 0.8 * errors["svd"][0]
    assert errors["svd_llm"][1] < errors["svd"][1]


@pytest.mark.parametrize("kwargs", [{"compression": 0.1}, {"compression": 0.3, "sequential": True}])
def test_lowrank_end_to_end(tiny2, adapter_of, calib, held_out, nll, kwargs) -> None:
    dense = nll(tiny2, held_out)
    lowrank_model(adapter_of(tiny2), LowRankConfig(**kwargs), token_batches(calib))
    value = nll(tiny2, held_out)
    assert torch.isfinite(torch.tensor(value)) and value != dense and abs(value - dense) < 3


def test_quantization_sees_both_factors(tiny2, adapter_of, held_out) -> None:
    adapter = adapter_of(tiny2)
    lowrank_model(adapter, LowRankConfig(method="svd", compression=0.2))
    before = tiny2(held_out).logits
    summary = quantize_model(adapter, QuantConfig(format="int8", symmetric=True, group_size=None))
    assert summary["modules"] == 28  # every compressed map is two plain linear maps
    assert not torch.equal(tiny2(held_out).logits, before)


def test_calibrated_factorization_needs_data(tiny2, adapter_of) -> None:
    with pytest.raises(ConfigError, match="calibration"):
        lowrank_model(adapter_of(tiny2), LowRankConfig())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "asvd"},
        {"compression": 0.0},
        {"compression": 1.0},
        {"rank_multiple": 0},
        {"layers": (0,)},
        {"targets": ()},
    ],
)
def test_lowrank_config_validation(kwargs: dict) -> None:
    with pytest.raises(ConfigError):
        LowRankConfig(**kwargs)


def test_lowrank_config_validate_checks_layers(tiny2, adapter_of) -> None:
    with pytest.raises(ConfigError):
        LowRankConfig(layers=(3,)).validate(adapter_of(tiny2))


# ---- registered technique ------------------------------------------------------------------------------------------


@requires_weights_contract
def test_lowrank_technique(tiny2, tiny_factory, adapter_of, calib, attach_context) -> None:
    from economize.techniques.lowrank import LowRankTechnique

    adapter = adapter_of(tiny2)
    technique = LowRankTechnique(LowRankConfig(compression=0.2), adapter)
    assert technique.capabilities == Capability.WEIGHTS and technique.order == 20 and not technique.lossless
    with pytest.raises(ConfigError, match="calibration"):
        technique.attach(tiny2, adapter, attach_context(None))
    technique.attach(tiny2, adapter, attach_context(calib))
    assert technique.summary["modules"] == 14
    with pytest.raises(EconomizeError):
        technique.detach()
    model = tiny_factory(num_layers=2)
    other = adapter_of(model)
    data_free = LowRankTechnique(LowRankConfig(method="svd", compression=0.2), other)
    data_free.attach(model, other, None)
    assert data_free.summary["method"] == "svd"
