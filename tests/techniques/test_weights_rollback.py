"""All-or-nothing weight transforms: :class:`WeightTransaction` and its use in prune, low rank and draft quant.

* The transaction swaps replaced modules back (newest first) and then restores the parameters, on ``rollback`` or
  when the ``with`` block raises.
* ``prune_model`` (also the 2:4 conversion), ``lowrank_model`` and ``quantize_draft`` leave the model exactly as
  it was when they fail part way.
* ``restore_draft`` gives every weight back as a plain ``nn.Parameter`` with its original ``requires_grad``.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torch import nn

from economize.techniques._weights import WeightTransaction, replace_module
from economize.techniques.quant import QuantConfig, quantize_model
from economize.techniques.quant.backends import QUANT_BACKENDS, FakeQuantBackend, QuantBackend


def _state(model: nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, bool], dict[str, type]]:
    named = dict(model.named_modules())
    return (
        {n: p.detach().clone() for n, p in model.named_parameters()},
        {n: p.requires_grad for n, p in model.named_parameters()},
        {n: type(m) for n, m in named.items()},
    )


def _assert_same(model: nn.Module, before: tuple) -> None:
    values, grads, types = before
    params = dict(model.named_parameters())
    assert set(params) == set(values)
    for name, param in params.items():
        assert type(param) is nn.Parameter, name
        assert torch.equal(param, values[name]), name
        assert param.requires_grad == grads[name], name
    assert {n: type(m) for n, m in model.named_modules()} == types


def test_transaction_rolls_back_swaps_and_parameters_on_error() -> None:
    root = nn.Sequential(nn.Linear(4, 4), nn.Sequential(nn.Linear(4, 4)))
    first, inner = root[0], root[1][0]
    before = _state(root)
    with pytest.raises(RuntimeError, match="boom"), WeightTransaction() as tx:
        tx.take(root)
        first.weight.data.zero_()
        tx.replace_module(root, "1.0", nn.Identity())
        tx.replace_module(root, "1.0", nn.ReLU())  # replaced twice: the original comes back
        raise RuntimeError("boom")
    assert root[0] is first and root[1][0] is inner
    _assert_same(root, before)


def test_transaction_keeps_changes_without_error() -> None:
    root = nn.Sequential(nn.Linear(4, 4))
    with WeightTransaction() as tx:
        tx.take(root)
        old = tx.replace_module(root, "0", nn.Identity())
    assert isinstance(root[0], nn.Identity) and isinstance(old, nn.Linear)
    assert replace_module(root, "0", old) is not old and root[0] is old


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_offload_replaced_parks_the_old_module_on_the_cpu_and_rollback_brings_it_back() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4)).cuda()
    before = _state(model)
    old = model[1]
    tx = WeightTransaction(offload_replaced=True)
    assert tx.replace_module(model, "1", nn.Identity()) is old
    assert old.weight.device.type == "cpu"
    assert model[0].weight.is_cuda
    tx.rollback()
    assert model[1] is old and old.weight.is_cuda
    _assert_same(model, before)


def test_offload_replaced_is_a_no_op_for_cpu_modules() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    before = _state(model)
    tx = WeightTransaction(offload_replaced=True)
    tx.replace_module(model, "1", nn.Identity())
    tx.replace_module(model, "0", nn.Identity())
    tx.rollback()
    _assert_same(model, before)


def test_prune_model_restores_every_weight_when_a_layer_fails(tiny_factory, adapter_of, monkeypatch) -> None:
    from economize.techniques.prune import PRUNE_METHODS, PruneConfig, prune_model

    model = tiny_factory(num_layers=3)
    before = _state(model)
    method = PRUNE_METHODS.get("magnitude")
    original = method.prune
    calls: list[int] = []

    def flaky(self, weight, stats, pattern, config):  # type: ignore[no-untyped-def]
        calls.append(1)
        if len(calls) == 10:  # the third map of the second layer
            raise FloatingPointError("NaN")
        return original(self, weight, stats, pattern, config)

    monkeypatch.setattr(method, "prune", flaky)
    with pytest.raises(FloatingPointError):
        prune_model(adapter_of(model), PruneConfig(method="magnitude", sparsity=0.5))
    _assert_same(model, before)


def test_prune_model_restores_dense_weights_when_the_2_4_conversion_fails(tiny2, adapter_of, monkeypatch) -> None:
    from economize.techniques.prune import PruneConfig
    from economize.techniques.prune import model as prune_model_module

    before = _state(tiny2)
    calls: list[nn.Linear] = []

    def convert(linear: nn.Linear, kernel: str = "default") -> None:
        calls.append(linear)
        linear.weight = nn.Parameter(linear.weight.detach() * 0, requires_grad=False)  # a converted weight
        if len(calls) == 9:
            raise RuntimeError("no sparse kernels")

    monkeypatch.setattr(prune_model_module, "to_semi_structured", convert)
    config = PruneConfig(method="magnitude", sparsity="2:4", semi_structured=True)
    with pytest.raises(RuntimeError, match="no sparse kernels"):
        prune_model_module.prune_model(adapter_of(tiny2), config)
    _assert_same(tiny2, before)


def test_lowrank_model_swaps_every_factor_back_when_a_factorization_fails(
    tiny_factory, adapter_of, monkeypatch
) -> None:
    from economize.techniques.lowrank import FACTORIZERS, LowRankConfig, lowrank_model

    model = tiny_factory(num_layers=3)
    before = _state(model)
    factorizer = FACTORIZERS.get("svd")
    original = factorizer.factorize
    calls: list[int] = []

    def flaky(self, weight, gram, rank, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(1)
        if len(calls) == 9:
            raise torch.linalg.LinAlgError("SVD did not converge")
        return original(self, weight, gram, rank, **kwargs)

    monkeypatch.setattr(factorizer, "factorize", flaky)
    with pytest.raises(torch.linalg.LinAlgError):
        lowrank_model(adapter_of(model), LowRankConfig(method="svd", compression=0.8))
    assert len(calls) == 9
    _assert_same(model, before)


def test_restore_draft_restores_parameter_type_and_requires_grad() -> None:
    from economize.techniques.quant import DraftQuantConfig, quantize_draft, restore_draft

    modules = {"a": nn.Linear(64, 64), "b": nn.Linear(64, 32)}
    modules["b"].weight.requires_grad_(False)
    root = nn.ModuleDict(modules)
    before = _state(root)
    snapshot = quantize_draft(modules, DraftQuantConfig(group_size=32))
    assert snapshot.summary["modules"] == 2 and not modules["a"].weight.requires_grad
    modules["a"].weight = nn.Parameter(modules["a"].weight.detach() + 1, requires_grad=False)  # e.g. a kernel swap
    restore_draft(snapshot)
    _assert_same(root, before)


def _parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _assert_restored(model: nn.Module, before: dict[str, torch.Tensor], grads: dict[str, bool]) -> None:
    params = dict(model.named_parameters())
    assert set(params) == set(before)
    for name, p in params.items():
        assert type(p) is nn.Parameter and torch.equal(p, before[name]), name
        assert p.requires_grad == grads[name], name


class _RecordingBackend(QuantBackend):
    seen: list[tuple[str, ...]] = []

    def check(self, modules: Mapping[str, nn.Linear], config: object) -> None:
        type(self).seen.append(tuple(modules))
        if any(m.weight.dtype != torch.float32 for m in modules.values()):
            raise ValueError("float32 only")

    def apply(self, module: nn.Linear, qweight: object, config: object) -> None:
        pass


@pytest.fixture
def recording_backend():  # type: ignore[no-untyped-def]
    _RecordingBackend.seen = []
    QUANT_BACKENDS.register("recording_test")(_RecordingBackend)
    yield _RecordingBackend
    QUANT_BACKENDS.unregister("recording_test")


@pytest.mark.parametrize("method", ["rtn", "gptq", "awq"])
def test_quantize_model_restores_every_weight_when_a_conversion_fails(
    tiny_factory, adapter_of, calib, monkeypatch, method: str
) -> None:
    """The second layer's kernel conversion fails after the first layer was converted (and, for AWQ, after the
    norms were rescaled): every parameter, also the non-quantized ones, is back to its original value."""
    model = tiny_factory(num_layers=3)
    model.model.layers[0].mlp.up_proj.weight.requires_grad_(True)
    before, grads = _parameters(model), {n: p.requires_grad for n, p in model.named_parameters()}
    calls: list[nn.Module] = []

    def failing_apply(self, module, qweight, config):  # type: ignore[no-untyped-def]
        calls.append(module)
        module.weight = nn.Parameter(module.weight.detach() * 0, requires_grad=False)  # a converted weight
        if len(calls) == 9:
            raise RuntimeError("kernel conversion failed")

    monkeypatch.setattr(FakeQuantBackend, "apply", failing_apply)
    batches = list(calib[:2].split(1))
    with pytest.raises(RuntimeError, match="kernel conversion failed"):
        quantize_model(adapter_of(model), QuantConfig(method=method, group_size=32), batches)
    assert len(calls) == 9
    _assert_restored(model, before, grads)


def test_quantize_model_restores_weights_when_a_method_fails(tiny_factory, adapter_of, monkeypatch) -> None:
    from economize.techniques.quant.methods import QUANT_METHODS

    model = tiny_factory(num_layers=3)
    before, grads = _parameters(model), {n: p.requires_grad for n, p in model.named_parameters()}
    method = QUANT_METHODS.get("rtn")
    original = method.quantize_layer
    seen: list[int] = []

    def flaky(self, modules, step, grid, config, adapter):  # type: ignore[no-untyped-def]
        seen.append(len(seen))
        if len(seen) == 3:
            raise FloatingPointError("NaN in layer 3")
        return original(self, modules, step, grid, config, adapter)

    monkeypatch.setattr(method, "quantize_layer", flaky)
    with pytest.raises(FloatingPointError):
        quantize_model(adapter_of(model), QuantConfig(method="rtn", group_size=32))
    _assert_restored(model, before, grads)


def test_check_rejects_before_any_weight_changes_in_quantize_model(tiny2, adapter_of, recording_backend) -> None:
    tiny2.model.layers[1].mlp.down_proj.to(torch.float64)
    before = _parameters(tiny2)
    with pytest.raises(ValueError, match="float32 only"):
        quantize_model(adapter_of(tiny2), QuantConfig(group_size=32, backend="recording_test"))
    assert all(torch.equal(p, before[n]) for n, p in tiny2.named_parameters())
