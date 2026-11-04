"""The shared helpers of the one-shot weight techniques (quant, prune, low rank): one helper package, one backend
check for models and drafters, draft-module discovery, induced-sparsity reports and install hints."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torch import nn

from economize.techniques.quant import DraftQuantConfig, QuantConfig, draft_modules, quantize_model
from economize.techniques.quant.backends import QUANT_BACKENDS, QuantBackend
from economize.techniques.quant.model import quantize_modules


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


class _Wrapper(nn.Module):
    """A drafter whose linear map wraps a base weight in a new ``Parameter`` (same storage, other object)."""

    def __init__(self, base: nn.Linear) -> None:
        super().__init__()
        self.own = nn.Linear(64, 64)
        self.alias = nn.Linear(base.in_features, base.out_features, bias=False)
        self.alias.weight = nn.Parameter(base.weight.data, requires_grad=False)


def test_shared_weight_helpers_live_in_one_package() -> None:
    """The Hessian, validation, snapshot and transaction helpers have one home (``_weights``); the former
    ``prune.hessian`` / ``prune.validation`` aliases are gone and ``lowrank.replace_module`` is a re-export."""
    import importlib

    from economize.techniques import _weights
    from economize.techniques.lowrank import replace_module
    from economize.techniques.lowrank.technique import LowRankTechnique
    from economize.techniques.prune.technique import PruneTechnique
    from economize.techniques.quant import gptq
    from economize.techniques.quant.technique import QuantTechnique

    assert gptq.fix_dead_features is _weights.fix_dead_features
    assert replace_module is _weights.replace_module
    for gone in ("economize.techniques.prune.hessian", "economize.techniques.prune.validation"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(gone)
    for technique in (PruneTechnique, QuantTechnique, LowRankTechnique):
        assert issubclass(technique, _weights.WeightTransform)


def test_backends_have_one_check_for_models_and_drafters(tiny2, adapter_of, recording_backend) -> None:
    """``quantize_model`` and ``quantize_modules`` both call ``check`` with the modules they will quantize, before
    any weight changes."""
    assert QuantBackend.check is not None  # the default accepts everything
    quantize_model(adapter_of(tiny2), QuantConfig(group_size=32, layers=(2,), backend="recording_test"))
    assert len(recording_backend.seen) == 1 and len(recording_backend.seen[0]) == 7
    assert all(name.startswith("layers.1.") for name in recording_backend.seen[0])
    linears = {"a": nn.Linear(64, 64), "b": nn.Linear(64, 32).to(torch.float64)}
    before = {k: m.weight.detach().clone() for k, m in linears.items()}
    with pytest.raises(ValueError, match="float32 only"):
        quantize_modules(linears, DraftQuantConfig(group_size=32, backend="recording_test"))
    assert recording_backend.seen[1] == ("a", "b")
    assert all(torch.equal(m.weight, before[k]) for k, m in linears.items())


def test_draft_modules_skip_a_base_weight_wrapped_in_a_new_parameter(tiny2) -> None:
    base = tiny2.model.layers[0].self_attn.q_proj
    drafter = _Wrapper(base)
    assert drafter.alias.weight is not base.weight and drafter.alias.weight.data_ptr() == base.weight.data_ptr()
    config = DraftQuantConfig(group_size=32)
    assert set(draft_modules(drafter, config, tiny2.parameters())) == {"own"}


def test_prune_model_reports_induced_sparsity(tiny2, adapter_of) -> None:
    """A second identical run zeroes nothing new: its induced sparsity is 0. Weights that were zero before do not
    count as eligible; the overall zero fraction of the pruned maps is reported separately."""
    from economize.techniques.prune import PruneConfig, prune_model

    adapter = adapter_of(tiny2)
    config = PruneConfig(method="magnitude", sparsity=0.5)
    first = prune_model(adapter, config)
    assert first["sparsity"] == pytest.approx(0.5) and first["zeros"] == pytest.approx(0.5)
    assert first["eligible"] == first["weights"]
    second = prune_model(adapter, config)
    assert second["sparsity"] == 0 and second["zeros"] == pytest.approx(0.5)
    assert second["eligible"] == pytest.approx(first["weights"] / 2, rel=0.01)
    third = prune_model(adapter, PruneConfig(method="magnitude", sparsity=0.75))
    newly = third["zeros"] - second["zeros"]
    assert third["sparsity"] == pytest.approx(newly * third["weights"] / third["eligible"])
    assert third["sparsity"] == pytest.approx(0.5, abs=0.01)  # half of the remaining half


def test_torchao_install_hint_names_an_extra_that_installs_torchao(monkeypatch) -> None:
    import importlib.util
    import re

    from economize.errors import ConfigError
    from economize.techniques.quant import backends
    from helpers import read_pyproject

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "torchao" else real(name, *a))
    with pytest.raises(ConfigError) as error:
        backends._require_torchao()
    extra = re.search(r"economize\[(\w+)\]", str(error.value)).group(1)  # type: ignore[union-attr]
    pyproject = read_pyproject()
    requirements = pyproject["project"]["optional-dependencies"][extra]
    assert any(r.startswith("torchao") for r in requirements)
