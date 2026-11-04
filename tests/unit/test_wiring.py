"""The seams that keep the layering of docs/ARCHITECTURE.md (Section 2.1): the version module, the composition root
that wires ``EfficientModel.generate`` to the integrations, and the names lower layers expose to higher ones."""

from __future__ import annotations

from pathlib import Path
from typing import Any, get_args

import pytest
import torch

import economize
from economize import _version
from economize.core import GENERATE_IMPLEMENTATIONS, EfficientModel
from economize.errors import UnknownNameError
from economize.runtime import Phase
from helpers import read_pyproject

ROOT = Path(__file__).resolve().parents[2]


def test_the_version_has_one_source() -> None:
    assert economize.__version__ == _version.__version__
    project = read_pyproject()
    assert "version" not in project["project"]
    assert "version" in project["project"]["dynamic"]
    assert ROOT / project["tool"]["hatch"]["version"]["path"] == Path(_version.__file__).resolve()


def test_generate_goes_through_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "transformers" in GENERATE_IMPLEMENTATIONS
    calls: list[tuple[Any, Any, dict[str, Any]]] = []

    def fake(model: Any, inputs: Any = None, **kwargs: Any) -> str:
        calls.append((model, inputs, kwargs))
        return "generated"

    GENERATE_IMPLEMENTATIONS.register("test-fake", fake)
    try:
        monkeypatch.setattr(EfficientModel, "generate_implementation", "test-fake")
        model = object.__new__(EfficientModel)  # generate only reads the class attribute and the registry
        ids = torch.tensor([[1, 2]])
        assert EfficientModel.generate(model, ids, max_new_tokens=3) == "generated"
        assert calls == [(model, ids, {"max_new_tokens": 3})]
        monkeypatch.setattr(EfficientModel, "generate_implementation", "missing")
        with pytest.raises(UnknownNameError, match="generate implementation 'missing'"):
            EfficientModel.generate(model, ids)
    finally:
        GENERATE_IMPLEMENTATIONS.unregister("test-fake")


def test_the_transformers_generate_resolves_to_the_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    from economize.integrations import generate as integration

    calls: list[Any] = []

    def fake(model: Any, inputs: Any = None, **kwargs: Any) -> str:
        calls.append((model, inputs, kwargs))
        return "out"

    monkeypatch.setattr(integration, "generate", fake)  # the wiring imports it on every call
    assert GENERATE_IMPLEMENTATIONS.get("transformers")("model", "inputs", top_k=1) == "out"
    assert calls == [("model", "inputs", {"top_k": 1})]


def test_describe_quantization_lives_in_the_core() -> None:
    from economize.core import hub
    from economize.integrations import quantized

    assert quantized.describe_quantization is hub.describe_quantization
    assert "describe_quantization" in hub.__all__ and "describe_quantization" in quantized.__all__


def test_the_stop_checker_is_public() -> None:
    from economize.decoding import StopChecker, stop

    assert "StopChecker" in stop.__all__
    row = torch.tensor([5, 7, 9])
    assert StopChecker(7)(7, row, 1) and not StopChecker([3, 4])(7, row, 1)
    assert StopChecker(lambda tokens: tokens.numel() == 2)(7, row, 1)
    assert not StopChecker(None)(7, row, 1)


def test_eviction_passes_use_the_runtime_phase() -> None:
    from economize.techniques.kv_evict import PassInfo

    assert PassInfo.__annotations__["phase"] == "Phase | None"
    bounds = torch.zeros(1, dtype=torch.long)
    assert PassInfo("prefill", bounds, bounds).prefill
    assert not any(PassInfo(p, bounds, bounds).prefill for p in (*get_args(Phase), None) if p != "prefill")
