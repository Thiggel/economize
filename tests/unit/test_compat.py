"""The engine version guards of economize._compat."""

from __future__ import annotations

import sys
import types

import pytest

from economize import UnsupportedEngineVersionError, UntestedEngineWarning
from economize._compat import (
    ALLOW_UNTESTED_ENV,
    ENGINES,
    check_engine,
    engine_allowed_untested,
    installed_version,
    private_import,
    require_engine,
)


@pytest.fixture(autouse=True)
def _no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ALLOW_UNTESTED_ENV, raising=False)


def _fake(monkeypatch: pytest.MonkeyPatch, name: str, version: str, **attributes: object) -> None:
    monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(__version__=version, **attributes))


@pytest.mark.parametrize(("version", "supported"), [
    ("0.30", True), ("0.30.4", True), ("0.30.1rc2+cu130", True), ("v0.30.0", True),
    ("0.29.9", False), ("0.31.0rc1", False), ("1.0.0", False), ("dev", False),
])
def test_supported_releases_are_the_half_open_range(version: str, supported: bool,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(monkeypatch, "vllm", version)
    assert (check_engine("vllm") is None) is supported


def test_an_unsupported_release_raises_naming_the_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(monkeypatch, "sglang", "0.6.1")
    with pytest.raises(UnsupportedEngineVersionError) as raised:
        require_engine("sglang")
    message = str(raised.value)
    assert "sglang 0.6.1" in message and ENGINES["sglang"].requirement in message and ALLOW_UNTESTED_ENV in message
    assert isinstance(raised.value, ImportError)


def test_a_supported_release_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(monkeypatch, "trl", "1.18.1")
    assert require_engine("trl") == "1.18.1"


@pytest.mark.parametrize("value", ["1", "all", "vllm", "sglang, vllm"])
def test_the_override_turns_the_error_into_a_warning(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(monkeypatch, "vllm", "0.32.0")
    monkeypatch.setenv(ALLOW_UNTESTED_ENV, value)
    assert engine_allowed_untested("vllm")
    with pytest.warns(UntestedEngineWarning, match="vllm 0.32.0"):
        assert require_engine("vllm") == "0.32.0"


def test_the_override_names_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(monkeypatch, "vllm", "0.32.0")
    monkeypatch.setenv(ALLOW_UNTESTED_ENV, "sglang")
    with pytest.raises(UnsupportedEngineVersionError):
        require_engine("vllm")


def test_the_imported_module_wins_over_the_installed_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata

    def metadata(name: str) -> str:
        return "9.9.9"

    monkeypatch.setattr(importlib.metadata, "version", metadata)
    _fake(monkeypatch, "trl", "1.13.0")
    assert installed_version("trl") == "1.13.0"
    monkeypatch.delitem(sys.modules, "trl")
    assert installed_version("trl") == "9.9.9"


def test_a_missing_engine_is_reported_with_the_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata

    def missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.delitem(sys.modules, "sglang", raising=False)
    monkeypatch.setattr(importlib.metadata, "version", missing)
    problem = check_engine("sglang")
    assert problem is not None and "not installed" in problem and ENGINES["sglang"].requirement in problem
    with pytest.raises(UnsupportedEngineVersionError):
        require_engine("sglang")


def test_private_import_checks_before_importing(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(monkeypatch, "vllm", "0.30.0")
    monkeypatch.setitem(sys.modules, "vllm.fake_private", types.SimpleNamespace(Hook="hook"))
    assert private_import("vllm", "vllm.fake_private:Hook") == "hook"
    _fake(monkeypatch, "vllm", "0.31.0")
    with pytest.raises(UnsupportedEngineVersionError):
        private_import("vllm", "vllm.fake_private:Hook")
