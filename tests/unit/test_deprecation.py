"""``economize._deprecation.moved_getattr``: the module ``__getattr__`` behind every deprecation shim."""

from __future__ import annotations

import sys
import types
import warnings

import pytest

from economize._deprecation import moved_getattr


@pytest.fixture
def shim(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """A module ``old`` whose names moved to a module ``new`` (one renamed) or were renamed within ``old``."""
    new = types.ModuleType("new")
    new.Thing, new.renamed = object(), object()  # type: ignore[attr-defined]
    old = types.ModuleType("old")
    old.current = new.renamed  # type: ignore[attr-defined]
    old.__getattr__ = moved_getattr("old", {"Thing": "Thing", "original": "renamed"}, "pkg.new", lambda: new)  # type: ignore[method-assign]
    monkeypatch.setitem(sys.modules, "old", old)
    return old


def test_moved_names_warn_and_return_the_same_object(shim: types.ModuleType) -> None:
    with pytest.warns(DeprecationWarning, match=r"^old\.Thing is deprecated; use pkg\.new\.Thing instead$"):
        assert shim.Thing is sys.modules["old"].__getattr__("Thing")
    with pytest.warns(DeprecationWarning, match=r"^old\.original is deprecated; use pkg\.new\.renamed instead$"):
        assert shim.original is shim.current


def test_unknown_names_raise_attribute_error_without_a_warning(shim: types.ModuleType) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(AttributeError, match="module 'old' has no attribute 'missing'"):
            _ = shim.missing
        assert not hasattr(shim, "missing")


def test_renames_within_a_module_and_name_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("solo")
    module.new_name = object()  # type: ignore[attr-defined]
    module.__getattr__ = moved_getattr("solo", {"old_name": "new_name"})  # type: ignore[method-assign]
    monkeypatch.setitem(sys.modules, "solo", module)
    with pytest.warns(DeprecationWarning, match=r"solo\.old_name is deprecated; use solo\.new_name"):
        assert module.old_name is module.new_name
    module.__getattr__ = moved_getattr("solo", ["new_name"], "elsewhere", lambda: module)  # type: ignore[method-assign]
    with pytest.warns(DeprecationWarning, match=r"use elsewhere\.new_name"):
        assert module.__getattr__("new_name") is module.new_name


def test_the_warning_points_at_the_importing_code(shim: types.ModuleType) -> None:
    with pytest.warns(DeprecationWarning) as records:
        _ = shim.Thing
    assert records[0].filename == __file__
