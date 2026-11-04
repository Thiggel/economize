"""The generic registry behind every extension point."""

from __future__ import annotations

import importlib
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from economize import DuplicateNameError, UnknownNameError
from economize.registry import Registry


def test_register_and_get() -> None:
    registry: Registry[int] = Registry("number")
    assert registry.register("one", 1) == 1
    assert registry.get("one") == 1
    assert "one" in registry and "two" not in registry
    assert registry.names() == ["one"] and list(registry) == ["one"] and len(registry) == 1


def test_register_as_decorator() -> None:
    registry: Registry[type] = Registry("class")

    @registry.register("thing")
    class Thing:
        pass

    assert registry.get("thing") is Thing


def test_duplicate_name_needs_override() -> None:
    registry: Registry[int] = Registry("number")
    registry.register("x", 1)
    registry.register("x", 1)  # re-registering the same object is harmless
    with pytest.raises(DuplicateNameError, match="override=True"):
        registry.register("x", 2)
    registry.register("x", 2, override=True)
    assert registry.get("x") == 2


def test_unknown_name_lists_known_names() -> None:
    registry: Registry[int] = Registry("number")
    registry.register("a", 1)
    registry.register("b", 2)
    with pytest.raises(UnknownNameError, match="unknown number 'c'; registered: 'a', 'b'"):
        registry.get("c")
    with pytest.raises(KeyError):  # UnknownNameError is a KeyError
        registry.unregister("c")
    assert registry.unregister("a") == 1 and registry.names() == ["b"]


def _entry_point(name: str, load):  # the part of importlib.metadata.EntryPoint that Registry uses
    return SimpleNamespace(name=name, value=f"plugin:{name}", load=load)


def test_entry_point_plugins_load_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def entry_points(group: str):
        calls.append(group)
        return [_entry_point("plugged", lambda: 42), _entry_point("builtin", lambda: -1)]

    monkeypatch.setattr("economize.registry.metadata.entry_points", entry_points)
    registry: Registry[int] = Registry("number", "economize.test_group")
    registry.register("builtin", 7)
    assert registry.get("builtin") == 7 and calls == []  # no scan while every lookup hits
    assert registry.get("plugged") == 42 and calls == ["economize.test_group"]
    assert registry.names() == ["builtin", "plugged"]  # registered entries win over plugins of the same name
    assert calls == ["economize.test_group"]  # scanned once


def test_broken_plugin_is_logged_and_skipped(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def fail():
        raise ImportError("missing dependency")

    monkeypatch.setattr(
        "economize.registry.metadata.entry_points",
        lambda group: [_entry_point("broken", fail), _entry_point("ok", lambda: 1)],
    )
    registry: Registry[int] = Registry("number", "economize.test_group")
    assert registry.names() == ["ok"]
    assert "could not load number plugin 'broken'" in caplog.text


def test_builtin_registries_are_populated() -> None:
    import economize
    from economize.arch import ARCHITECTURES
    from economize.decoding import POLICIES, VERIFIERS
    from economize.techniques.jump import MIXERS, PREDICTORS

    assert "jump" in economize.TECHNIQUES
    assert {"economize", "tskip"} <= set(economize.FORMATS)
    assert {"qwen2", "llama"} <= set(ARCHITECTURES)
    assert {"greedy", "accept_all"} <= set(VERIFIERS)
    assert {"fixed", "confidence"} <= set(POLICIES)
    assert {"mlp", "linear"} <= set(PREDICTORS)
    assert {"concat", "fused_linear", "layer_attention"} <= set(MIXERS)


# ---- plugin removal, unavailable entries and the documented registry list ----
def _entry_points(monkeypatch: pytest.MonkeyPatch, **plugins: Any) -> None:
    """Make ``importlib.metadata.entry_points`` return one fake entry point per keyword."""
    import economize.registry as module

    def load(value: Any) -> Any:
        if isinstance(value, Exception):
            raise value
        return value

    points = [SimpleNamespace(name=name, value=f"pkg:{name}", load=lambda v=value: load(v))
              for name, value in plugins.items()]
    monkeypatch.setattr(module.metadata, "entry_points", lambda group: points)


def test_unregister_removes_a_plugin_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plugin can be unregistered before anything looked it up (and it stays removed)."""
    _entry_points(monkeypatch, plugin=7)
    registry: Registry[int] = Registry("number", "economize.test_numbers")
    assert registry.unregister("plugin") == 7
    assert "plugin" not in registry


def test_unknown_names_list_unavailable_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    _entry_points(monkeypatch, broken=ImportError("no module named 'kernels_x'"))
    registry: Registry[int] = Registry("number", "economize.test_numbers")
    with pytest.raises(UnknownNameError, match=r"broken.*kernels_x"):
        registry.get("broken")
    registry.unavailable["vendor"] = "optional dependency 'vendor_lib' is not installed"
    with pytest.raises(UnknownNameError, match="vendor_lib"):
        registry.get("other")


def test_unknown_technique_mentions_unavailable_subpackages(monkeypatch: pytest.MonkeyPatch) -> None:
    """``economize.techniques.unavailable`` (subpackages skipped at import) shows up in the error of a technique
    lookup, so that a missing optional dependency is named where the user meets it."""
    import economize.techniques as techniques
    from economize.core.technique import TECHNIQUES

    monkeypatch.setitem(techniques.unavailable, "fancy", "optional dependency 'fancy_lib' is not installed")
    with pytest.raises(UnknownNameError, match="fancy_lib"):
        TECHNIQUES.get("fancy_technique")


@pytest.mark.parametrize(("document", "heading"), [
    ("docs/extending.md", "## Registries and entry-point groups"),
    ("docs/ARCHITECTURE.md", "### 3.12 Registries and entry-point groups"),
])
def test_docs_list_every_registry_and_its_entry_point_group(document: str, heading: str) -> None:
    """The registry table of the extension guide and of the architecture document (Section 3.12) names every registry
    with an entry-point group, and the module named there exports it. Registries of unlisted techniques are left out
    of the documentation and of this check."""
    import ast
    import importlib
    import re
    from pathlib import Path

    import economize

    root = Path(__file__).resolve().parents[2]
    hidden = set(economize.TECHNIQUES.names()) - set(economize.listed_techniques())
    found: dict[str, str] = {}
    for path in (root / "src" / "economize").rglob("*.py"):
        if path.parent.parent.name == "techniques" and path.parent.name in hidden:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
            if (isinstance(value, ast.Call) and getattr(value.func, "id", None) == "Registry" and len(value.args) == 2
                    and isinstance(value.args[1], ast.Constant)):
                target = node.targets[0] if isinstance(node, ast.Assign) else node.target
                found[target.id] = value.args[1].value  # type: ignore[union-attr]
    text = (root / document).read_text()
    table = text[text.index(heading) + len(heading):]
    level = heading.split(" ")[0]
    end = re.search(rf"^#{{1,{len(level)}}} ", table, re.MULTILINE)  # the next heading of the same or a higher level
    table = table[:end.start()] if end else table
    rows = {}
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")] if line.startswith("| `") else []
        if len(cells) == 3:
            for name in re.findall(r"`(\w+)`", cells[0]):
                rows[name] = (cells[1].strip("`"), re.findall(r"`([\w.]+)`", cells[2]))
    assert found and set(found) <= set(rows), sorted(set(found) - set(rows))
    for name, group in found.items():
        module, groups = rows[name]
        assert group in groups, (name, group, groups)
        assert hasattr(importlib.import_module(module), name), (module, name)


def test_broken_technique_subpackage_does_not_break_discovery(monkeypatch, caplog) -> None:
    techniques = importlib.import_module("economize.techniques")
    real = importlib.import_module

    def flaky(name: str, *args: object, **kwargs: object):
        if name == "economize.techniques.lookup":
            raise RuntimeError("CUDA driver too old")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", flaky)  # the module import_submodules calls through
    monkeypatch.setattr(techniques, "unavailable", {}, raising=False)
    with caplog.at_level(logging.WARNING, logger="economize.registry"):
        names = techniques._import_subpackages()
    assert "lookup" not in names and "dynamic_depth" in names and "frspec" in names
    assert "lookup" in techniques.unavailable and "CUDA driver too old" in techniques.unavailable["lookup"]
    assert any("lookup" in record.getMessage() for record in caplog.records)
