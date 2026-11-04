"""Import layering (docs/ARCHITECTURE.md, Section 2.1): a module imports only its own layer and lower ones.

Every module under ``src/economize`` is parsed with :mod:`ast`; module-level, function-level and ``TYPE_CHECKING``
imports all count. The layer table is read from the architecture document, so the document and the test cannot
drift apart.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "economize"
ARCHITECTURE = ROOT / "docs" / "ARCHITECTURE.md"
PACKAGE_ROOT = "<root>"  # economize/__init__.py


def _layers() -> dict[str, int]:
    """The ``| Layer | Packages |`` table of Section 2.1, as ``{top-level name: layer}``."""
    text = ARCHITECTURE.read_text()
    section = text.split("### 2.1 Layering", 1)[1].split("\n## ", 1)[0]
    layers: dict[str, int] = {}
    for match in re.finditer(r"^\|\s*(\d+)\s*\|(.*)\|\s*$", section, flags=re.MULTILINE):
        for name in re.findall(r"`([A-Za-z_]+)`", match.group(2)):
            layers[name] = int(match.group(1))
    layers[PACKAGE_ROOT] = max(layers.values())
    return layers


LAYERS = _layers()


def _top_level_names() -> set[str]:
    names = {p.stem for p in SRC.glob("*.py") if p.name != "__init__.py"}
    return names | {p.name for p in SRC.iterdir() if p.is_dir() and (p / "__init__.py").exists()}


def _module_name(path: Path, src: Path = SRC) -> list[str]:
    """Dotted parts of ``path`` below ``economize`` (``[]`` for the package root)."""
    parts = list(path.relative_to(src).with_suffix("").parts)
    return parts[:-1] if parts[-1] == "__init__" else parts


def _exists(parts: list[str], src: Path) -> bool:
    target = src.joinpath(*parts)
    return target.with_suffix(".py").exists() or (target / "__init__.py").exists()


def _imported(path: Path, src: Path = SRC) -> Iterator[tuple[int, list[str]]]:
    """``(line, parts below economize)`` of every economize module that ``path`` imports; ``[]`` is the package root."""
    parts = _module_name(path, src)
    package = parts if path.name == "__init__.py" else parts[:-1]
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "economize" or alias.name.startswith("economize."):
                    yield node.lineno, alias.name.split(".")[1:]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level + 1] if node.level > 1 else list(package)
                target = base + (node.module.split(".") if node.module else [])
            elif node.module and (node.module == "economize" or node.module.startswith("economize.")):
                target = node.module.split(".")[1:]
            else:
                continue
            for alias in node.names:  # ``from pkg import submodule`` imports the submodule
                child = [*target, alias.name]
                yield node.lineno, child if _exists(child, src) else target


def _top(parts: list[str]) -> str:
    return parts[0] if parts else PACKAGE_ROOT


def _edges() -> list[tuple[str, int, str, str]]:
    """``(file, line, importer's top-level name, imported top-level name)`` of every cross-package import."""
    edges = []
    for path in sorted(SRC.rglob("*.py")):
        source = _top(_module_name(path))
        for line, target in _imported(path):
            if _top(target) != source:
                edges.append((str(path.relative_to(SRC)), line, source, _top(target)))
    return edges


EDGES = _edges()


def test_the_layer_table_names_every_top_level_module_and_package() -> None:
    assert _top_level_names() <= set(LAYERS), f"add {sorted(_top_level_names() - set(LAYERS))} to Section 2.1"
    assert set(LAYERS) - {PACKAGE_ROOT} <= _top_level_names(), "Section 2.1 names a package that does not exist"


def test_no_module_imports_a_higher_layer() -> None:
    upward = [
        f"{file}:{line} ({source}, layer {LAYERS[source]}) imports {target} (layer {LAYERS[target]})"
        for file, line, source, target in EDGES
        if LAYERS[target] > LAYERS[source]
    ]
    assert not upward, "upward imports:\n" + "\n".join(upward)


def test_the_core_imports_only_lower_layers() -> None:
    allowed = {name for name, layer in LAYERS.items() if layer < LAYERS["core"]}
    imported = {target for _, _, source, target in EDGES if source == "core"}
    assert imported <= allowed, f"economize.core imports {sorted(imported - allowed)}"


def test_only_the_package_root_imports_the_package_root() -> None:
    offenders = [f"{file}:{line}" for file, line, _source, target in EDGES if target == PACKAGE_ROOT]
    assert not offenders, "import the defining subpackage instead of economize itself: " + ", ".join(offenders)


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        ("from .. import __version__", PACKAGE_ROOT),
        ("from ..integrations.generate import generate", "integrations"),
        ("from ..integrations import generate", "integrations"),
        ("import economize", PACKAGE_ROOT),
        ("import economize.integrations.generate", "integrations"),
        ("from economize import integrations", "integrations"),
    ],
)
def test_the_checker_finds_upward_imports(statement: str, expected: str, tmp_path: Path) -> None:
    """The parser itself: an upward import inside a function of a core module is found and resolved."""
    package = tmp_path / "economize"
    for name in ("core", "integrations"):
        (package / name).mkdir(parents=True)
        (package / name / "__init__.py").write_text("")
    (package / "__init__.py").write_text("__version__ = '0'\n")
    (package / "integrations" / "generate.py").write_text("def generate(): ...\n")
    module = package / "core" / "model.py"
    module.write_text(f"def f():\n    {statement}\n")
    targets = [_top(parts) for _, parts in _imported(module, package)]
    assert targets == [expected]
    assert LAYERS[expected] > LAYERS["core"]
