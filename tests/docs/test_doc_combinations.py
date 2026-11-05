"""Every combination the documentation shows is accepted, and those of the main pages save for real.

The test reads the ``economize.apply`` calls with two or more techniques from the pages' Python blocks (also the
blocks marked ``not run``), evaluates their configurations as written, without the CPU rewrites of
:mod:`page_runner`, and checks that every pair is :func:`economize.compatible`. On the README, Getting started and
Examples pages it also checks that no technique, nor a drafter nested in a wrapper, takes a simulated path on the
``transformers`` backend; the tutorials and the weights page may show simulated techniques, and a call there whose
arguments the page builds at run time is skipped. A call inside a ``try`` block that catches
:class:`~economize.IncompatibleTechniquesError` demonstrates a refusal and is skipped.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from page_runner import DOCS, ROOT, blocks

import economize
from economize import SupportStatus, TechniqueConfig

PAGES = [ROOT / "README.md", DOCS / "getting_started.md", DOCS / "examples.md"]
OTHER_PAGES = [*sorted((DOCS / "tutorials").glob("*.md")), DOCS / "weights.md"]


@dataclasses.dataclass(frozen=True)
class Combination:
    """One ``economize.apply`` call of a page with its technique configurations."""

    page: str
    line: int
    configs: tuple[TechniqueConfig, ...]

    def __str__(self) -> str:
        return f"{self.page}:{self.line}"


def _is_apply(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "apply"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "economize"
    )


def _refusals(tree: ast.AST) -> set[int]:
    """The ids of the ``apply`` calls inside a ``try`` that catches ``IncompatibleTechniquesError``."""
    skipped: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(
            "IncompatibleTechniquesError" in ast.unparse(handler.type) for handler in node.handlers if handler.type
        ):
            skipped.update(id(call) for stmt in node.body for call in ast.walk(stmt) if _is_apply(call))
    return skipped


def _economize_imports(tree: ast.AST, namespace: dict[str, Any]) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("economize"):
            exec(compile(ast.Module([node], []), "<page>", "exec"), namespace)


def _config_assignments(tree: ast.Module, namespace: dict[str, Any], strict: bool) -> None:
    """Runs the top-level ``name = SomeConfig(...)`` assignments, such as ``eagle3 = Eagle3Config()``."""
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id.endswith("Config")
            and node.value.func.id in namespace
        ):
            try:
                exec(compile(ast.Module([node], []), "<page>", "exec"), namespace)
            except NameError:
                if strict:
                    raise


def combinations(page: Path, strict: bool = True) -> list[Combination]:
    """The ``economize.apply`` calls of ``page`` that attach two or more techniques.

    With ``strict=False`` a call whose arguments name something the page defines at run time is skipped.
    """
    namespace: dict[str, Any] = {"economize": economize}
    trees = [(block.line, ast.parse(block.code)) for block in blocks(page)]
    for _, tree in trees:
        _economize_imports(tree, namespace)
    for _, tree in trees:
        _config_assignments(tree, namespace, strict)
    found = []
    for line, tree in trees:
        refusals = _refusals(tree)
        for call in ast.walk(tree):
            if not _is_apply(call) or id(call) in refusals:
                continue
            assert isinstance(call, ast.Call)
            techniques = call.args[1:]
            if len(techniques) < 2:
                continue
            try:
                configs = tuple(eval(compile(ast.Expression(arg), "<page>", "eval"), namespace) for arg in techniques)
            except NameError:
                if strict:
                    raise
                continue
            found.append(Combination(page.relative_to(ROOT).as_posix(), line + call.lineno - 1, configs))
    return found


def _with_nested(config: TechniqueConfig) -> Iterator[TechniqueConfig]:
    """``config`` and the configurations it wraps (the drafter of FR-Spec, the fallback of the hybrid drafter)."""
    yield config
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        if isinstance(value, TechniqueConfig):
            yield from _with_nested(value)


ALL = [combination for page in PAGES for combination in combinations(page)]
OTHERS = [combination for page in OTHER_PAGES for combination in combinations(page, strict=False)]


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_every_page_shows_a_combination(page: Path) -> None:
    assert combinations(page), f"no economize.apply call with two or more techniques found in {page}"


@pytest.mark.parametrize("combination", ALL + OTHERS, ids=str)
def test_shown_combinations_are_compatible(combination: Combination) -> None:
    for first, second in itertools.combinations(combination.configs, 2):
        reason = economize.incompatibility(first, second)
        assert reason is None, f"{combination}: {reason}"


@pytest.mark.parametrize("combination", ALL, ids=str)
def test_shown_combinations_take_no_simulated_path(combination: Combination) -> None:
    for config in combination.configs:
        for part in _with_nested(config):
            support = economize.backend_support(part)["transformers"]
            assert support.status is not SupportStatus.SIMULATED, f"{combination}: {part.technique} {support.detail}"
