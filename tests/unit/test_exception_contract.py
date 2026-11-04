"""Exception contract (docs/ARCHITECTURE.md, Section 2.1, last paragraph): caller-facing errors are economize errors.

A bad argument raises a :class:`~economize.errors.ConfigError` (through the ``economize.validation.check_*`` helpers
where one fits) or another :class:`~economize.errors.EconomizeError`, so that ``except EconomizeError`` catches every
configuration mistake. Every ``raise ValueError(...)``, ``raise TypeError(...)`` and every ``assert`` that does
more than narrow a type (``x is not None``, ``isinstance``) must appear in the allowlist below with its reason.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "economize"

# (path under src/economize, qualified function name, "ValueError" | "TypeError" | "assert") -> reason
ALLOWED: dict[tuple[str, str, str], str] = {
    ("bench/timing.py", "count_tokens", "TypeError"): "an argument of the wrong Python type",
    ("train/data/sources.py", "as_batch", "TypeError"): "an argument of the wrong Python type",
    ("train/data/sources.py", "as_source", "TypeError"): "an argument of the wrong Python type",
    ("techniques/prune/model.py", "prune_model.observer", "TypeError"): "a plugin breaks the observer contract",
    ("validation.py", "check_options", "TypeError"): "the helper itself is misused (not a dataclass)",
    ("core/metadata.py", "check_declaration", "TypeError"): "a technique class breaks its declaration contract "
    "(raised when the class is defined, like a wrong __init_subclass__ argument)",
    ("decoding/generation.py", "Generation.outputs", "assert"): "internal invariant: every request has finished",
    ("decoding/plain.py", "run_plain", "assert"): "internal invariant of the decode loop",
    ("techniques/quant/backends.py", "int4_tinygemm_tensors", "assert"): "internal invariant of the int4 packer",
}

# Sites in modules that are being reworked elsewhere; listed so the test passes now, never reported as stale.
PENDING: dict[tuple[str, str, str], str] = {}


def _narrows(test: ast.expr) -> bool:
    """``x is not None``, ``isinstance(...)`` or a conjunction of them: a type narrowing for the type checker."""
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        return all(_narrows(value) for value in test.values)
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.IsNot):
        return isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value is None
    return isinstance(test, ast.Call) and isinstance(test.func, ast.Name) and test.func.id == "isinstance"


def _visit(node: ast.AST, relative: str, scope: list[str]) -> Iterator[tuple[str, str, str, int]]:
    named = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    if named:
        scope.append(node.name)
    if (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id in ("ValueError", "TypeError")):
        yield relative, ".".join(scope), node.exc.func.id, node.lineno
    if isinstance(node, ast.Assert) and not _narrows(node.test):
        yield relative, ".".join(scope), "assert", node.lineno
    for child in ast.iter_child_nodes(node):
        yield from _visit(child, relative, scope)
    if named:
        scope.pop()


def _sites() -> Iterator[tuple[str, str, str, int]]:
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        yield from _visit(ast.parse(path.read_text(), filename=str(path)), relative, [])


def test_builtin_raises_and_asserts_are_allowlisted() -> None:
    found = list(_sites())
    unlisted = [f"{path}:{line} {name} ({kind})" for path, name, kind, line in found
                if (path, name, kind) not in ALLOWED and (path, name, kind) not in PENDING]
    assert not unlisted, (
        "raise a ConfigError (or another economize error, see economize.validation) instead, or add the site to "
        "ALLOWED with its reason:\n" + "\n".join(unlisted)
    )


def test_the_allowlist_has_no_stale_entries() -> None:
    present = {(path, name, kind) for path, name, kind, _ in _sites()}
    assert not set(ALLOWED) - present, f"remove from ALLOWED: {sorted(set(ALLOWED) - present)}"
