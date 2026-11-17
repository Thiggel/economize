"""Environment variables (docs/CONTRIBUTING.md, design rule "No environment variables as configuration").

Options are typed fields of the object that owns them. An ``ECONOMIZE_*`` variable survives only as a documented
debug override of a kernel path. Every such name in ``src/economize`` must appear in the table of the design rule,
and every name in the table must still be read somewhere, so the code and the document cannot drift apart.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "economize"
CONTRIBUTING = ROOT / "docs" / "CONTRIBUTING.md"
NAME = re.compile(r"\bECONOMIZE_[A-Z0-9_]+\b")


def _documented() -> set[str]:
    section = CONTRIBUTING.read_text().split("**No environment variables as configuration.**", 1)[1]
    section = section.split("\n- **", 1)[0]
    return set(re.findall(r"^\s*\|\s*`(ECONOMIZE_[A-Z0-9_]+)`\s*\|", section, flags=re.MULTILINE))


def _read_in_src() -> set[str]:
    return {name for path in SRC.rglob("*.py") for name in NAME.findall(path.read_text())}


def test_every_environment_variable_is_a_documented_debug_override() -> None:
    undocumented = _read_in_src() - _documented()
    assert not undocumented, (
        f"{sorted(undocumented)} are read from the environment but not listed as debug overrides in "
        "docs/CONTRIBUTING.md; make them typed config fields instead (or document a genuine debug override)"
    )


def test_every_documented_override_is_still_read() -> None:
    stale = _documented() - _read_in_src()
    assert not stale, f"{sorted(stale)} are listed in docs/CONTRIBUTING.md but no longer appear in src/economize"
