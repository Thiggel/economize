"""economize reads no environment variable as configuration.

Options belong to the technique configs, state to the model or engine that owns it. The only ``ECONOMIZE_*``
variables are the debug overrides below: each switches one kernel path off (or forces it on) for debugging and
kernel measurements, is read in one place and is documented there as a debug override.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "economize"

#: Every debug override and the module that reads it.
DEBUG_OVERRIDES = {
    "ECONOMIZE_FLASH_DECODE": "attention/flash.py",
    "ECONOMIZE_FLASH_PREFILL": "attention/interface.py",
    "ECONOMIZE_HUB_KERNELS": "integrations/kernels.py",
    "ECONOMIZE_SLOT_GATHER": "modules/slot_gather.py",
}

_VARIABLE = re.compile(r"\b(?:ECONOMIZE|EFFKIT)_[A-Z0-9_]+\b")


def test_only_the_documented_debug_overrides_are_read() -> None:
    found: dict[str, set[str]] = {}
    for path in SRC.rglob("*.py"):
        for name in _VARIABLE.findall(path.read_text()):
            found.setdefault(name, set()).add(path.relative_to(SRC).as_posix())
    assert set(found) == set(DEBUG_OVERRIDES), "a new ECONOMIZE_* variable: make it a config option instead"
    for name, module in DEBUG_OVERRIDES.items():
        assert found[name] == {module}, f"{name} is read outside {module}"


def test_each_debug_override_is_documented_as_one() -> None:
    for name, module in DEBUG_OVERRIDES.items():
        text = re.sub(r"\s*\n\s*(?:#:?\s*)?", " ", (SRC / module).read_text())  # comments joined to one line
        near = rf"debug override.{{0,160}}{name}|{name}.{{0,160}}debug override"
        assert re.search(near, text, flags=re.IGNORECASE), f"{module} does not document {name} as a debug override"
