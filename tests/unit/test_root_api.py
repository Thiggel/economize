"""The root ``economize`` namespace: technique-independent API only.

Every technique's config lives in its subpackage, so adding a technique never edits ``economize/__init__.py``.
``economize.JumpConfig`` stays importable for code written against earlier releases, outside ``__all__``.
"""

from __future__ import annotations

import economize
from economize.core import TechniqueConfig


def _technique_configs(namespace: object, names: list[str]) -> list[str]:
    return sorted(name for name in names
                  if isinstance(obj := getattr(namespace, name), type) and issubclass(obj, TechniqueConfig)
                  and obj is not TechniqueConfig)


def test_the_root_exports_no_technique_config() -> None:
    from economize.techniques.jump import JumpConfig

    assert _technique_configs(economize, list(economize.__all__)) == []
    assert _technique_configs(economize, dir(economize)) == ["JumpConfig"]
    assert economize.JumpConfig is JumpConfig


def test_every_root_export_exists() -> None:
    missing = [name for name in economize.__all__ if not hasattr(economize, name)]
    assert not missing, missing
