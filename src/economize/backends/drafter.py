"""Deprecated: the exported jump drafter format moved to :mod:`economize.techniques.jump.export`.

``DrafterSpec`` is now :class:`~economize.techniques.jump.export.JumpDrafterSpec`; the other names kept theirs. The
format on disk (the ``economize_jump`` block of ``config.json``, ``jump.*`` tensors) is unchanged.
"""

from __future__ import annotations

from types import ModuleType

from .._deprecation import moved_getattr

__all__: list[str] = []

_MOVED = {
    "CONFIG_FILE": "CONFIG_FILE",
    "CONFIG_KEY": "CONFIG_KEY",
    "FORMAT_VERSION": "FORMAT_VERSION",
    "WEIGHTS_FILE": "WEIGHTS_FILE",
    "WEIGHTS_PREFIX": "WEIGHTS_PREFIX",
    "DrafterSpec": "JumpDrafterSpec",
    "ExportedJumpDrafter": "ExportedJumpDrafter",
}


def _export() -> ModuleType:
    from ..techniques.jump import export

    return export


__getattr__ = moved_getattr(__name__, _MOVED, "economize.techniques.jump.export", _export)
