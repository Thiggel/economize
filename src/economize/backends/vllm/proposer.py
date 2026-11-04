"""Deprecated module path: the jump proposers moved to :mod:`economize.backends.vllm.jump.proposer`."""

from __future__ import annotations

from types import ModuleType

from ..._deprecation import moved_getattr

__all__: list[str] = []

_MOVED = ("JumpProposer", "JumpSpeculator")


def _proposer() -> ModuleType:
    from .jump import proposer

    return proposer


__getattr__ = moved_getattr(__name__, _MOVED, "economize.backends.vllm.jump.proposer", _proposer)
