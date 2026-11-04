"""Deprecated module path: the jump drafter's vLLM model moved to :mod:`economize.backends.vllm.jump.model`.

``SYNC_PREFIX`` (now :data:`economize.backends.vllm.drafters.SYNC_PREFIX`, shared by every drafter) and
``DRAFTER_ARCHITECTURE`` (also :data:`economize.backends.vllm.DRAFTER_ARCHITECTURE`) stay importable from here
without a warning; the other names warn.
"""

from __future__ import annotations

from types import ModuleType

from ..._deprecation import moved_getattr
from .drafters import SYNC_PREFIX
from .jump.model import DRAFTER_ARCHITECTURE

__all__ = ["DRAFTER_ARCHITECTURE", "SYNC_PREFIX"]

_MOVED = ("STACKED_PARAMS", "EconomizeJumpEagle3ForCausalLM", "TargetModules", "_copy_layer")


def _model() -> ModuleType:
    from .jump import model

    return model


__getattr__ = moved_getattr(__name__, _MOVED, "economize.backends.vllm.jump.model", _model)
