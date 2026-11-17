"""The libdevice functions of the attention kernels.

Compiled kernels round every step on its own through libdevice's ``*_rn`` functions, so that no two steps contract
into one fused multiply-add. The Triton interpreter (``TRITON_INTERPRET=1``) cannot call extern functions, and it
computes in numpy, which rounds every operation on its own anyway, so there the same names are plain arithmetic.
"""

from __future__ import annotations

import os
from typing import Any


def _interpreting() -> bool:
    """Whether Triton runs kernels in its interpreter (the same truthy values as ``triton.knobs``)."""
    return os.environ.get("TRITON_INTERPRET", "0").lower() in ("1", "true", "on", "yes", "y")


class _Interpreted:
    """The ``*_rn`` functions as plain arithmetic, for the Triton interpreter."""

    @staticmethod
    def add_rn(a: Any, b: Any) -> Any:
        return a + b

    @staticmethod
    def sub_rn(a: Any, b: Any) -> Any:
        return a - b

    @staticmethod
    def mul_rn(a: Any, b: Any) -> Any:
        return a * b

    @staticmethod
    def div_rn(a: Any, b: Any) -> Any:
        return a / b


def _compiled() -> Any:
    """libdevice of the installed triton (its module moved under ``extra.cuda`` in triton 3)."""
    try:
        from triton.language.extra.cuda import libdevice as compiled
    except ImportError:  # pragma: no cover - older triton
        from triton.language.extra import libdevice as compiled
    return compiled


libdevice: Any = _Interpreted() if _interpreting() else _compiled()

__all__ = ["libdevice"]
