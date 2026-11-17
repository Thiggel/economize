"""Correctly rounded float arithmetic for the attention kernels.

When the kernels compile, :data:`ld` is triton's CUDA libdevice module: ``mul_rn`` and friends round every step to
nearest and stop the compiler from fusing a multiply and an add into an FMA, so that every kernel rounds a score
alike. The Triton interpreter (``TRITON_INTERPRET=1``) runs kernels on numpy and implements no extern function, so
there :data:`ld` holds the same functions as plain operators; numpy rounds every float32 operation on its own and
never fuses, which gives the same values.
"""

from __future__ import annotations

import os
from types import ModuleType, SimpleNamespace

import triton
import triton.language as tl  # noqa: F401  # older interpreters patch the triton.language a jit function sees

__all__ = ["interpreting", "ld"]


def interpreting() -> bool:
    """Whether triton runs kernels in its interpreter (``TRITON_INTERPRET=1`` when triton was imported)."""
    knobs = getattr(triton, "knobs", None)
    if knobs is not None:
        return bool(knobs.runtime.interpret)
    return os.environ.get("TRITON_INTERPRET", "0") == "1"  # pragma: no cover - triton < 3.4


@triton.jit
def _add_rn(a, b):
    return a + b


@triton.jit
def _sub_rn(a, b):
    return a - b


@triton.jit
def _mul_rn(a, b):
    return a * b


@triton.jit
def _div_rn(a, b):
    return a / b


def _libdevice() -> ModuleType | SimpleNamespace:
    if interpreting():
        return SimpleNamespace(add_rn=_add_rn, sub_rn=_sub_rn, mul_rn=_mul_rn, div_rn=_div_rn)
    try:
        from triton.language.extra.cuda import libdevice
    except ImportError:  # pragma: no cover - older triton
        from triton.language.extra import libdevice
    return libdevice


#: ``add_rn``, ``sub_rn``, ``mul_rn`` and ``div_rn`` for ``@triton.jit`` code (see the module docstring).
ld = _libdevice()
