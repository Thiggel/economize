"""The type of a compiled Triton kernel, shared by the techniques that ship one (Triton is imported lazily, and
untyped where it is installed, so the kernels are typed by what their callers do with them)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

__all__ = ["TritonKernel"]


class TritonKernel(Protocol):
    """A ``@triton.jit`` kernel: indexed by its launch grid, then called with its arguments and launch options."""

    def __getitem__(self, grid: tuple[int, ...]) -> Callable[..., object]: ...
