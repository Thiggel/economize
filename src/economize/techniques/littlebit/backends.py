"""Run-time backends of :class:`~.module.LittleBitLinear`: how the packed factors multiply the input.

``torch`` unpacks the signs to ``±1`` in the activation dtype on every call and multiplies with dense matmuls. It is
the numerical reference, works on any device and dtype and supports autograd through the input. ``triton`` runs the
binary GEMM of :mod:`.kernel` for 16-bit CUDA inputs without autograd and falls back to the reference otherwise. A
new kernel is one :class:`LittleBitBackend` subclass registered in :data:`LITTLEBIT_BACKENDS`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ...registry import Registry
from .pack import unpack_signs

if TYPE_CHECKING:
    from .module import LittleBitLinear

__all__ = ["LITTLEBIT_BACKENDS", "LittleBitBackend", "TorchBackend", "TritonBackend", "reference_forward"]


class LittleBitBackend(ABC):
    """Computes the forward of a :class:`~.module.LittleBitLinear`."""

    @abstractmethod
    def forward(self, module: LittleBitLinear, x: Tensor) -> Tensor:
        """``module(x)`` for ``x [..., in]``."""


LITTLEBIT_BACKENDS: Registry[type[LittleBitBackend]] = Registry("LittleBit backend", "economize.littlebit_backends")


def reference_forward(module: LittleBitLinear, x: Tensor) -> Tensor:
    """The forward with unpacked signs and dense matmuls in the dtype of ``x``."""
    rows = x.reshape(-1, module.in_features)
    y = None
    for p in range(module.paths):
        v = unpack_signs(module.v_bits[p], module.in_features, x.dtype)
        t = ((rows * module.g[p].to(x.dtype)) @ v.T) * module.l[p].to(x.dtype)
        u = unpack_signs(module.u_bits[p], module.rank, x.dtype)
        part = (t @ u.T) * module.h[p].to(x.dtype)
        y = part if y is None else y + part
    assert y is not None
    if module.bias is not None:
        y = y + module.bias.to(x.dtype)
    return y.reshape(*x.shape[:-1], module.out_features)


@LITTLEBIT_BACKENDS.register("torch")
class TorchBackend(LittleBitBackend):
    """Unpack-and-matmul reference (any device, any dtype, autograd)."""

    def forward(self, module: LittleBitLinear, x: Tensor) -> Tensor:
        return reference_forward(module, x)


@LITTLEBIT_BACKENDS.register("triton")
class TritonBackend(LittleBitBackend):
    """The binary GEMM of :mod:`.kernel` for bfloat16 / float16 CUDA inputs outside autograd (the reference
    otherwise)."""

    def forward(self, module: LittleBitLinear, x: Tensor) -> Tensor:
        from .kernel import littlebit_matmul, triton_available

        usable = (
            x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
            and not (x.requires_grad and torch.is_grad_enabled())
            and triton_available()
        )
        if not usable:
            return reference_forward(module, x)
        scales = [getattr(module, name).to(x.dtype) for name in ("g", "l", "h")]
        bias = module.bias.to(x.dtype) if module.bias is not None else None
        y = littlebit_matmul(x.reshape(-1, module.in_features), module.v_bits, module.u_bits, *scales, bias)
        return y.reshape(*x.shape[:-1], module.out_features)
