"""Bit packing of sign matrices: 32 signs per ``int32`` word along the last dimension.

Bit ``i`` of word ``j`` holds the sign of element ``32 j + i`` (1 for ``x >= 0``, 0 for ``x < 0``), so ``sign(0)`` is
``+1``. A last dimension that is not a multiple of 32 is padded with ``-1`` signs; the kernels mask the matching
inputs to zero, so the padding never reaches an output.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["WORD_BITS", "pack_signs", "packed_width", "unpack_signs"]

#: Signs per packed word.
WORD_BITS = 32


def packed_width(n: int) -> int:
    """Words that hold ``n`` signs."""
    return -(-n // WORD_BITS)


def _shifts(device: torch.device) -> Tensor:
    return torch.arange(WORD_BITS, device=device, dtype=torch.int32)


def pack_signs(x: Tensor) -> Tensor:
    """Pack the signs of ``x [..., n]`` into ``int32 [..., ceil(n / 32)]``."""
    n = x.shape[-1]
    bits = (x >= 0).to(torch.int64)
    bits = F.pad(bits, (0, packed_width(n) * WORD_BITS - n))
    bits = bits.reshape(*x.shape[:-1], -1, WORD_BITS)
    words = (bits << _shifts(x.device).to(torch.int64)).sum(-1)
    words = torch.where(words >= 2**31, words - 2**32, words)  # two's complement of the top bit
    return words.to(torch.int32)


def unpack_signs(words: Tensor, n: int, dtype: torch.dtype = torch.float32) -> Tensor:
    """The ``±1`` matrix ``[..., n]`` (in ``dtype``) that :func:`pack_signs` packed into ``words``."""
    bits = (words.unsqueeze(-1) >> _shifts(words.device)) & 1
    bits = bits.reshape(*words.shape[:-1], -1)[..., :n]
    return (bits * 2 - 1).to(dtype)
