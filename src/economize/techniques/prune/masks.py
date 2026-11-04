"""Sparsity patterns and pruning masks.

A mask is a boolean tensor of the weight's shape that is ``True`` where a weight is removed. ``N:M`` follows the
NVIDIA convention: at most ``N`` nonzeros in every group of ``M`` consecutive weights along the input dimension, so
``2:4`` removes two weights of every four and is the pattern that sparse tensor cores (sm80+) accelerate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import Tensor

from ...errors import ConfigError

__all__ = ["SparsityPattern", "check_nm", "nm_mask", "parse_pattern", "unstructured_mask"]

_NM = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s*$")


@dataclass(frozen=True)
class SparsityPattern:
    """Either an unstructured fraction of removed weights or an ``N:M`` pattern (``n`` kept of every ``m``)."""

    fraction: float
    n: int | None = None
    m: int | None = None

    @property
    def structured(self) -> bool:
        return self.m is not None

    def __str__(self) -> str:
        return f"{self.n}:{self.m}" if self.structured else f"{self.fraction:g}"


def parse_pattern(value: float | str) -> SparsityPattern:
    """Parse ``0.5`` (unstructured) or ``"2:4"`` (N:M). Raise :class:`ConfigError` on anything else."""
    if isinstance(value, str):
        match = _NM.match(value)
        if match is None:
            try:
                value = float(value)
            except ValueError:
                raise ConfigError(f"sparsity must be a fraction in [0, 1) or 'N:M', got {value!r}") from None
        else:
            n, m = int(match.group(1)), int(match.group(2))
            if not 0 < n < m:
                raise ConfigError(f"N:M sparsity needs 0 < N < M, got {n}:{m}")
            return SparsityPattern(fraction=(m - n) / m, n=n, m=m)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) < 1.0:
        raise ConfigError(f"sparsity must be a fraction in [0, 1) or 'N:M', got {value!r}")
    return SparsityPattern(fraction=float(value))


def unstructured_mask(score: Tensor, fraction: float, *, per_row: bool = True) -> Tensor:
    """Remove the ``fraction`` of weights with the lowest ``score``, in every output row or over the whole matrix.

    Exactly ``int(fraction * count)`` weights are removed per row (or matrix). Ties are broken by position with a
    stable sort, as in the Wanda reference. (The references' per-matrix variant, ``score <= sorted[k]``, removes one
    extra weight; this one removes exactly ``k``.)
    """
    mask = torch.zeros_like(score, dtype=torch.bool)
    if per_row:
        k = int(score.shape[-1] * fraction)
        if k:
            order = torch.sort(score, dim=-1, stable=True).indices[..., :k]
            mask.scatter_(-1, order, True)
        return mask
    k = int(score.numel() * fraction)
    if k:
        order = torch.sort(score.reshape(-1), stable=True).indices[:k]
        mask.view(-1)[order] = True
    return mask


def nm_mask(score: Tensor, n: int, m: int) -> Tensor:
    """Remove the ``m - n`` lowest-score weights of every ``m`` consecutive weights along the last dimension."""
    rows, cols = score.shape
    if cols % m:
        raise ConfigError(f"N:M sparsity needs the input dimension ({cols}) to be a multiple of M={m}")
    groups = score.reshape(rows, cols // m, m)
    drop = torch.topk(groups, m - n, dim=-1, largest=False).indices
    mask = torch.zeros_like(groups, dtype=torch.bool).scatter_(-1, drop, True)
    return mask.reshape(rows, cols)


def check_nm(weight: Tensor, n: int, m: int) -> bool:
    """Whether every group of ``m`` consecutive weights along the last dimension has at most ``n`` nonzeros."""
    if weight.shape[-1] % m:
        return False
    nonzeros = (weight.reshape(-1, m) != 0).sum(-1)
    return bool((nonzeros <= n).all())
