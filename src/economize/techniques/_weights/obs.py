"""The blocked Optimal Brain Surgeon column sweep shared by SparseGPT and GPTQ.

Both references (``SparseGPT.fasterprune`` and ``GPTQ.fasterquant``) walk the columns of a weight matrix left to
right in blocks of ``blocksize`` columns. Each column is replaced by a pruned or rounded version ``q``, and its error
``(w - q) / [H^{-1}]_ii`` is compensated on the remaining columns of the block through row ``i`` of the upper
Cholesky factor of ``H^{-1}``; the errors of a whole block reach the columns after it in one lazy update. The two
methods differ only in how a column is replaced, which a :class:`ColumnRule` supplies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["ColumnRule", "ObsBlock", "obs_sweep"]


@dataclass(frozen=True)
class ObsBlock:
    """One block of the sweep.

    Attributes:
        start: Index of the block's first column in the swept matrix.
        weight: ``[rows, count]`` working copy of the block's columns, updated in place as earlier columns of the
            block are compensated (column ``i`` is final when the rule is asked for it).
        hinv: ``[count, count]`` block of the Cholesky factor of ``H^{-1}``.
    """

    start: int
    weight: Tensor
    hinv: Tensor


class ColumnRule(ABC):
    """How the sweep replaces a column: prune it (SparseGPT) or round it to a grid (GPTQ)."""

    def begin_block(self, block: ObsBlock) -> None:
        """Called before the first column of every block (SparseGPT chooses its unstructured mask here)."""

    @abstractmethod
    def replace(self, block: ObsBlock, i: int) -> Tensor:
        """The replacement ``q [rows]`` of column ``i`` of ``block`` (``block.weight[:, i]``)."""


@torch.no_grad()
def obs_sweep(W: Tensor, Hinv: Tensor, rule: ColumnRule, blocksize: int = 128) -> None:
    """Replace the columns of ``W [rows, columns]`` (float32) left to right with ``rule``, in place, compensating
    each column's error on the columns after it.

    Args:
        W: The weight in sweep order; on return it holds the replaced columns.
        Hinv: ``[columns, columns]`` upper Cholesky factor of the damped inverse Hessian, in the same order.
        rule: Supplies the replacement of every column.
        blocksize: Columns per lazy-update block (128 in both papers).
    """
    columns = W.shape[1]
    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        block = ObsBlock(i1, W1, Hinv1)
        rule.begin_block(block)

        for i in range(i2 - i1):
            w = W1[:, i]
            d = Hinv1[i, i]
            q = rule.replace(block, i)
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err

        W[:, i1:i2] = Q1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
