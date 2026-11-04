"""SparseGPT.

Frantar & Alistarh, "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot", ICML 2023
(https://github.com/IST-DASLab/sparsegpt). Columns are processed left to right in blocks of 128. At the start of a
block the mask is chosen by the OBS saliency ``w^2 / [H^{-1}]_jj^2`` (unstructured: lowest fraction of the whole
block; N:M: lowest ``M - N`` of every ``M`` columns, chosen when the group starts), and the error of every removed
weight is compensated on the remaining columns through the rows of the Cholesky factor of ``H^{-1}`` (the sweep of
:func:`~economize.techniques._weights.obs.obs_sweep`, shared with GPTQ). This is a line-by-line port of
``SparseGPT.fasterprune`` without the optional joint quantization.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ...errors import ConfigError
from .._weights.hessian import inverse_hessian_cholesky
from .._weights.obs import ColumnRule, ObsBlock, obs_sweep
from .masks import SparsityPattern

__all__ = ["sparsegpt_prune"]


@torch.no_grad()
def sparsegpt_prune(
    weight: Tensor, hessian: Tensor, pattern: SparsityPattern, *, blocksize: int = 128, percdamp: float = 0.01
) -> Tensor:
    """Return the pruned and error-compensated weight (same dtype and device as ``weight``).

    Args:
        weight: ``[out, in]`` weight.
        hessian: ``[in, in]`` sum of ``x x^T`` over the calibration tokens.
        pattern: Unstructured fraction or N:M pattern.
        blocksize: Columns per lazy-update block (128 in the paper).
        percdamp: Relative diagonal damping of ``H``.

    Raises:
        ConfigError: An N:M pattern whose ``M`` does not divide the input dimension and ``blocksize`` (the last
            group would be cut short: pruned beyond ``N:M``, or too narrow to drop ``M - N`` weights).
    """
    W = weight.detach().float().clone()
    columns = W.shape[1]
    keep_n, group_m = (pattern.n, pattern.m) if pattern.structured else (0, 0)
    assert keep_n is not None and group_m is not None
    if group_m and (columns % group_m or blocksize % group_m):
        raise ConfigError(
            f"N:M sparsity needs the input dimension ({columns}) and blocksize ({blocksize}) to be multiples of "
            f"M={group_m}"
        )
    Hinv, _ = inverse_hessian_cholesky(hessian, W, percdamp)
    obs_sweep(W, Hinv, _SparseGPTRule(pattern.fraction, group_m, group_m - keep_n), blocksize)
    return W.to(dtype=weight.dtype)


class _SparseGPTRule(ColumnRule):
    """Zero the masked entries of a column. The unstructured mask of a block is chosen when the block starts, the
    N:M mask of a group of ``M`` columns when the group's first column is reached (on the compensated columns)."""

    def __init__(self, fraction: float, group_m: int, drop_n: int) -> None:
        self.fraction = fraction
        self.group_m = group_m
        self.drop_n = drop_n
        self.mask = torch.zeros(0, dtype=torch.bool)

    def begin_block(self, block: ObsBlock) -> None:
        W1 = block.weight
        self.mask = torch.zeros_like(W1, dtype=torch.bool)
        if self.group_m == 0:
            saliency = W1**2 / torch.diag(block.hinv).reshape(1, -1) ** 2
            k = int(saliency.numel() * self.fraction)
            if k:
                self.mask.view(-1)[torch.sort(saliency.reshape(-1), stable=True).indices[:k]] = True

    def replace(self, block: ObsBlock, i: int) -> Tensor:
        if self.group_m and i % self.group_m == 0:
            hdiag = torch.diag(block.hinv)
            saliency = block.weight[:, i : i + self.group_m] ** 2 / hdiag[i : i + self.group_m].reshape(1, -1) ** 2
            self.mask.scatter_(1, i + torch.topk(saliency, self.drop_n, dim=1, largest=False).indices, True)
        q = block.weight[:, i].clone()
        q[self.mask[:, i]] = 0
        return q
