"""Magnitude pruning and Wanda.

Wanda: Sun, Liu, Bair & Kolter, "A Simple and Effective Pruning Approach for Large Language Models", ICLR 2024
(https://github.com/locuslab/wanda). The score of weight ``W_ij`` is ``|W_ij| * ||X_j||_2``, the weight magnitude
times the norm of input feature ``j`` over the calibration tokens, and the lowest-score weights of every output row
are removed. No weight update follows.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .masks import SparsityPattern, nm_mask, unstructured_mask

__all__ = ["magnitude_mask", "wanda_mask", "wanda_score"]


def wanda_score(weight: Tensor, input_sq_norm: Tensor) -> Tensor:
    """``|W| * sqrt(sum_t x_tj^2)``. ``input_sq_norm`` may be scaled by any positive constant (the reference divides
    by the number of sequences), which leaves every row ranking unchanged."""
    return weight.float().abs() * torch.sqrt(input_sq_norm.float().to(weight.device)).reshape(1, -1)


def _mask(score: Tensor, pattern: SparsityPattern, *, per_row: bool) -> Tensor:
    if pattern.structured:
        assert pattern.n is not None and pattern.m is not None
        return nm_mask(score, pattern.n, pattern.m)
    return unstructured_mask(score, pattern.fraction, per_row=per_row)


def wanda_mask(weight: Tensor, input_sq_norm: Tensor, pattern: SparsityPattern) -> Tensor:
    """Wanda's mask: per output row (unstructured) or per group of ``M`` (N:M), lowest ``|W| * ||X||`` removed."""
    return _mask(wanda_score(weight, input_sq_norm), pattern, per_row=True)


def magnitude_mask(weight: Tensor, pattern: SparsityPattern) -> Tensor:
    """The magnitude baseline of the Wanda repository: lowest ``|W|`` over the whole matrix (unstructured) or per
    group of ``M`` (N:M)."""
    return _mask(weight.float().abs(), pattern, per_row=False)
