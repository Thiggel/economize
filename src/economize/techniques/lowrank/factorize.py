"""Rank-``k`` factorizations ``W ~ up @ down`` of a weight ``W [out, in]``.

* ``"svd"``: truncated SVD of ``W`` (the data-free baseline). It minimizes ``||W - W'||_F``, which ignores the
  inputs and is known to collapse LLMs at moderate compression.
* ``"svd_llm"``: SVD-LLM (Wang et al., "SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language
  Model Compression", ICLR 2025). With the Cholesky factor ``S`` of ``X X^T`` (``S S^T = X X^T``), the truncated SVD
  of ``W S`` gives ``W' = (W S)_k S^{-1}``, which minimizes the output error ``||(W - W') X||_F`` exactly, and every
  dropped singular value equals the error it causes. This is the whitening step of the paper ("SVD-LLM (W)"); its
  optional LoRA-style parameter update is not part of this package.

A new factorization is one :class:`Factorizer` subclass registered in :data:`FACTORIZERS`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import ClassVar

import torch
from torch import Tensor

from ...errors import ConfigError
from ...registry import Registry
from ...validation import check_unit_interval

__all__ = [
    "FACTORIZERS",
    "Factorizer",
    "SVDFactorizer",
    "SVDLLMFactorizer",
    "rank_for_ratio",
    "svd_factors",
    "svd_llm_factors",
    "whitening",
]

logger = logging.getLogger(__name__)


def rank_for_ratio(out_features: int, in_features: int, compression: float, multiple: int = 1) -> int:
    """The rank whose two factors hold ``(1 - compression)`` of the ``out x in`` parameters (SVD-LLM's convention).

    ``k = floor((1 - compression) * out * in / (out + in))``, rounded down to a multiple of ``multiple`` (at least
    ``multiple``).
    """
    check_unit_interval("compression", compression, closed_low=False)
    k = int((1 - compression) * out_features * in_features / (out_features + in_features))
    return max(multiple, k // multiple * multiple)


def _split(u: Tensor, sigma: Tensor, vh: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    root = sigma[:rank].sqrt()
    return u[:, :rank] * root, root.unsqueeze(1) * vh[:rank]


def svd_factors(weight: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    """``(up [out, k], down [k, in])`` of the truncated SVD, with ``sqrt(sigma)`` split between the factors."""
    u, sigma, vh = torch.linalg.svd(weight.detach().double(), full_matrices=False)
    return _split(u, sigma, vh, rank)


def whitening(gram: Tensor, *, overwrite: bool = False) -> Tensor:
    """Lower Cholesky factor ``S`` of ``X X^T`` in float64.

    As in the reference, a matrix that is not positive definite is shifted by ``-lambda_min + 1e-6`` first. With
    ``overwrite`` a float64 ``gram`` is symmetrized (and shifted) in place, so the factor is the only matrix of its
    size allocated; at a 21504-wide MLP input that saves three 3.7 GB temporaries. The result is the same either way.
    """
    gram = gram.detach()
    if not (overwrite and gram.dtype == torch.float64):
        gram = gram.to(torch.float64, copy=True)
    _symmetrize_(gram)
    factor, info = torch.linalg.cholesky_ex(gram)
    if int(info) != 0:
        del factor
        shift = -torch.linalg.eigvalsh(gram)[0] + 1e-6
        gram.diagonal().add_(shift)  # gram + shift * I
        factor = torch.linalg.cholesky(gram)
    return factor


def _symmetrize_(matrix: Tensor, block: int = 1024) -> None:
    """``matrix = (matrix + matrix.T) / 2`` in place, bit for bit, ``block`` rows at a time.

    Floating-point addition commutes, so the result is exactly symmetric: each block of rows gets its lower part
    (up to the diagonal block) from the original entries and mirrors it into the upper part of the earlier rows, which
    later blocks no longer read.
    """
    size = matrix.shape[0]
    for start in range(0, size, block):
        end = min(start + block, size)
        matrix[start:end, :end] = (matrix[start:end, :end] + matrix[:end, start:end].T) / 2
        matrix[:start, start:end] = matrix[start:end, :start].T


def svd_llm_factors(weight: Tensor, gram: Tensor, rank: int, *, overwrite_gram: bool = False) -> tuple[Tensor, Tensor]:
    """``(up [out, k], down [k, in])`` with ``up @ down = (W S)_k S^{-1}`` for ``S S^T = gram`` (SVD-LLM).

    ``overwrite_gram``: ``gram`` may be overwritten (see :func:`whitening`).
    """
    s = whitening(gram, overwrite=overwrite_gram).to(weight.device)
    u, sigma, vh = torch.linalg.svd(weight.detach().double() @ s, full_matrices=False)
    up, down_white = _split(u, sigma, vh, rank)
    down = torch.linalg.solve_triangular(s, down_white, upper=False, left=False)  # down_white @ S^{-1}
    return up, down


class Factorizer(ABC):
    """A rank-``k`` factorization method."""

    #: Whether :meth:`factorize` needs the input Gram matrix ``X X^T`` (from calibration data).
    needs_calibration: ClassVar[bool] = True

    @abstractmethod
    def factorize(
        self, weight: Tensor, gram: Tensor | None, rank: int, *, overwrite_gram: bool = False
    ) -> tuple[Tensor, Tensor]:
        """``(up [out, k], down [k, in])`` in float64. With ``overwrite_gram`` the caller no longer needs ``gram``
        and the method may reuse its memory."""


FACTORIZERS: Registry[type[Factorizer]] = Registry("low-rank factorization", "economize.factorizers")


@FACTORIZERS.register("svd")
class SVDFactorizer(Factorizer):
    """Truncated SVD of the weight (data-free baseline)."""

    needs_calibration = False

    def factorize(
        self, weight: Tensor, gram: Tensor | None, rank: int, *, overwrite_gram: bool = False
    ) -> tuple[Tensor, Tensor]:
        return svd_factors(weight, rank)


@FACTORIZERS.register("svd_llm")
class SVDLLMFactorizer(Factorizer):
    """SVD-LLM truncation-aware whitening (Wang et al., ICLR 2025)."""

    def factorize(
        self, weight: Tensor, gram: Tensor | None, rank: int, *, overwrite_gram: bool = False
    ) -> tuple[Tensor, Tensor]:
        if gram is None:
            raise ConfigError("svd_llm needs the input Gram matrix from calibration data")
        return svd_llm_factors(weight, gram, rank, overwrite_gram=overwrite_gram)
