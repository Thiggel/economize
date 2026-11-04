"""Principal components of residual-stream activations, as the official SliceGPT ``pca_calc``."""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["CovarianceAccumulator", "principal_basis"]

#: Relative ridge added to the diagonal before the eigendecomposition (``pca_calc``'s ``damp = 0.01 * mean(diag)``).
DAMPING = 0.01


class CovarianceAccumulator:
    """Sums ``X^T X`` in float64 over calibration activations ``X`` of shape ``[..., width]``.

    Args:
        width: The channel count.
        normalize: Divide each batch's ``X^T X`` by its squared Frobenius norm first, so that residual points with a
            larger scale (the stream grows with depth) do not dominate a basis shared by several points.
    """

    def __init__(self, width: int, *, normalize: bool = False) -> None:
        self.total = torch.zeros(width, width, dtype=torch.float64)
        self.normalize = normalize
        self.count = 0

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        """Add the activations ``x`` (any leading shape)."""
        flat = x.reshape(-1, x.shape[-1]).double()
        gram = flat.T @ flat
        if self.normalize:
            gram = gram / gram.trace().clamp_min(torch.finfo(torch.float64).tiny)
        self.total += gram.cpu()
        self.count += 1

    def basis(self) -> Tensor:
        """The principal basis of everything added so far (see :func:`principal_basis`)."""
        if self.count == 0:
            raise RuntimeError("no activations were accumulated")
        return principal_basis(self.total)


def principal_basis(gram: Tensor) -> Tensor:
    """Orthogonal ``[width, width]`` float64 matrix whose columns are the eigenvectors of ``gram``, by decreasing
    eigenvalue, after adding ``DAMPING * mean(diag(gram))`` to the diagonal.

    Args:
        gram: Symmetric ``X^T X`` (or a sum of such matrices).
    """
    gram = gram.double().clone()
    gram.diagonal().add_(DAMPING * gram.diagonal().mean())
    values, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(values, descending=True)
    return vectors[:, order]
