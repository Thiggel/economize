"""The damped inverse Hessian shared by SparseGPT and GPTQ (Optimal Brain Surgeon updates in column order)."""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["damped_inverse_cholesky", "fix_dead_features", "inverse_hessian_cholesky"]


def fix_dead_features(hessian: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
    """Float32 copy of ``H`` with ``H_jj = 1`` for input features that never fire; their weight columns are zeroed.

    Returns the fixed Hessian and the boolean mask of dead features. ``weight`` (``[out, in]`` float32) is modified
    in place, as in the reference code.
    """
    H = hessian.to(device=weight.device, dtype=torch.float32).clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    weight[:, dead] = 0
    return H, dead


def damped_inverse_cholesky(H: Tensor, percdamp: float) -> Tensor:
    """Upper Cholesky factor of ``(H + percdamp * mean(diag(H)) * I)^{-1}`` (``H`` is modified in place)."""
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[0], device=H.device)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    return torch.linalg.cholesky(H, upper=True)


def inverse_hessian_cholesky(hessian: Tensor, weight: Tensor, percdamp: float) -> tuple[Tensor, Tensor]:
    """Prepare ``H = X X^T`` as in the reference GPTQ and SparseGPT code.

    Input features that never fire (``H_jj = 0``) get ``H_jj = 1`` and their weight column is zeroed in place. The
    diagonal is damped by ``percdamp * mean(diag(H))``. Returns the upper Cholesky factor of ``H^{-1}`` (whose rows
    drive the column-by-column error compensation) and the boolean mask of dead features.

    Args:
        hessian: ``[in, in]`` sum of ``x x^T`` over calibration tokens (any positive scale).
        weight: ``[out, in]`` float32 weight, modified in place for dead features.
        percdamp: Relative diagonal damping (0.01 in both papers).
    """
    H, dead = fix_dead_features(hessian, weight)
    return damped_inverse_cholesky(H, percdamp), dead
