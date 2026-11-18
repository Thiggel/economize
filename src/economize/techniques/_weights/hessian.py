"""The damped inverse Hessian shared by SparseGPT and GPTQ (Optimal Brain Surgeon updates in column order)."""

from __future__ import annotations

import warnings

import torch
from torch import Tensor

__all__ = ["DAMP_ESCALATION", "damped_inverse_cholesky", "fix_dead_features", "inverse_hessian_cholesky"]


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


#: Multipliers of ``max(percdamp, 0.01)`` tried in turn when the Cholesky factorization of the Hessian damped by the
#: requested ``percdamp`` fails.
DAMP_ESCALATION = (10.0, 100.0, 1000.0)


def damped_inverse_cholesky(H: Tensor, percdamp: float) -> Tensor:
    """Upper Cholesky factor of ``(H + percdamp * mean(diag(H)) * I)^{-1}`` (``H`` is modified in place).

    The reference code factors once and fails on a Hessian that is not numerically positive definite (rank-deficient
    calibration data in float32). Here a failed factorization is retried with the relative damping raised to
    ``max(percdamp, 0.01)`` times each factor of :data:`DAMP_ESCALATION`, with a warning naming the damping that
    worked.

    Raises:
        torch.linalg.LinAlgError: The Hessian is not positive definite even at the largest damping.
    """
    mean = torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[0], device=H.device)
    damps = [percdamp] + [max(percdamp, 0.01) * factor for factor in DAMP_ESCALATION]
    applied = 0.0
    for relative in damps:
        H[diag, diag] += relative * mean - applied
        applied = float(relative * mean)
        lower, info = torch.linalg.cholesky_ex(H)
        if int(info) != 0:
            continue
        upper, info = torch.linalg.cholesky_ex(torch.cholesky_inverse(lower), upper=True)
        if int(info) != 0:
            continue
        if relative != percdamp:
            warnings.warn(
                f"the damped Hessian was not positive definite at percdamp={percdamp:g}; factored it with "
                f"percdamp={relative:g}",
                RuntimeWarning,
                stacklevel=2,
            )
        return upper
    raise torch.linalg.LinAlgError(
        f"the Hessian is not positive definite even with percdamp={damps[-1]:g}; use more calibration data or a "
        "larger percdamp"
    )


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
