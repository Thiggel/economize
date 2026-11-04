"""Divergences between a teacher's and a drafter's next-token distributions, per token.

Each entry of :data:`DIVERGENCES` maps float32 log-probabilities ``teacher`` and ``student`` ``[..., V]`` to one value
per token ``[...]``; objectives reduce them with :func:`masked_mean`. Log-probabilities may be ``-inf`` (tokens a
draft vocabulary or a top-k head masks): where a distribution weights a log-ratio, its log-probabilities are clamped
to the dtype's smallest finite value first, so that a zero weight gives zero instead of ``0 * -inf = NaN`` in the
value and in the gradients. Finite inputs keep their values. A new divergence is added by registration::

    @DIVERGENCES.register("alpha")
    def alpha_divergence(teacher, student):
        ...
"""

from __future__ import annotations

import math
from typing import Protocol

import torch

from ...registry import Registry

__all__ = ["DIVERGENCES", "Divergence", "masked_mean", "safe_log", "token_nll"]


class Divergence(Protocol):
    def __call__(self, teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
        """Per-token divergence ``[...]`` from log-probabilities ``[..., V]`` (gradients flow through ``student``)."""
        ...


#: Divergences by name (entry-point group ``economize.divergences``).
DIVERGENCES: Registry[Divergence] = Registry("divergence", "economize.divergences")


def safe_log(log_probs: torch.Tensor) -> torch.Tensor:
    """``log_probs`` with ``-inf`` replaced by the dtype's smallest finite value (no gradient flows into those
    entries; every finite entry and its gradient is unchanged)."""
    return log_probs.clamp_min(torch.finfo(log_probs.dtype).min)


@DIVERGENCES.register("kl")
def forward_kl(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    """``KL(teacher || student)``, the usual distillation loss."""
    teacher = safe_log(teacher)
    return (teacher.exp() * (teacher - student)).sum(-1)


@DIVERGENCES.register("rkl")
def reverse_kl(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    """``KL(student || teacher)`` (mode seeking)."""
    student = safe_log(student)
    return (student.exp() * (student - teacher)).sum(-1)


@DIVERGENCES.register("js")
def jensen_shannon(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence."""
    teacher, student = safe_log(teacher), safe_log(student)
    mix = torch.logaddexp(teacher, student) - math.log(2)
    return 0.5 * ((teacher.exp() * (teacher - mix)).sum(-1) + (student.exp() * (student - mix)).sum(-1))


@DIVERGENCES.register("tvd")
def total_variation(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    """Total variation distance, one minus the acceptance probability of speculative sampling."""
    return 0.5 * (teacher.exp() - student.exp()).abs().sum(-1)


@DIVERGENCES.register("ce_top1")
def greedy_cross_entropy(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of the student on the teacher's most likely token (the target of greedy acceptance)."""
    return -student.gather(-1, teacher.argmax(-1, keepdim=True))[..., 0]


def masked_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    """Mean of ``values``, or their ``weights``-weighted mean (``0`` when every weight is zero)."""
    if weights is None:
        return values.mean()
    weights = weights.to(values.device, values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def token_nll(log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood ``[...]`` of the ``targets`` ``[...]`` under ``log_probs`` ``[..., V]``."""
    return -log_probs.gather(-1, targets.to(log_probs.device)[..., None])[..., 0]
