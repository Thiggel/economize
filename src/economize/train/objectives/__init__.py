"""The built-in objectives and the registries they use.

Importing this package registers the technique-agnostic objectives ``distill_kl``, ``step_distill``, ``ce`` and
``logit_distill`` in :data:`~economize.train.OBJECTIVES`, the divergences ``kl``, ``rkl``, ``js``, ``tvd`` and
``ce_top1`` in :data:`DIVERGENCES` and the state losses ``nmse`` and ``cos`` in :data:`STATE_LOSSES`.
Technique-specific objectives (``layerskip``, ``early_exit_kl``, ``eagle3_ttt``, ``medusa_ce``, ...) live in their
technique's subpackage under :mod:`economize.techniques` and register there. :mod:`.common` holds the pieces the
distillation objectives share (:func:`distill_term`, :func:`decayed_mean`, :func:`checkpointed_chunks`,
:func:`find_drafter`), for new objectives to reuse.
"""

from __future__ import annotations

from .base import BaseParameters, base_parameter_groups
from .ce import CrossEntropy, next_token_nll
from .common import (
    STATE_LOSSES,
    DistillParts,
    StateLoss,
    check_loss_weights,
    checkpointed_chunks,
    decayed_mean,
    distill_parts,
    distill_term,
    find_drafter,
    normalized_mse,
)
from .distill import DistillKL, TrainableDrafter
from .divergences import DIVERGENCES, Divergence, masked_mean, token_nll
from .logit import LogitDistill
from .steps import STEP_CURVE_METRICS, StepDistill, StepDrafter
from .weights import rollout_starts, target_weights

__all__ = [
    "DIVERGENCES",
    "STATE_LOSSES",
    "STEP_CURVE_METRICS",
    "BaseParameters",
    "CrossEntropy",
    "DistillKL",
    "DistillParts",
    "Divergence",
    "LogitDistill",
    "StateLoss",
    "StepDistill",
    "StepDrafter",
    "TrainableDrafter",
    "base_parameter_groups",
    "check_loss_weights",
    "checkpointed_chunks",
    "decayed_mean",
    "distill_parts",
    "distill_term",
    "find_drafter",
    "masked_mean",
    "next_token_nll",
    "normalized_mse",
    "rollout_starts",
    "target_weights",
    "token_nll",
]
