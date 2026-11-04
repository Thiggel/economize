"""``distill_kl``: distill the target's next-token distributions into any drafter that exposes train-time logits.

A drafter opts in by implementing :class:`TrainableDrafter` on its technique: it states what it needs from the
teacher pass and returns, for each draft step ``k``, logits whose position ``t`` is its distribution of token
``t + k`` given the exact states up to ``t`` and the true tokens in between (teacher forcing). Medusa heads,
multi-token prediction heads, EAGLE-style heads trained with teacher forcing and early-exit readouts all fit this
form. The loss at step ``k`` compares position ``t`` with the teacher's distribution at ``t + k - 1``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import torch

from ...errors import ConfigError
from ...validation import check_registered
from ..objective import (
    Batch,
    Objective,
    ObjectiveOutput,
    ParamGroup,
    TeacherOutputs,
    TeacherRequest,
    TrainRequestProvider,
    register_objective,
    require_teacher,
)
from .common import decayed_mean, distill_term, find_drafter
from .divergences import DIVERGENCES
from .weights import target_weights

if TYPE_CHECKING:
    from ...core.model import EfficientModel
    from ...core.technique import Technique

__all__ = ["DistillKL", "TrainableDrafter"]


@runtime_checkable
class TrainableDrafter(TrainRequestProvider, Protocol):
    """Implemented by techniques whose drafts can be scored against the teacher in one teacher-forced pass; their
    ``train_request`` states what :meth:`train_logits` needs from the teacher pass (e.g. the states its heads read).
    """

    def train_logits(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs) -> Mapping[int, torch.Tensor]:
        """Logits ``[B, T, V]`` by draft step ``k >= 1``: position ``t`` is the drafter's distribution of token
        ``t + k`` (positions ``t > T - 1 - k`` are ignored). Gradients must reach the drafter's parameters."""
        ...


@register_objective("distill_kl")
@dataclass
class DistillKL(Objective):
    """KL distillation of a :class:`TrainableDrafter`, averaged over positions and weighted over draft steps.

    Attributes:
        technique: Name of the drafting technique (``None``: the attached technique that implements
            :class:`TrainableDrafter`).
        divergence: A name in :data:`~economize.train.objectives.DIVERGENCES`.
        ce_weight: Weight of the cross-entropy on the true token ``t + k``.
        step_decay: Draft step ``k`` has weight ``step_decay ** (k - 1)``.
    """

    technique: str | None = None
    divergence: str = "kl"
    ce_weight: float = 0.0
    step_decay: float = 1.0

    def __post_init__(self) -> None:
        check_registered("divergence", self.divergence, DIVERGENCES)

    def drafter(self, model: EfficientModel) -> Technique:
        """The trained technique."""
        return find_drafter(model, self.technique, TrainableDrafter,
                            "distill_kl", "train_request/train_logits")

    def parameter_groups(self, model: EfficientModel) -> list[ParamGroup]:
        technique = self.drafter(model)
        return [ParamGroup(technique.name, [p for p in technique.parameters() if p.requires_grad])]

    def teacher_request(self, model: EfficientModel) -> TeacherRequest:
        technique = cast(TrainableDrafter, self.drafter(model))
        own = technique.train_request(model)
        return own.merge(TeacherRequest(hidden_layers=(model.adapter.num_layers,)))

    def loss(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs | None) -> ObjectiveOutput:
        teacher = require_teacher(teacher, type(self).__name__)
        technique = cast(TrainableDrafter, self.drafter(model))
        logits = technique.train_logits(model, batch, teacher)
        divergence = DIVERGENCES.get(self.divergence)
        length = batch.input_ids.shape[1]
        losses, metrics = [], {}
        steps = sorted(k for k in logits if k < length)
        for k in steps:
            student = logits[k][:, : length - k].float().log_softmax(-1)
            value, div = distill_term(teacher.log_probs(slice(None), slice(k - 1, length - 1)), student,
                                      batch.input_ids[:, k:], divergence, self.ce_weight,
                                      target_weights(batch, slice(k, None)))
            metrics[f"{self.divergence}{k}"] = div.detach()
            losses.append(value)
        if not losses:
            raise ConfigError("the drafter returned no draft step shorter than the sequence")
        loss = decayed_mean(losses, self.step_decay, [k - 1 for k in steps])
        return ObjectiveOutput(loss, metrics)
