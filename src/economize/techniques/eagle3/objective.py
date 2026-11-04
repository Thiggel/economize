"""``eagle3_ttt``: train an EAGLE-3 draft layer with the training-time test (Li et al., 2025, section 3.2).

The objective unrolls the draft layer for ``steps`` steps on teacher-forced sequences
(:meth:`~economize.techniques.eagle3.Eagle3Technique.rollout`): step ``i`` at position ``t`` drafts token
``t + 1 + i`` from the exact states up to ``t`` and the layer's own outputs of steps ``1..i - 1``. Each step is
scored against the teacher's distribution of the same token::

    loss = sum_i decay**(i-1) divergence(teacher_i || draft_i) / sum_i decay**(i-1)

With a draft vocabulary (``Eagle3Config.draft_vocab``) the teacher's distribution is restricted to it and
renormalized, and positions whose teacher argmax lies outside it do not count (the SpecForge recipe). Without one the
loss equals ``distill_kl`` with ``step_decay`` on the same rollout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from ...errors import ConfigError
from ...train.objective import (
    Batch,
    ObjectiveOutput,
    OptimDefaults,
    TeacherOutputs,
    TechniqueObjective,
    register_objective,
    require_teacher,
)
from ...train.objectives.common import decayed_mean
from ...train.objectives.divergences import DIVERGENCES, masked_mean
from ...train.objectives.weights import target_weights
from ...validation import check_number, check_positive, check_registered
from .technique import Eagle3Technique

if TYPE_CHECKING:
    from ...core.model import EfficientModel

__all__ = ["Eagle3TTT"]


@register_objective("eagle3_ttt", default_for=(Eagle3Technique.name,))
@dataclass
class Eagle3TTT(TechniqueObjective[Eagle3Technique]):
    """Training-time-test distillation of an :class:`~economize.techniques.eagle3.Eagle3Technique`.

    Attributes:
        technique: Name of the attached EAGLE-3 technique.
        steps: Unrolled draft steps (``None``: the technique's ``train_steps``, 7 by default).
        step_decay: Step ``i`` has weight ``step_decay ** (i - 1)`` (EAGLE-3 and SpecForge use ``0.8``).
        divergence: A name in :data:`~economize.train.objectives.DIVERGENCES` (the paper's forward KL by default).
    """

    technique: str = Eagle3Technique.name
    steps: int | None = None
    step_decay: float = 0.8
    divergence: str = "kl"

    #: SpecForge's recipe for EAGLE-3: AdamW at 1e-4 without weight decay, gradients clipped to 0.5.
    defaults: ClassVar[OptimDefaults] = OptimDefaults(lr=1e-4, weight_decay=0.0, grad_clip=0.5)
    group: ClassVar[str] = Eagle3Technique.name

    def __post_init__(self) -> None:
        check_registered("divergence", self.divergence, DIVERGENCES)
        if self.steps is not None:
            check_positive("steps", self.steps)
        check_number("step_decay", self.step_decay, low=0, closed_low=False)

    def loss(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs | None) -> ObjectiveOutput:
        teacher = require_teacher(teacher, type(self).__name__)
        technique = self.trained(model)
        length = batch.input_ids.shape[1]
        steps = min(self.steps or technique.config.train_steps, length - 2)
        if steps < 1:
            raise ConfigError(f"sequences of {length} tokens are too short for the training-time test (need >= 3)")
        ids = technique.draft_ids
        restricted = technique.draft_vocab_size < technique.vocab_size
        divergence = DIVERGENCES.get(self.divergence)
        terms, metrics = [], {}
        for i, out in enumerate(technique.rollout(teacher.hidden, steps), start=1):
            k = i + 1  # step i at position t drafts token t + k
            student = technique.head_logits(out[:, : length - k]).float().log_softmax(-1)
            reference = teacher.log_probs(slice(None), slice(k - 1, length - 1)).to(student.device)
            mask = target_weights(batch, slice(k, None))
            if mask is not None:
                mask = mask.to(student.device)
            if ids is not None:
                if restricted:
                    inside = technique.t2d[reference.argmax(-1)]
                    mask = inside if mask is None else mask & inside
                reference = reference.index_select(-1, ids).log_softmax(-1)
            value = masked_mean(divergence(reference, student), mask)
            terms.append(value)
            metrics[f"{self.divergence}{i}"] = value.detach()
            with torch.no_grad():
                metrics[f"acc{i}"] = masked_mean((student.argmax(-1) == reference.argmax(-1)).float(), mask)
        loss = decayed_mean(terms, self.step_decay)
        return ObjectiveOutput(loss, metrics)
