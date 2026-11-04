"""``medusa_ce``: train Medusa heads on a frozen model (Medusa-1).

Head ``j`` reads the model's last hidden state at position ``t`` and predicts token ``t + 1 + j``. Its loss is the
cross-entropy on that true token (Medusa-1) plus, optionally, a divergence from the teacher's distribution of the
same token (the self-distillation of Medusa-2; with ``normalize=True`` it matches the reference implementation in
``tests/oracle/test_drafter_medusa_oracle.py``). As in the paper (Cai et al., 2024, eq. 1), the per-head losses,
each averaged over the valid tokens, are summed with weight ``lambda_j = decay**j``::

    loss = sum_j decay**j [ce_weight * CE_j + kl_weight * divergence(teacher || head_j)]

``normalize=True`` divides by ``sum_j decay**j`` instead (a weighted mean, whose scale does not grow with the number
of heads).

The teacher pass provides the last hidden state of every position at once, so a batch trains all heads on all
positions without a rollout. :class:`~economize.train.objectives.DistillKL` trains the same heads through the
``TrainableDrafter`` protocol (pure distillation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

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
from ...train.objectives.common import check_loss_weights, decayed_mean, distill_parts
from ...train.objectives.divergences import DIVERGENCES, masked_mean
from ...train.objectives.weights import target_weights
from ...validation import check_number, check_registered
from .technique import MedusaTechnique

if TYPE_CHECKING:
    from ...core.model import EfficientModel

__all__ = ["MedusaCE"]


@register_objective("medusa_ce", default_for=(MedusaTechnique.name,))
@dataclass
class MedusaCE(TechniqueObjective[MedusaTechnique]):
    """Cross-entropy (and optional distillation) training of a :class:`~economize.techniques.medusa.MedusaTechnique`.

    Attributes:
        technique: Name of the attached Medusa technique.
        ce_weight: Weight of the cross-entropy on the true token.
        kl_weight: Weight of the divergence from the teacher (``0``: Medusa-1, true tokens only).
        divergence: A name in :data:`~economize.train.objectives.DIVERGENCES`.
        step_decay: Head ``j`` has weight ``step_decay ** j`` (Medusa's ``lambda_k = 0.8 ** k``).
        normalize: Divide the weighted sum by the sum of the weights (the paper sums).
    """

    technique: str = MedusaTechnique.name
    ce_weight: float = 1.0
    kl_weight: float = 0.0
    divergence: str = "kl"
    step_decay: float = 0.8
    normalize: bool = False

    #: Medusa-1's recipe: learning rate 1e-3 without weight decay, cosine schedule.
    defaults: ClassVar[OptimDefaults] = OptimDefaults(lr=1e-3, weight_decay=0.0)
    group: ClassVar[str] = MedusaTechnique.name

    def __post_init__(self) -> None:
        check_registered("divergence", self.divergence, DIVERGENCES)
        check_loss_weights(self.ce_weight, self.kl_weight)
        check_number("step_decay", self.step_decay, low=0, closed_low=False)

    def loss(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs | None) -> ObjectiveOutput:
        teacher = require_teacher(teacher, type(self).__name__)
        technique = self.trained(model)
        ids = batch.input_ids
        length = ids.shape[1]
        hidden = teacher.hidden[model.adapter.num_layers]
        terms, heads, metrics = [], [], {}
        for j in range(1, technique.num_heads + 1):
            shift = j + 1  # head j at position t predicts the token at t + shift
            if length <= shift:
                break
            student = technique.logits_of(hidden[:, : length - shift], j).log_softmax(-1)
            mask = target_weights(batch, slice(shift, None))
            targets = ids[:, shift:].to(student.device)
            reference = (teacher.log_probs(slice(None), slice(shift - 1, length - 1)).to(student.device)
                         if self.kl_weight else None)
            parts = distill_parts(student, mask, teacher=reference, divergence=DIVERGENCES.get(self.divergence),
                                  targets=targets, kl_weight=self.kl_weight, ce_weight=self.ce_weight)
            metrics.update(parts.metrics(f"ce{j}", f"{self.divergence}{j}"))
            with torch.no_grad():
                metrics[f"acc{j}"] = masked_mean((student.argmax(-1) == targets).float(), mask)
            terms.append(parts.loss)
            heads.append(j)
        if not terms:
            raise ConfigError(f"sequences of {length} tokens are too short to train Medusa head 1 (need >= 3)")
        if self.normalize:
            loss = decayed_mean(terms, self.step_decay, heads)
        else:
            loss = cast(torch.Tensor, sum(self.step_decay ** j * t for j, t in zip(heads, terms)))  # terms not empty
        return ObjectiveOutput(loss, metrics)
