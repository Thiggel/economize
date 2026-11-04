"""``logit_distill``: distill a larger teacher's next-token distributions into the whole model (Minitron's retraining
of a pruned model, Muralidharan et al., NeurIPS 2024; Sreenivas et al., 2024).

The student is the model's ordinary forward pass with every attached technique active (a width- or depth-pruned
base, a recovery LoRA, ...); the teacher is the ``teacher_model`` given to :func:`~economize.train.fit`, typically
the unpruned original. Minitron trains on the forward KL of the logits alone, which is the default here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ...errors import ConfigError
from ...validation import check_number, check_registered
from ..objective import (
    Batch,
    Objective,
    ObjectiveOutput,
    OptimDefaults,
    ParamGroup,
    TeacherOutputs,
    TeacherRequest,
    register_objective,
    require_teacher,
)
from .base import BaseParameters, base_parameter_groups
from .common import distill_term
from .divergences import DIVERGENCES
from .weights import target_weights

if TYPE_CHECKING:
    from ...core.model import EfficientModel

__all__ = ["LogitDistill"]


@register_objective("logit_distill")
@dataclass
class LogitDistill(Objective):
    """Divergence between the teacher's and the model's next-token distributions at every position.

    The teacher pass of the trained model alone would distill the model into itself, so a ``teacher_model`` is
    required: :meth:`loss` raises when the teacher's distributions are those of the trained model.

    Attributes:
        divergence: A name in :data:`~economize.train.objectives.DIVERGENCES` (Minitron: ``"kl"``, the forward KL).
        ce_weight: Weight of the next-token cross-entropy on the data (Minitron: ``0``).
        parameters: Which base parameters are trained (see :data:`~economize.train.objectives.base.BaseParameters`);
            the techniques' own trainable parameters are trained as well.
    """

    #: Minitron's Llama-3.1 retraining: peak 1e-4, cosine decay to 1e-5.
    defaults: ClassVar[OptimDefaults] = OptimDefaults(lr=1e-4, warmup_steps=100, min_lr_ratio=0.1)
    divergence: str = "kl"
    ce_weight: float = 0.0
    parameters: BaseParameters = "trainable"

    def __post_init__(self) -> None:
        check_registered("divergence", self.divergence, DIVERGENCES)
        check_number("ce_weight", self.ce_weight, low=0)

    def parameter_groups(self, model: EfficientModel) -> list[ParamGroup]:
        return base_parameter_groups(model, self.parameters)

    def teacher_request(self, model: EfficientModel) -> TeacherRequest:
        # only the teacher model's distributions are read: its pass replaces the trained model's own one
        return TeacherRequest(hidden_layers=(model.adapter.num_layers,), own_pass=False)

    def loss(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs | None) -> ObjectiveOutput:
        teacher = require_teacher(teacher, type(self).__name__)
        if teacher.final is teacher.hidden.get(model.adapter.num_layers):
            raise ConfigError("logit_distill distills a teacher_model into the trained model; pass "
                              "teacher_model=<the larger or unpruned model> to fit()")
        ids = batch.input_ids
        length = ids.shape[1]
        out = model.base_model(input_ids=ids, attention_mask=batch.attention_mask, use_cache=False)
        student = out.logits[:, :-1].float().log_softmax(-1)
        target = teacher.log_probs(slice(None), slice(0, length - 1)).to(student.device)
        value, div = distill_term(target, student, ids[:, 1:].to(student.device), DIVERGENCES.get(self.divergence),
                                  self.ce_weight, target_weights(batch, slice(1, None)))
        return ObjectiveOutput(value, {self.divergence: div.detach()})
