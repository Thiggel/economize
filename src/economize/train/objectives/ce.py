"""``ce``: next-token cross-entropy of the model on the data (fine-tuning, recovery training after pruning or
quantization, PEFT adapters)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from ..objective import Batch, Objective, ObjectiveOutput, OptimDefaults, ParamGroup, TeacherOutputs, register_objective
from .base import BaseParameters, base_parameter_groups
from .divergences import masked_mean, token_nll
from .weights import target_weights

if TYPE_CHECKING:
    from ...core.model import EfficientModel

__all__ = ["CrossEntropy", "next_token_nll"]


def next_token_nll(logits: torch.Tensor, batch: Batch) -> torch.Tensor:
    """Mean negative log-likelihood of the next tokens of ``batch`` under ``logits`` ``[B, T, V]`` (padding and
    unmarked positions left out)."""
    ids = batch.input_ids
    log_probs = logits[:, :-1].float().log_softmax(-1)
    return masked_mean(token_nll(log_probs, ids[:, 1:]), target_weights(batch, slice(1, None)))


@register_objective("ce")
@dataclass
class CrossEntropy(Objective):
    """Next-token cross-entropy of the model's forward pass (with every attached technique active).

    Attributes:
        parameters: Which base parameters are trained (see :data:`~economize.train.objectives.base.BaseParameters`);
            the techniques' own trainable parameters are trained as well.
    """

    defaults: ClassVar[OptimDefaults] = OptimDefaults(lr=1e-4, warmup_steps=100)
    parameters: BaseParameters = "trainable"

    def parameter_groups(self, model: EfficientModel) -> list[ParamGroup]:
        return base_parameter_groups(model, self.parameters)

    def teacher_request(self, model: EfficientModel) -> None:
        return None

    def loss(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs | None) -> ObjectiveOutput:
        out = model.base_model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
        nll = next_token_nll(out.logits, batch)
        return ObjectiveOutput(nll, {"nll": nll.detach()})
