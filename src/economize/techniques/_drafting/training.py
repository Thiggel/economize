"""Defaults of the training protocols that drafters implement (``economize.train``'s ``StepDrafter``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ...core.model import EfficientModel
    from ...train.objective import TeacherRequest, TrainRequestProvider

__all__ = ["BLOCK_LR", "LORA_LR", "StepDrafterDefaults"]

#: Learning rate of LoRA factors on a draft path (the ``jump_rollout`` objective's ``group_lr["lora"]``).
LORA_LR = 2e-4
#: Learning rate of trainable copies of pretrained decoder layers on a draft path: pretrained weights move less than
#: new modules (``jump_rollout``'s ``group_lr["blocks"]``).
BLOCK_LR = 3e-5


class StepDrafterDefaults:
    """The common answers of :class:`~economize.train.objectives.steps.StepDrafter`: rollouts need what
    ``train_request`` asks of the teacher, score the whole vocabulary and may run any number of steps.

    A drafter technique lists this mixin before :class:`~economize.core.technique.Technique` and overrides only the
    answers that differ (EAGLE-3's draft vocabulary, Medusa's step limit).
    """

    def step_request(self: TrainRequestProvider, model: EfficientModel) -> TeacherRequest:
        """As ``train_request``."""
        return self.train_request(model)

    def step_vocab(self) -> torch.Tensor | None:
        """``None``: the drafter scores the model's whole vocabulary."""
        return None

    def step_limit(self) -> int | None:
        """``None``: no limit on the draft steps after an exact pass."""
        return None
