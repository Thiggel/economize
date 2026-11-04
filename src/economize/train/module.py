"""The objective as a module: teacher pass and loss in one ``forward``, the unit that the loops (and data-parallel
wrappers such as DDP) call."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from ..errors import ConfigError
from .data.masks import loss_mask_from_labels
from .objective import Batch, Objective, ParamGroup, trainable
from .teacher import TeacherHead, teacher_pass

if TYPE_CHECKING:
    from ..core.model import EfficientModel

__all__ = ["ObjectiveModule"]


class ObjectiveModule(nn.Module):
    """Wraps an :class:`~economize.EfficientModel` and an :class:`~economize.train.Objective`.

    ``forward(input_ids, attention_mask=None, loss_mask=None)`` runs the teacher pass the objective requests
    (without gradients, and outside any autocast region the caller opened) and returns
    ``{"loss": ..., "metrics": {...}}``. Because the loss is computed inside ``forward``, a
    ``DistributedDataParallel`` wrapper around this module synchronizes the gradients of the trained parameters.

    Construction calls :meth:`Objective.setup`, collects the objective's parameter groups and freezes every other
    parameter of the model; :meth:`close` undoes both. ``train(mode)`` switches the techniques, and the base model
    only when the objective put it in training mode during setup (objectives that train the base model); a drafter's
    teacher always runs in eval mode.

    Args:
        model: The model.
        objective: The objective.
        teacher_head: Replaces the teacher's final states and head after each teacher pass (a larger teacher).
    """

    #: ``transformers.Trainer`` divides the loss by the accumulation steps itself when this is ``False``.
    accepts_loss_kwargs = False

    def __init__(self, model: EfficientModel, objective: Objective, teacher_head: TeacherHead | None = None) -> None:
        super().__init__()
        self.model = model
        self.objective = objective
        self.teacher_head = teacher_head
        self._modes = (model.base_model.training, model.techniques.training)
        model.eval()
        model.techniques.train()
        objective.setup(model)
        #: modules the objective trains outside the model (:meth:`Objective.modules`), registered here for DDP
        self.owned = nn.ModuleList(objective.modules(model))
        self._train_base = model.base_model.training
        self.groups: list[ParamGroup] = [g for g in objective.parameter_groups(model) if g.params]
        if not self.groups:
            raise ConfigError(f"objective {objective.name!r} returned no parameters to train")
        self.params = trainable(self.groups)
        self._requires_grad = {id(p): p.requires_grad for p in model.parameters()}
        for p in model.parameters():
            p.requires_grad_(False)
        for p in self.params:
            p.requires_grad_(True)
        self.request = objective.teacher_request(model)
        self._closed = False

    @property
    def trains_base(self) -> bool:
        """Whether some trained parameter belongs to the base model."""
        base = {id(p) for p in self.model.base_model.parameters()}
        return any(id(p) in base for p in self.params)

    def train(self, mode: bool = True) -> ObjectiveModule:
        self.training = mode
        self.model.techniques.train(mode)
        self.model.base_model.train(mode and self._train_base)
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_loss: bool = True,
    ) -> dict[str, Any]:
        """Teacher pass and loss of one batch. ``labels`` (``-100`` = ignored, Hugging Face convention) stand in
        for ``loss_mask`` when it is not given; their values are otherwise unused (the targets are the inputs)."""
        if loss_mask is None and labels is not None:
            loss_mask = loss_mask_from_labels(labels)
        batch = Batch(input_ids, attention_mask, loss_mask).to(self.model.adapter.device)
        teacher = None
        if self.request is not None:
            # the loops call forward inside their autocast region; the teacher runs in the model's own dtypes
            with torch.no_grad(), torch.autocast(self.model.adapter.device.type, enabled=False):
                teacher = teacher_pass(self.model, batch, self.request, self.teacher_head)
        output = self.objective.loss(self.model, batch, teacher)
        return {"loss": output.loss, "metrics": output.metrics}

    @property
    def hf_device_map(self) -> dict[str, Any]:
        """The base model's ``hf_device_map`` (``transformers.Trainer`` leaves models spread over devices in place).
        Absent, as on the base model, when the model was loaded without a device map: ``accelerate`` tests it with
        ``hasattr``."""
        device_map = getattr(self.model.base_model, "hf_device_map", None)
        if device_map is None:
            raise AttributeError("hf_device_map")
        return device_map

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        """Enable gradient checkpointing in the base model (``TrainingArguments(gradient_checkpointing=True)``)."""
        cast(Any, self.model.base_model).gradient_checkpointing_enable(**kwargs)  # an HF PreTrainedModel

    def close(self) -> None:
        """Call :meth:`Objective.teardown`, restore the parameters' ``requires_grad`` flags and the model's modes."""
        if self._closed:
            return
        self._closed = True
        self.objective.teardown(self.model)
        for p in self.model.parameters():
            p.requires_grad_(self._requires_grad.get(id(p), p.requires_grad))
        self.model.base_model.train(self._modes[0])
        self.model.techniques.train(self._modes[1])
