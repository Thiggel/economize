"""``early_exit_kl``: distill the target's next-token distributions into an early-exit drafter.

Every position ``t`` of a batch is a draft step: layers ``1..r`` run on the true token at ``t`` and the exit's
distribution of token ``t + 1`` is compared with the teacher's. With ``adaptation="frozen"`` the draft path equals the
model's own layers, so the teacher's output of layer ``r`` serves every position at once and only the readout trains.
With LoRA or trainable copies, rollouts of ``horizon`` steps start at every ``stride``-th position: step ``k`` sees
the teacher's keys up to the start and the draft path's own keys after it, as in decoding (``horizon=1`` is the
plain one-step loss).

    loss = sum_k w_k * [divergence(teacher || exit) + ce_weight * CE(exit, true next token)] / sum_k w_k,
    w_k = step_decay**(k-1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...errors import IncompatibleTechniquesError
from ...train.objective import (
    Batch,
    ObjectiveOutput,
    ParamGroup,
    TeacherOutputs,
    TeacherRequest,
    TechniqueObjective,
    register_objective,
    require_teacher,
    teacher_cache,
)
from ...train.objectives.common import decayed_mean, distill_term
from ...train.objectives.divergences import DIVERGENCES
from ...train.objectives.weights import target_weights
from ...validation import check_positive, check_registered
from .._drafting import BLOCK_LR, LORA_LR
from .technique import EarlyExitTechnique

if TYPE_CHECKING:
    from ...core.model import EfficientModel

__all__ = ["EarlyExitKL", "path_parameter_groups"]


def path_parameter_groups(technique: torch.nn.Module, path_name: str = "path") -> list[ParamGroup]:
    """Optimizer groups of a self-drafter: its own modules as ``head``, the draft path's LoRA factors and layer
    copies as ``backbone`` (the learning rates of the ``jump_rollout`` objective)."""
    path = getattr(technique, path_name)
    head = [p for name, p in technique.named_parameters() if name.split(".")[0] != path_name]
    groups = [ParamGroup("drafter", head, "head")] if head else []
    kinds = path.backbone_parameters()
    for kind, lr in (("lora", LORA_LR), ("blocks", BLOCK_LR)):
        if kind in kinds:
            groups.append(ParamGroup(kind, kinds[kind], "backbone", lr=lr, weight_decay=0.0))
    return groups


@register_objective("early_exit_kl", default_for=(EarlyExitTechnique.name,))
@dataclass
class EarlyExitKL(TechniqueObjective[EarlyExitTechnique]):
    """KL distillation of an :class:`~economize.techniques.early_exit.EarlyExitTechnique`.

    Attributes:
        technique: Name of the attached early-exit technique.
        horizon: Draft steps per rollout (adapted paths only; the frozen path has one step per position).
        stride: Distance between rollout starts (adapted paths only).
        divergence: A name in :data:`~economize.train.objectives.DIVERGENCES` (``"kl"`` = KL(teacher || exit)).
        ce_weight: Weight of the cross-entropy on the true next token.
        step_decay: Rollout step ``k`` has weight ``step_decay ** (k - 1)``.
    """

    technique: str = EarlyExitTechnique.name
    horizon: int = 1
    stride: int = 1
    divergence: str = "kl"
    ce_weight: float = 0.0
    step_decay: float = 1.0

    def __post_init__(self) -> None:
        check_registered("divergence", self.divergence, DIVERGENCES)
        check_positive("horizon", self.horizon)
        check_positive("stride", self.stride)

    def setup(self, model: EfficientModel) -> None:
        """Check that the technique is attached and, for an adapted path, that its layers can run in a rollout."""
        from ...train.rollout import check_rollout_layers

        technique = self.trained(model)
        if technique.config.adaptation != "frozen":
            check_rollout_layers(model, *technique.layer_span)

    def parameter_groups(self, model: EfficientModel) -> list[ParamGroup]:
        """The readout as ``head``, the draft path's LoRA factors or layer copies as ``backbone``."""
        groups = path_parameter_groups(self.trained(model))
        if not groups:
            raise IncompatibleTechniquesError("a frozen early exit with the shared head has no parameters to train; "
                                              "train the base model with the 'layerskip' objective instead")
        return groups

    def teacher_request(self, model: EfficientModel) -> TeacherRequest:
        """The technique's exact pass plus the last layer's states."""
        technique = self.trained(model)
        return technique.train_request(model).merge(TeacherRequest(hidden_layers=(model.adapter.num_layers,)))

    def _term(self, batch: Batch, teacher: TeacherOutputs, logits: torch.Tensor,
              positions: torch.Tensor | slice) -> torch.Tensor:
        """Loss of exit logits ``[B, S, V]`` at ``positions`` (their targets sit one position later)."""
        student = logits.float().log_softmax(-1)
        if isinstance(positions, slice):
            targets: torch.Tensor | slice = slice(positions.start + 1, positions.stop + 1)
        else:
            targets = positions + 1
        return distill_term(teacher.log_probs(slice(None), positions), student, batch.input_ids[:, targets],
                            DIVERGENCES.get(self.divergence), self.ce_weight, target_weights(batch, targets))[0]

    def loss(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs | None) -> ObjectiveOutput:
        teacher = require_teacher(teacher, type(self).__name__)
        technique = self.trained(model)
        ids = batch.input_ids
        length = ids.shape[1]
        r = technique.exit_layer
        if technique.config.adaptation == "frozen":
            logits = technique.logits(teacher.hidden[r][:, : length - 1])
            value = self._term(batch, teacher, logits, slice(0, length - 1))
            return ObjectiveOutput(value, {f"{self.divergence}1": value.detach()})
        teacher_kv = teacher_cache(teacher, type(self).__name__)
        from .rollout import self_draft_rollout

        offset = int(torch.randint(0, self.stride, ()).item()) if self.stride > 1 else 0
        starts = torch.arange(offset - 1, length - self.horizon - 1, self.stride, device=ids.device)

        def step(k: int, positions: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
            return self._term(batch, teacher, technique.logits(hidden), positions)

        losses = self_draft_rollout(model, lambda x, ctx: technique.run(model.runner, x, ctx), ids, teacher_kv,
                                    starts, self.horizon, step)
        loss = decayed_mean(losses, self.step_decay)
        return ObjectiveOutput(loss, {f"{self.divergence}1": losses[0].detach(),
                                      f"{self.divergence}K": losses[-1].detach()})
