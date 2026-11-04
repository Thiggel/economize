"""``hass``: train an EAGLE-3 draft layer with HASS (Zhang et al., ICLR 2025, "Learning Harmonized Representations
for Speculative Sampling").

HASS keeps the drafter and changes its training in two ways:

* Harmonized objective distillation: the loss of every draft step adds a top-K term, the cross-entropy of the
  drafter on the teacher's ``top_k`` most likely tokens weighted by their (unnormalized) teacher probabilities,
  ``-sum_{y in top-K(p)} p(y) log q(y)``. Greedy acceptance only asks the drafter to rank those tokens right, and
  the term puts more weight on them than the full KL does.
* Harmonized context alignment: training unrolls ``steps`` draft steps on teacher-forced sequences (EAGLE-3's
  training-time test), and each step reads the earlier steps' outputs without their gradients
  (:meth:`~economize.techniques.eagle3.Eagle3Technique.rollout` with ``detach_steps``), all steps weighted equally.

Relative to EAGLE-3's training-time test (``eagle3`` with ``step_distill`` or ``eagle3_ttt``: 7 steps, step
weights ``0.8 ** (k - 1)``, gradients through all steps) HASS therefore adds the top-K term and changes the step
schedule (3 steps, equal weights, no gradient through the fed-back outputs). The drafter, its checkpoints and its
decoding are those of :class:`~economize.techniques.eagle3.Eagle3Technique`.

The reference implementation also regresses the draft features onto the model's last hidden states (smooth L1,
weight 1, and the soft cross-entropy at weight 0.1), as EAGLE-1/2 do. EAGLE-3 drops that term (its fused source
is not the last hidden state), and so does this objective: the full KL (whose gradient is that of the soft
cross-entropy) takes its place at weight 1, so that HASS and EAGLE-3 differ only in the two terms above.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import torch

from ...train.objective import Batch, TeacherOutputs, register_objective
from ...train.objectives.common import find_drafter
from ...train.objectives.divergences import safe_log
from ...train.objectives.steps import StepDistill, StepDrafter
from ...validation import check_number, check_positive

if TYPE_CHECKING:
    from ...core.model import EfficientModel
    from ...core.technique import Technique

__all__ = ["ContextAlignedDrafter", "HassDistill", "top_k_cross_entropy"]


@runtime_checkable
class ContextAlignedDrafter(StepDrafter, Protocol):
    """A step drafter HASS can train: it unrolls its draft steps on a teacher-forced sequence and can feed each
    step's output into the next one without its gradient (:class:`~economize.techniques.eagle3.Eagle3Technique`)."""

    @property
    def device(self) -> torch.device:
        """Where the drafter's parameters live."""
        ...

    def rollout(self, hidden: Mapping[int, torch.Tensor], steps: int, *,
                detach_steps: bool = False) -> list[torch.Tensor]:
        """The drafter's states ``[B, T, ...]`` of draft steps ``1..steps`` from the teacher's ``hidden`` states
        (one per step, each read by :meth:`StepDrafter.step_logits` after indexing the draft positions)."""
        ...


def top_k_cross_entropy(teacher: torch.Tensor, student: torch.Tensor, k: int) -> torch.Tensor:
    """``-sum_{y in top-k(teacher)} p(y) log q(y)`` per token ``[...]`` from log-probabilities ``[..., V]``.

    The teacher probabilities of the ``k`` tokens are not renormalized (HASS, ``train/main_hass.py``)."""
    top = teacher.topk(min(k, teacher.shape[-1]), dim=-1)
    return -(top.values.exp() * safe_log(student).gather(-1, top.indices)).sum(-1)


@register_objective("hass")
@dataclass
class HassDistill(StepDistill):
    """HASS distillation of an attached :class:`ContextAlignedDrafter` (EAGLE-3's draft layer).

    Training loss per draft step ``k = 1..steps``: ``divergence(teacher_k || draft_k) + top_k_weight *
    top_k_cross_entropy(teacher_k, draft_k, top_k)``, steps weighted by ``step_decay ** (k - 1)``. With a restricted
    draft vocabulary both terms use the teacher restricted to it and renormalized (as ``step_distill``). The
    evaluation curve is :meth:`StepDistill.evaluate`'s, unchanged.

    Attributes:
        steps: Draft steps of the context alignment (the paper's 3).
        step_decay: Step weights (``1.0``: equal, as the paper).
        top_k: Tokens of the top-K term (the paper's 10).
        top_k_weight: Weight of the top-K term (the paper's 1.0).
        detach_steps: Feed each step's output into the next step without its gradient (the paper's alignment;
            ``False`` gives EAGLE-3's training-time test plus the top-K term).

    The remaining attributes are those of :class:`~economize.train.objectives.steps.StepDistill`.
    """

    steps: int = 3
    top_k: int = 10
    top_k_weight: float = 1.0
    detach_steps: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        check_positive("top_k", self.top_k)
        check_number("top_k_weight", self.top_k_weight, low=0)

    def drafter(self, model: EfficientModel) -> Technique:
        """The trained technique (a :class:`ContextAlignedDrafter`)."""
        return find_drafter(model, self.technique, ContextAlignedDrafter, "hass",
                            "ContextAlignedDrafter (step_* and rollout(hidden, steps, detach_steps=...))")

    def step_divergence(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        base = super().step_divergence()
        if not self.top_k_weight:
            return base
        k, weight = self.top_k, self.top_k_weight

        def harmonized(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
            return base(teacher, student) + weight * top_k_cross_entropy(teacher, student, k)

        return harmonized

    def rollout_states(self, drafter: StepDrafter, model: EfficientModel, batch: Batch, teacher: TeacherOutputs,
                       starts: torch.Tensor, steps: int) -> Sequence[torch.Tensor]:
        aligned = cast(ContextAlignedDrafter, drafter)  # the technique drafter() found
        index = starts.to(aligned.device)
        return [out[:, index] for out in aligned.rollout(teacher.hidden, steps, detach_steps=self.detach_steps)]
