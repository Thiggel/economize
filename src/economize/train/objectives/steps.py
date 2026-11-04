"""``step_distill``: distill the teacher into any multi-step drafter on decode-faithful, teacher-forced draft steps.

A drafter opts in by implementing :class:`StepDrafter` on its technique. The objective picks rollout starts ``s``
(every ``stride``-th position) and asks the drafter for the states of draft steps ``k = 1..steps`` after an exact
pass that ended at ``s``: step ``k`` sits at position ``s + k`` (the true token there is its input) and drafts token
``s + k + 1``, which is the convention of :mod:`economize.train.rollout`. How a drafter computes the
states of step ``k`` is its own business: EAGLE-3 unrolls its draft layer, an MTP module feeds its own output back,
Medusa reads the last exact state with head ``k``, an early exit runs its draft path on the true tokens. Every step
is scored against the teacher's distribution of the same token::

    loss = sum_k decay**(k-1) [divergence(teacher_k || draft_k) + ce_weight * CE(draft_k, true token)]
           / sum_k decay**(k-1)

The logits of a step are computed in chunks of at most ``head_chunk`` tokens under activation checkpointing, so that
full-vocabulary drafters can be trained at every position of long sequences. A drafter with a restricted draft
vocabulary (:meth:`StepDrafter.step_vocab`) is compared with the teacher restricted to that vocabulary and
renormalized, and positions whose teacher argmax lies outside it do not count (the SpecForge recipe of EAGLE-3).

:meth:`StepDistill.evaluate` returns a teacher-forced curve (one value per draft step ``k = 1..eval_horizon``) that
the other drafter objectives also report, so that drafters of different families are compared on the same held-out
metric.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import torch

from ...errors import IncompatibleTechniquesError
from ...validation import check_non_negative, check_number, check_positive, check_registered
from ..objective import (
    Batch,
    Objective,
    ObjectiveOutput,
    ParamGroup,
    TeacherOutputs,
    TeacherRequest,
    register_objective,
    require_teacher,
)
from .common import checkpointed_chunks, decayed_mean, find_drafter
from .divergences import DIVERGENCES, forward_kl, masked_mean, token_nll
from .weights import rollout_starts, target_weights

if TYPE_CHECKING:
    from ...core.model import EfficientModel
    from ...core.technique import Technique

__all__ = ["STEP_CURVE_METRICS", "StepDistill", "StepDrafter"]

#: The metrics of :meth:`StepDistill.evaluate`: the per-step distribution metrics of the drafter curves, plus
#: ``outside`` (the teacher's probability mass outside a restricted draft vocabulary, ``0`` for full-vocabulary
#: drafters).
STEP_CURVE_METRICS = ("kl", "excess_nll", "agree", "teacher_nll", "run", "outside")


@runtime_checkable
class StepDrafter(Protocol):
    """Implemented by drafting techniques that can be rolled out step by step on teacher-forced sequences."""

    def step_request(self, model: EfficientModel) -> TeacherRequest:
        """What :meth:`step_states` needs from the teacher pass."""
        ...

    def step_states(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs, starts: torch.Tensor,
                    steps: int) -> Sequence[torch.Tensor]:
        """The inputs of the drafter's head for draft steps ``k = 1..steps`` after exact passes that ended at
        ``starts`` ``[S]``: entry ``k - 1`` is ``[B, S, ...]``, its column ``i`` drafts token ``starts[i] + k + 1``
        from the exact states up to ``starts[i]`` and the true tokens up to ``starts[i] + k``, exactly as in
        decoding. ``starts + steps + 1`` is below the sequence length. Gradients must reach the trained parameters.
        """
        ...

    def step_logits(self, states: torch.Tensor, step: int) -> torch.Tensor:
        """Float logits ``[B, S, V']`` of draft step ``step`` from (a slice along ``S`` of) its entry of
        :meth:`step_states`; ``V'`` is the size of :meth:`step_vocab`."""
        ...

    def step_vocab(self) -> torch.Tensor | None:
        """The model token ids of the columns of :meth:`step_logits` (sorted), or ``None`` for the model's whole
        vocabulary."""
        ...

    def step_limit(self) -> int | None:
        """The most draft steps the drafter makes after an exact pass (``None``: no limit), e.g. Medusa's heads."""
        ...

    def step_parameter_groups(self) -> list[ParamGroup]:
        """The trainable parameters in optimizer groups: new modules as ``"head"``, adapted or copied pretrained
        layers as ``"backbone"``."""
        ...


@register_objective("step_distill")
@dataclass
class StepDistill(Objective):
    """Divergence distillation of a :class:`StepDrafter` over ``steps`` decode-faithful draft steps.

    Attributes:
        technique: Name of the trained technique (``None``: the one attached technique that implements
            :class:`StepDrafter`).
        steps: Draft steps per rollout (capped at the drafter's :meth:`~StepDrafter.step_limit`).
        stride: Distance between rollout starts in training (``1``: every position, the first start drawn
            uniformly from ``[0, stride)`` otherwise).
        divergence: A name in :data:`~economize.train.objectives.DIVERGENCES` (``"kl"`` = KL(teacher || drafter)).
        ce_weight: Weight of the cross-entropy on the true token (positions whose token lies outside a restricted
            draft vocabulary are left out of it).
        step_decay: Step ``k`` has weight ``step_decay ** (k - 1)``.
        head_chunk: Compute each step's logits and divergence in chunks of at most this many tokens, each under
            activation checkpointing (``0``: the whole step at once, still checkpointed).
        eval_horizon: Draft steps of the evaluation curve (steps beyond the drafter's limit score ``agree = run = 0``
            and ``NaN`` divergences).
        eval_stride: Distance between rollout starts in evaluation.
        eval_first: First rollout start in evaluation.
    """

    technique: str | None = None
    steps: int = 8
    stride: int = 1
    divergence: str = "kl"
    ce_weight: float = 0.0
    step_decay: float = 1.0
    head_chunk: int = 0
    eval_horizon: int = 16
    eval_stride: int = 32
    eval_first: int = 64

    def __post_init__(self) -> None:
        check_registered("divergence", self.divergence, DIVERGENCES)
        for name in ("steps", "stride", "eval_horizon", "eval_stride"):
            check_positive(name, getattr(self, name))
        check_number("step_decay", self.step_decay, low=0, closed_low=False)
        check_number("ce_weight", self.ce_weight, low=0)
        check_non_negative("head_chunk", self.head_chunk)

    # ---- objective interface -------------------------------------------------------------------------------------
    def drafter(self, model: EfficientModel) -> Technique:
        """The trained technique."""
        return find_drafter(model, self.technique, StepDrafter, "step_distill", "StepDrafter")

    def setup(self, model: EfficientModel) -> None:
        self.drafter(model)

    def parameter_groups(self, model: EfficientModel) -> list[ParamGroup]:
        groups = [g for g in cast(StepDrafter, self.drafter(model)).step_parameter_groups() if g.params]
        if not groups:
            raise IncompatibleTechniquesError("the drafter has no trainable parameters")
        return groups

    def teacher_request(self, model: EfficientModel) -> TeacherRequest:
        own = cast(StepDrafter, self.drafter(model)).step_request(model)
        return own.merge(TeacherRequest(hidden_layers=(model.adapter.num_layers,)))

    # ---- extension points (subclasses change the training loss, never the evaluation curve) -----------------------
    def step_divergence(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Per-position divergence ``(teacher log-probs, drafter log-probs) -> [B, S]`` of the training loss."""
        return DIVERGENCES.get(self.divergence)

    def rollout_states(self, drafter: StepDrafter, model: EfficientModel, batch: Batch, teacher: TeacherOutputs,
                       starts: torch.Tensor, steps: int) -> Sequence[torch.Tensor]:
        """The draft-step states the training loss scores (default: :meth:`StepDrafter.step_states`)."""
        return drafter.step_states(model, batch, teacher, starts, steps)

    def _steps(self, drafter: StepDrafter, wanted: int) -> int:
        limit = drafter.step_limit()
        return wanted if limit is None else min(wanted, limit)

    def loss(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs | None) -> ObjectiveOutput:
        teacher = require_teacher(teacher, type(self).__name__)
        drafter = cast(StepDrafter, self.drafter(model))
        ids = batch.input_ids
        length = ids.shape[1]
        steps = self._steps(drafter, self.steps)
        starts = rollout_starts(length, length - steps - 1, self.stride, first=0 if self.stride == 1 else None,
                                device=ids.device)
        vocab = drafter.step_vocab()
        inside = None if vocab is None else _teacher_inside(teacher, vocab, length, self.head_chunk)
        divergence = self.step_divergence()

        def step_sums(states: torch.Tensor, positions: torch.Tensor, counted: torch.Tensor,
                      k: int) -> torch.Tensor:
            """Sums of the divergence and of the true tokens' NLL over the counted positions of a column chunk."""
            student = drafter.step_logits(states, k).float().log_softmax(-1)
            reference = teacher.log_probs(slice(None), positions).to(student.device)
            if vocab is not None:
                reference = reference.index_select(-1, vocab.to(student.device)).log_softmax(-1)
            weights = counted.to(student.device, torch.float32)
            sums = [(divergence(reference, student) * weights).sum()]
            if self.ce_weight:
                targets = ids[:, positions + 1].to(student.device)
                column, known = _columns(targets, vocab)
                if known is not None:
                    weights = weights * known
                sums.append((token_nll(student, column) * weights).sum())
            return torch.stack(sums)

        terms, metrics = [], {}
        all_states = self.rollout_states(drafter, model, batch, teacher, starts, steps)
        for k, states in enumerate(all_states, start=1):
            positions = starts + k
            weights = target_weights(batch, positions + 1)
            counted = torch.ones(states.shape[:2], dtype=torch.bool, device=states.device)
            if weights is not None:
                counted = counted & weights.to(states.device)
            if inside is not None:
                counted = counted & inside[:, positions.to(inside.device)].to(states.device)
            sums = checkpointed_chunks(step_sums, [(states, 1), (positions, 0), (counted, 1)], k,
                                       rows=states.shape[0], chunk=self.head_chunk)
            value = sums[0] / counted.sum().clamp_min(1)
            if self.ce_weight:
                known = _columns(ids[:, positions + 1], vocab)[1]
                ce_count = counted if known is None else counted & known.to(counted.device)
                value = value + self.ce_weight * sums[1] / ce_count.sum().clamp_min(1)
            metrics[f"{self.divergence}{k}"] = value.detach()
            terms.append(value)
        loss = decayed_mean(terms, self.step_decay)
        return ObjectiveOutput(loss, metrics)

    @torch.no_grad()
    def evaluate(
        self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs | None
    ) -> Mapping[str, torch.Tensor | float]:
        """The teacher-forced curve of one batch: per draft step ``k = 1..eval_horizon`` (vectors) the metrics of
        :data:`STEP_CURVE_METRICS`, defined as for the other drafter objectives. With a
        restricted draft vocabulary ``agree`` compares the drafter's argmax (a model token) with the teacher's over
        the whole vocabulary, ``kl`` is taken against the teacher restricted to the draft vocabulary, and
        ``excess_nll`` counts only true tokens inside it."""
        teacher = require_teacher(teacher, type(self).__name__)
        drafter = cast(StepDrafter, self.drafter(model))
        ids = batch.input_ids
        length = ids.shape[1]
        starts = rollout_starts(length, length - self.eval_horizon - 2, self.eval_stride, first=self.eval_first,
                                device=ids.device)
        steps = self._steps(drafter, self.eval_horizon)
        vocab = drafter.step_vocab()
        rows: list[torch.Tensor] = []
        alive: torch.Tensor | None = None
        for k, states in enumerate(drafter.step_states(model, batch, teacher, starts, steps), start=1):
            positions = starts + k
            student = drafter.step_logits(states, k).float().log_softmax(-1)
            device = student.device
            weights = target_weights(batch, positions + 1)
            weights = None if weights is None else weights.to(device)
            full = teacher.log_probs(slice(None), positions).to(device)
            targets = ids[:, positions + 1].to(device)
            nll_teacher = token_nll(full, targets)
            top = student.argmax(-1)
            if vocab is None:
                reference, outside = full, torch.zeros_like(nll_teacher)
                column, known = targets, None
            else:
                vocab = vocab.to(device)
                restricted = full.index_select(-1, vocab)
                outside = 1 - restricted.exp().sum(-1)
                reference = restricted.log_softmax(-1)
                top = vocab[top]
                column, known = _columns(targets, vocab)
            kl = forward_kl(reference, student)
            nll_student = token_nll(student, column)
            agree = (top == full.argmax(-1)).float()
            alive = agree if alive is None else alive * agree
            excess_weights = known if weights is None else (weights if known is None else weights & known)
            values = (masked_mean(kl, weights), masked_mean(nll_student - nll_teacher, excess_weights),
                      masked_mean(agree, weights), masked_mean(nll_teacher, weights), masked_mean(alive, weights),
                      masked_mean(outside, weights))
            rows.append(torch.stack([v.float() for v in values]))
        curve = torch.stack(rows)  # [steps, metrics]
        if steps < self.eval_horizon:  # steps the drafter cannot make: rejected, no distribution
            missing = curve.new_full((self.eval_horizon - steps, len(STEP_CURVE_METRICS)), float("nan"))
            for j, name in enumerate(STEP_CURVE_METRICS):
                if name in ("agree", "run"):
                    missing[:, j] = 0.0
            curve = torch.cat([curve, missing])
        return {name: curve[:, j] for j, name in enumerate(STEP_CURVE_METRICS)}


def _columns(targets: torch.Tensor, vocab: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Column of every target token in the logits of a (sorted) draft vocabulary, and whether it is inside it."""
    if vocab is None:
        return targets, None
    vocab = vocab.to(targets.device)
    column = torch.searchsorted(vocab, targets).clamp(max=len(vocab) - 1)
    return column, vocab[column] == targets


def _teacher_inside(teacher: TeacherOutputs, vocab: torch.Tensor, length: int, chunk: int) -> torch.Tensor:
    """Whether the teacher's argmax at every position ``[B, T]`` lies in ``vocab`` (computed in chunks)."""
    size = chunk if chunk > 0 else length
    with torch.no_grad():
        tops = [teacher.log_probs(slice(None), slice(begin, begin + size)).argmax(-1)
                for begin in range(0, length, size)]
    top = torch.cat(tops, 1)
    member = torch.zeros(int(max(int(vocab.max()), int(top.max()))) + 1, dtype=torch.bool, device=top.device)
    member[vocab.to(top.device)] = True
    return member[top]

