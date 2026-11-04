"""``jump_rollout``: train a jump drafter on parallel open-loop rollouts (teacher-forced tokens, BPTT over ``K`` steps).

Each batch is passed once through the unmodified model. Rollouts of ``horizon`` draft steps then start at every
``stride``-th position (with a random offset), all at once, and every step contributes

    divergence(teacher logits || jump logits) + ce_weight * CE(jump logits, true next token)
        + state_weight * state_loss(predicted h^m, exact h^m)

averaged over the batch's positions, with step ``k`` weighted by ``step_decay ** (k - 1)``. A predictor with its own
loss (``Predictor.has_auxiliary_loss``, e.g. the flow-matching ``"flow"`` predictor) contributes that loss in place
of ``state_loss``, under the same ``state_weight``. The loss matches the reference implementation in
``tests/train/test_jump_rollout_oracle.py`` (same starts, same terms, same reductions) to float32 precision.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

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
from ...train.objectives.common import STATE_LOSSES, checkpointed_chunks, decayed_mean, distill_term, normalized_mse
from ...train.objectives.divergences import DIVERGENCES, forward_kl, masked_mean, token_nll
from ...train.objectives.weights import rollout_starts, target_weights
from ...train.rollout import check_rollout_layers, recurrent_input_layers
from ...validation import check_registered
from .._drafting import BLOCK_LR, LORA_LR
from .rollout import jump_rollout
from .technique import JumpTechnique

if TYPE_CHECKING:
    from ...core.model import EfficientModel

__all__ = ["CURVE_METRICS", "JumpRollout"]


@register_objective("jump_rollout", default_for=(JumpTechnique.name,))
@dataclass
class JumpRollout(TechniqueObjective[JumpTechnique]):
    """Open-loop rollout loss of a :class:`~economize.techniques.jump.JumpTechnique`.

    Attributes:
        technique: Name of the attached jump technique.
        horizon: Draft steps ``K`` per rollout.
        stride: Distance between rollout starts (the first start is drawn uniformly from ``[0, stride)``).
        state_weight: Weight of the state loss.
        state_loss: A name in :data:`~economize.train.objectives.STATE_LOSSES` (``"nmse"``: squared error normalized
            by the target's mean square, ``"cos"``: one minus the cosine similarity) between the predicted and the
            exact state after layer ``m``. A predictor with its own
            loss (:meth:`~economize.techniques.jump.Predictor.auxiliary_loss`) uses that one instead (``nmse`` is still
            reported, the predictor's loss appears as the metrics ``aux1`` and ``auxK``).
        divergence: A name in :data:`~economize.train.objectives.DIVERGENCES` (``"kl"`` = KL(teacher || drafter)), or
            ``"none"`` for a state-only loss.
        ce_weight: Weight of the cross-entropy of the drafter on the true next token.
        step_decay: Step ``k`` of a rollout has weight ``step_decay ** (k - 1)`` (EAGLE-3 uses 0.8).
        eval_horizon: Draft steps of the teacher-forced evaluation curve.
        eval_stride: Distance between rollout starts in evaluation.
        eval_first: First rollout start in evaluation (the rollouts see at least this much context).
        checkpoint_layers: Recompute the layers ``m+1..r`` of the rollout during backward instead of keeping their
            activations (see :func:`~economize.techniques.jump.jump_rollout`); for long horizons and dense starts.
        head_chunk: Compute each step's logits and divergence in chunks of at most this many tokens, each under
            activation checkpointing (``0``: the whole step at once). Bounds the ``[tokens, vocab]`` float32
            buffers of a step when rollouts start at almost every position.
        group_lr: Learning rate of each parameter role of the technique
            (:meth:`~economize.techniques.jump.JumpTechnique.parameter_groups`) that trains at its own rate. Roles
            left out train at the run's rate for their kind (``lr`` for :attr:`head_groups`, else ``lr_backbone``,
            else ``lr``). A run's ``lr_backbone`` overrides these rates.
        group_weight_decay: Weight decay of each role (roles left out: the run's ``weight_decay``).
        head_groups: Roles that are new modules (optimizer kind ``"head"``); the others adapt pretrained layers
            (kind ``"backbone"``).
    """

    technique: str = JumpTechnique.name
    horizon: int = 16
    stride: int = 8
    state_weight: float = 1.0
    state_loss: str = "nmse"
    divergence: str = "kl"
    ce_weight: float = 0.0
    step_decay: float = 1.0
    eval_horizon: int = 64
    eval_stride: int = 32
    eval_first: int = 64
    checkpoint_layers: bool = False
    head_chunk: int = 0
    group_lr: dict[str, float] = field(default_factory=lambda: {"lora": LORA_LR, "blocks": BLOCK_LR})
    group_weight_decay: dict[str, float] = field(default_factory=lambda: {"lora": 0.0, "blocks": 0.0})
    head_groups: tuple[str, ...] = ("drafter",)

    def __post_init__(self) -> None:
        check_registered("state_loss", self.state_loss, STATE_LOSSES)
        if self.divergence != "none":
            check_registered("divergence", self.divergence, DIVERGENCES)

    # ---- objective interface -------------------------------------------------------------------------------------
    def setup(self, model: EfficientModel) -> None:
        technique = self.trained(model)
        check_rollout_layers(model, *technique.layer_span, recurrent=True)

    def parameter_groups(self, model: EfficientModel) -> list[ParamGroup]:
        """One group per parameter role of the technique, at the role's own rate (see :attr:`group_lr`)."""
        return [
            ParamGroup(role, params, "head" if role in self.head_groups else "backbone",
                       lr=self.group_lr.get(role), weight_decay=self.group_weight_decay.get(role))
            for role, params in self.trained(model).parameter_groups().items() if params
        ]

    def teacher_request(self, model: EfficientModel) -> TeacherRequest:
        technique = self.trained(model)
        inputs = recurrent_input_layers(model, *technique.layer_span)
        layers = {technique.target_layer, *technique.source_layers, *inputs}
        return TeacherRequest(hidden_layers=tuple(sorted(layers)), cache=True)

    def loss(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs | None) -> ObjectiveOutput:
        teacher = require_teacher(teacher, type(self).__name__)
        teacher_kv = teacher_cache(teacher, type(self).__name__)
        technique = self.trained(model)
        m = technique.target_layer
        ids = batch.input_ids
        length = ids.shape[1]
        starts = rollout_starts(length, length - self.horizon - 2, self.stride, device=ids.device)
        divergence = None if self.divergence == "none" else DIVERGENCES.get(self.divergence)

        state_loss = STATE_LOSSES.get(self.state_loss)

        def step_divergence(h_r_hat: torch.Tensor, positions: torch.Tensor, weights: torch.Tensor | None,
                            total: torch.Tensor | float) -> torch.Tensor:
            """The step's divergence term over the columns ``positions``, scaled by their share of the step's
            weight ``total`` (so that the chunks of a step add up to its mean)."""
            assert divergence is not None  # only called for an objective with a divergence
            student = technique.logits(h_r_hat).float().log_softmax(-1)
            device = student.device  # the head's, which differs from the input's under ``device_map``
            value = distill_term(teacher.log_probs(slice(None), positions).to(device), student,
                                 ids[:, positions + 1].to(device), divergence, self.ce_weight,
                                 None if weights is None else weights.to(device))[0]
            share = h_r_hat.shape[0] * len(positions) if weights is None else weights.to(device).float().sum()
            if isinstance(total, torch.Tensor):
                total = total.to(device)
            return value * (share / total)

        def step(k: int, positions: torch.Tensor, h_m_hat: torch.Tensor, h_r_hat: torch.Tensor,
                 auxiliary: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
            weights = target_weights(batch, positions + 1)
            target = teacher.hidden[m][:, positions.to(teacher.hidden[m].device)].to(h_m_hat.device).float()
            nmse = masked_mean(normalized_mse(h_m_hat, target), weights)
            if divergence is None:
                div = 0 * nmse
            else:
                rows, count = h_r_hat.shape[:2]
                total = rows * count if weights is None else weights.float().sum().clamp_min(1.0)
                div = checkpointed_chunks(step_divergence, [(h_r_hat, 1), (positions, 0), (weights, 1)], total,
                                          rows=rows, chunk=self.head_chunk).to(nmse.device)
            if auxiliary is not None:
                state = masked_mean(auxiliary, weights)
                return div + self.state_weight * state, div.detach(), nmse.detach(), state.detach()
            state = nmse if self.state_loss == "nmse" else masked_mean(state_loss(h_m_hat, target), weights)
            return div + self.state_weight * state, div.detach(), nmse.detach()

        auxiliary = technique.predictor.has_auxiliary_loss
        outputs = jump_rollout(model, technique, ids, dict(teacher.hidden), teacher_kv, starts, self.horizon, step,
                               checkpoint_layers=self.checkpoint_layers, auxiliary=auxiliary,
                               auxiliary_mask=batch.attention_mask)
        loss = decayed_mean([out[0] for out in outputs], self.step_decay)
        metrics = {
            "kl1": outputs[0][1],
            "klK": outputs[-1][1],
            "nmse1": outputs[0][2],
            "nmseK": outputs[-1][2],
        }
        if auxiliary:
            metrics.update(aux1=outputs[0][3], auxK=outputs[-1][3])
        return ObjectiveOutput(loss, metrics)

    @torch.no_grad()
    def evaluate(
        self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs | None
    ) -> Mapping[str, torch.Tensor | float]:
        """The teacher-forced curve of one batch: per draft step ``k = 1..eval_horizon`` (vectors), the KL to the
        teacher (``kl``), the excess NLL of the true next token over the teacher's (``excess_nll``), top-1
        agreement (``agree``), state error (``nmse``) and cosine (``cos``) at layer ``m``, the teacher's NLL
        (``teacher_nll``) and the fraction of rollouts whose steps ``1..k`` all agree with the teacher (``run``).
        Matches the reference implementation in ``tests/train/test_jump_rollout_oracle.py`` for one batch."""
        teacher = require_teacher(teacher, type(self).__name__)
        teacher_kv = teacher_cache(teacher, type(self).__name__)
        technique = self.trained(model)
        m = technique.target_layer
        ids = batch.input_ids
        length = ids.shape[1]
        starts = rollout_starts(length, length - self.eval_horizon - 2, self.eval_stride, first=self.eval_first,
                                device=ids.device)
        alive: list[torch.Tensor] = []

        def step(k: int, positions: torch.Tensor, h_m_hat: torch.Tensor, h_r_hat: torch.Tensor) -> torch.Tensor:
            weights = target_weights(batch, positions + 1)
            teacher_lp = teacher.log_probs(slice(None), positions)
            student = technique.logits(h_r_hat).float().log_softmax(-1).to(teacher_lp.device)
            nxt = ids[:, positions + 1]
            kl = forward_kl(teacher_lp, student)
            nll_student = token_nll(student, nxt)
            nll_teacher = token_nll(teacher_lp, nxt)
            agree = (student.argmax(-1) == teacher_lp.argmax(-1)).float()
            alive.append(agree if not alive else alive[-1] * agree)
            target = teacher.hidden[m][:, positions.to(teacher.hidden[m].device)].to(h_m_hat.device).float()
            nmse = normalized_mse(h_m_hat, target)
            cos = torch.nn.functional.cosine_similarity(h_m_hat.float(), target, dim=-1)
            values = (kl, nll_student - nll_teacher, agree, nmse, cos, nll_teacher, alive[-1])
            return torch.stack([masked_mean(v, weights).to(kl.device) for v in values])

        outputs = jump_rollout(model, technique, ids, dict(teacher.hidden), teacher_kv, starts,
                               self.eval_horizon, step)
        curve = torch.stack(outputs)  # [K, 7]
        return {name: curve[:, j] for j, name in enumerate(CURVE_METRICS)}


#: The metrics of :meth:`JumpRollout.evaluate`, in the order of the reference implementation's curve.
CURVE_METRICS = ("kl", "excess_nll", "agree", "nmse", "cos", "teacher_nll", "run")
