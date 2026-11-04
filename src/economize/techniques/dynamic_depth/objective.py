"""Training objectives of the dynamic-depth techniques.

The base model stays frozen; only the routers, gates, adapters and classifiers the techniques add are trained.
Every objective except ``calm`` runs a student pass over the batch with the technique's policy in training mode
(every layer runs on every token and the policy gates the outputs differentiably, see
:class:`~.base.DepthPolicy`), and scores its logits with

    lm = ce_weight * CE(student, next token) + kl_weight * divergence(teacher || student)

plus the technique's own regularizer:

* ``gateskip``: ``sparsity * mean(gate)``, with a per-sequence top-k budget annealed from 1 to ``final_budget``;
* ``mod``: ``aux_weight * BCE(sigmoid(router), top-k membership)``, the predictor that lets decoding route causally;
* ``flexidepth``: ``layer_penalty * mean((sum_l g_l)^2)``, the paper's squared layer-count penalty;
* ``loradrop``: none (the low-rank updates learn to imitate the layers they replace);
* ``calm``: no student pass; the exit classifier learns from the teacher's states which layers already predict
  the final layer's top token.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ...errors import ConfigError, IncompatibleTechniquesError
from ...train.objective import (
    Batch,
    ObjectiveOutput,
    TeacherOutputs,
    TeacherRequest,
    TechniqueObjective,
    register_objective,
    require_teacher,
)
from ...train.objectives.common import check_loss_weights, distill_parts
from ...train.objectives.divergences import DIVERGENCES, masked_mean, token_nll
from ...train.objectives.weights import target_weights
from ...validation import check_registered, check_unit_interval
from .base import DepthPolicy, DepthTechnique, real_tokens
from .calm import CalmTechnique
from .flexidepth import FlexiDepthTechnique
from .gateskip import GateSkipTechnique
from .loradrop import LoRADropTechnique
from .mod import MoDTechnique

if TYPE_CHECKING:
    from ...core.model import EfficientModel

__all__ = ["CalmObjective", "DepthLMObjective", "FlexiDepthObjective", "GateSkipObjective", "LoRADropObjective",
           "MoDObjective", "student_pass", "student_states"]

_D = TypeVar("_D", bound=DepthTechnique)


def student_pass(model: EfficientModel, batch: Batch, policy: DepthPolicy) -> torch.Tensor:
    """Logits ``[B, T, V]`` of a full pass over ``batch`` with ``policy`` in training mode (fresh cache)."""
    return model.runner.logits(student_states(model, batch, policy))


def student_states(model: EfficientModel, batch: Batch, policy: DepthPolicy) -> torch.Tensor:
    """Last-layer states ``[B, T, D]`` of a full pass over ``batch`` with ``policy`` in training mode (fresh
    cache)."""
    runner = model.runner
    ids = batch.input_ids.to(model.adapter.device)
    rows, length = ids.shape
    cache = runner.new_cache(length, rows)
    valid = None if batch.attention_mask is None else batch.attention_mask.to(ids.device).bool()
    ctx = runner.context(cache, torch.arange(length, device=ids.device), valid, trusted=True)
    policy.training = True
    return runner.forward(ids, ctx, policy=policy).final


@dataclass
class DepthLMObjective(TechniqueObjective[_D]):
    """Shared base: the language-model loss of the student pass (see the module docstring).

    Subclasses name the technique class they train as the type argument (``DepthLMObjective[GateSkipTechnique]``);
    its registry name is the default attachment name.

    Attributes:
        technique: Name of the trained technique (empty: the registry name of :meth:`technique_type`).
        ce_weight: Weight of the cross-entropy on the next token.
        kl_weight: Weight of the divergence from the frozen model's distribution (``0``: no teacher pass).
        divergence: A name in :data:`~economize.train.objectives.DIVERGENCES`.
        head_chunk: Tokens per chunk of the LM head and the loss (``0``: the whole batch at once). Each chunk runs
            under activation checkpointing, so the float32 log-probabilities of a large vocabulary never exist for
            the whole batch at once (they need ``tokens * V * 4`` bytes, 4 GiB for 4k tokens of a 262k vocabulary).
    """

    technique: str = ""
    ce_weight: float = 1.0
    kl_weight: float = 0.0
    divergence: str = "kl"
    head_chunk: int = 1024

    def __post_init__(self) -> None:
        self.technique = self.technique or self.technique_type().name
        check_registered("divergence", self.divergence, DIVERGENCES)
        check_loss_weights(self.ce_weight, self.kl_weight)
        if self.head_chunk < 0:
            raise ConfigError(f"head_chunk must be >= 0, got {self.head_chunk}")

    def teacher_request(self, model: EfficientModel) -> TeacherRequest | None:
        """The final layer's states when the loss has a divergence term, else no teacher pass."""
        return TeacherRequest(hidden_layers=(model.adapter.num_layers,)) if self.kl_weight else None

    def lm_loss(self, batch: Batch, logits: torch.Tensor, teacher: TeacherOutputs | None
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """The language-model term and its parts."""
        length = logits.shape[1]
        student = logits[:, : length - 1].float().log_softmax(-1)
        weights = target_weights(batch, slice(1, length))
        reference = None
        if self.kl_weight:
            teacher = require_teacher(teacher, type(self).__name__)
            reference = teacher.log_probs(slice(None), slice(0, length - 1)).to(student.device)
        parts = distill_parts(student, weights, teacher=reference, divergence=DIVERGENCES.get(self.divergence),
                              targets=batch.input_ids[:, 1:].to(student.device), kl_weight=self.kl_weight,
                              ce_weight=self.ce_weight)
        return parts.loss, parts.metrics("ce", self.divergence)

    def chunked_lm_loss(self, model: EfficientModel, batch: Batch, final: torch.Tensor,
                        teacher: TeacherOutputs | None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """:meth:`lm_loss` from the student's last-layer states ``final`` ``[B, T, D]``, with the head and the loss
        run over chunks of at most :attr:`head_chunk` tokens (same value up to the rounding of the head's
        matmuls)."""
        rows, length = final.shape[:2]
        count = length - 1
        reference_of = None
        if self.kl_weight:
            reference_of = require_teacher(teacher, type(self).__name__)
        divergence = DIVERGENCES.get(self.divergence)
        targets = batch.input_ids[:, 1:].to(final.device)
        weights = target_weights(batch, slice(1, length))
        weights = (torch.ones(rows, count, device=final.device) if weights is None
                   else weights.to(final.device).float())
        kl_on, ce_on = bool(self.kl_weight), bool(self.ce_weight)

        def sums(states: torch.Tensor, chunk_targets: torch.Tensor, chunk_weights: torch.Tensor,
                 begin: int) -> torch.Tensor:
            student = model.runner.logits(states).float().log_softmax(-1)
            zero = student.new_zeros(())
            div = ce = zero
            if kl_on:
                assert reference_of is not None
                reference = reference_of.log_probs(slice(None), slice(begin, begin + states.shape[1]))
                div = (divergence(reference.to(student.device), student) * chunk_weights).sum()
            if ce_on:
                ce = (token_nll(student, chunk_targets) * chunk_weights).sum()
            return torch.stack([div, ce])

        size = count if not self.head_chunk else max(self.head_chunk // rows, 1)
        total: torch.Tensor | None = None
        for begin in range(0, count, size):
            width = min(size, count - begin)
            value = checkpoint(sums, final[:, begin: begin + width], targets[:, begin: begin + width],
                               weights[:, begin: begin + width], begin, use_reentrant=False)
            total = value if total is None else total + value
        assert total is not None
        total = total / weights.sum().clamp_min(1.0)
        div, ce = total[0], total[1]
        terms = []
        metrics: dict[str, torch.Tensor] = {}
        if ce_on:
            terms.append(self.ce_weight * ce)
            metrics["ce"] = ce.detach()
        if kl_on:
            terms.append(div if self.kl_weight == 1 else self.kl_weight * div)
            metrics[self.divergence] = div.detach()
        return sum(terms[1:], terms[0]), metrics

    def regularizer(self, model: EfficientModel, batch: Batch, policy: Any
                    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        """The technique's own loss term after the student pass (``None``: none)."""
        return None, {}

    def configure(self, policy: Any) -> None:
        """Adjust a fresh training policy (budgets, schedules) before the student pass."""

    def loss(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs | None) -> ObjectiveOutput:
        policy = self.trained(model).make_policy()
        self.configure(policy)
        loss, metrics = self.chunked_lm_loss(model, batch, student_states(model, batch, policy), teacher)
        extra, more = self.regularizer(model, batch, policy)
        if extra is not None:
            loss = loss + extra
        metrics.update(more)
        return ObjectiveOutput(loss, metrics)


@register_objective("gateskip", default_for=(GateSkipTechnique.name,))
@dataclass
class GateSkipObjective(DepthLMObjective[GateSkipTechnique]):
    """GateSkip: LM loss plus ``sparsity * mean(gate)``; from ``anneal_start`` of training on, only the
    ``budget`` fraction of each sequence's tokens with the highest gates keeps its update, the budget falling
    linearly from 1 to ``final_budget`` at the end of training."""

    sparsity: float = 0.01
    final_budget: float = 1.0
    anneal_start: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        check_unit_interval("final_budget", self.final_budget, closed_low=False, closed_high=True)
        check_unit_interval("anneal_start", self.anneal_start)
        self._budget = 1.0

    def begin_step(self, step: int, total_steps: int) -> None:
        progress = step / max(total_steps, 1)
        ramp = max(0.0, (progress - self.anneal_start) / (1.0 - self.anneal_start))
        self._budget = 1.0 - (1.0 - self.final_budget) * min(1.0, ramp)

    def configure(self, policy: Any) -> None:
        policy.train_budget = self._budget

    def regularizer(self, model: EfficientModel, batch: Batch, policy: Any
                    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        if not policy.gates:
            return None, {}
        gates = torch.stack([g.to(model.adapter.device) for g in policy.gates.values()])  # [G, B, T]
        mean_gate = masked_mean(gates, real_tokens(batch.attention_mask, gates[0])[None].expand_as(gates))
        return self.sparsity * mean_gate, {"mean_gate": mean_gate.detach(),
                                           "budget": torch.tensor(self._budget)}


@register_objective("mod", default_for=(MoDTechnique.name,))
@dataclass
class MoDObjective(DepthLMObjective[MoDTechnique]):
    """Mixture-of-Depths: LM loss under per-sequence top-k routing plus ``aux_weight`` times the BCE of the causal
    router predictor."""

    aux_weight: float = 0.1

    def regularizer(self, model: EfficientModel, batch: Batch, policy: Any
                    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        if not policy.aux_losses:
            return None, {}
        device = model.adapter.device
        aux = torch.stack([a.to(device) for a in policy.aux_losses]).mean()
        routed = torch.stack([r.to(device) for r in policy.routed_fraction]).mean()
        return self.aux_weight * aux, {"router_bce": aux.detach(), "causal_routed_fraction": routed.detach()}


@register_objective("flexidepth", default_for=(FlexiDepthTechnique.name,))
@dataclass
class FlexiDepthObjective(DepthLMObjective[FlexiDepthTechnique]):
    """FlexiDepth: LM loss plus ``layer_penalty * mean((sum_l g_l)^2)`` over tokens."""

    layer_penalty: float = 1e-3

    def regularizer(self, model: EfficientModel, batch: Batch, policy: Any
                    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        if not policy.scores:
            return None, {}
        scores = torch.stack([s.to(model.adapter.device) for s in policy.scores.values()])  # [F, B, T]
        valid = real_tokens(batch.attention_mask, scores[0])
        penalty = masked_mean(scores.sum(0).pow(2), valid)
        threshold = self.trained(model).config.threshold
        used = masked_mean((scores > threshold).float().mean(0), valid)
        return self.layer_penalty * penalty, {"layer_penalty": penalty.detach(), "run_fraction": used.detach()}


@register_objective("loradrop", default_for=(LoRADropTechnique.name,))
@dataclass
class LoRADropObjective(DepthLMObjective[LoRADropTechnique]):
    """LoRA-Drop: LM loss with the dropped positions of every droppable layer taking the low-rank update."""


@register_objective("calm", default_for=(CalmTechnique.name,))
@dataclass
class CalmObjective(TechniqueObjective[CalmTechnique]):
    """CALM's exit classifier: binary cross-entropy against the consistency label ``argmax head(h_l) ==
    argmax head(h_L)`` at every exit layer ``l`` of the frozen model.

    Attributes:
        technique: Name of the attached CALM technique (``confidence="classifier"``).
        positive_weight: Weight of the positive (consistent) examples in the BCE.
    """

    technique: str = CalmTechnique.name
    positive_weight: float = 1.0

    group: ClassVar[str] = "exit_classifier"

    def trained(self, model: EfficientModel) -> CalmTechnique:
        """The trained technique.

        Raises:
            IncompatibleTechniquesError: No CALM technique is attached as :attr:`technique`, or its confidence
                measure has no trainable logit.
        """
        technique = super().trained(model)
        if not technique.measure.trainable:
            raise IncompatibleTechniquesError(
                f"CALM's {technique.config.confidence!r} confidence has no trainable logit (use 'classifier')"
            )
        return technique

    def teacher_request(self, model: EfficientModel) -> TeacherRequest:
        """The states of every exit layer and of the last layer."""
        config = self.trained(model).config
        return TeacherRequest(hidden_layers=tuple(range(config.min_layer, model.adapter.num_layers + 1)))

    def loss(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs | None) -> ObjectiveOutput:
        teacher = require_teacher(teacher, type(self).__name__)
        technique = self.trained(model)
        last = model.adapter.num_layers
        with torch.no_grad():
            final_top = teacher.head(teacher.final).argmax(-1)
        weights = None if batch.attention_mask is None else batch.attention_mask.bool()
        losses, consistent = [], []
        for layer in range(technique.config.min_layer, last):
            hidden = teacher.hidden[layer]
            with torch.no_grad():
                label = (teacher.head(hidden).argmax(-1) == final_top).float()
            logit = technique.measure.logit(layer, hidden)
            label = label.to(logit.device)
            bce = F.binary_cross_entropy_with_logits(
                logit, label, reduction="none", pos_weight=torch.tensor(self.positive_weight, device=logit.device)
            )
            w = None if weights is None else weights.to(logit.device)
            losses.append(masked_mean(bce, w))
            consistent.append(masked_mean(label, w))
        loss = torch.stack(losses).mean()
        return ObjectiveOutput(loss, {"bce": loss.detach(), "consistent": torch.stack(consistent).mean().detach()})
