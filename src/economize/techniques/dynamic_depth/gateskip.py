"""GateSkip, layer-level variant: residual gates that rank tokens by importance and skip the least important ones.

Every gated layer ``l`` gets a gate ``g = sigmoid(w . rms(x) + b)`` computed from its input ``x``. A token that runs
the layer leaves it as ``x + g (layer(x) - x)``; a token that skips it leaves as ``x``. Training (objective
``gateskip``) teaches the gates with a sparsity penalty, and at inference the tokens with the lowest gates skip.

Deviation from the paper (Laitenberger et al. 2025, arXiv 2510.13876): GateSkip gates the attention and the MLP
branch of every layer separately, with a vector-valued gate on each branch output whose mean is the token's score,
and a token skips a branch, not the whole layer. economize's runtime decides per layer only, so this technique gates
whole layers with one scalar gate computed from the layer input; tables should name it "GateSkip (layer-level
variant)". The per-branch method needs the runtime's sub-layer actions (``ArchAdapter.branches(layer)`` and a
branch-level decision, see :mod:`economize.techniques.layer_skip` for the same requirement).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from ...core.config import TechniqueConfig
from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import register_technique
from ...validation import check_choice, check_number, check_unit_interval
from .base import (
    DepthTechnique,
    LayeredDepthConfig,
    ScalarScore,
    ScoredDepthPolicy,
    gated_residual,
    layer_modules,
    module_at,
    top_fraction,
)

__all__ = ["GATESKIP_RULES", "GateSkipConfig", "GateSkipPolicy", "GateSkipTechnique", "ResidualGate", "top_fraction"]

#: ``"threshold"``: a token skips when its gate is below ``threshold``. ``"budget"``: within every sequence of a
#: pass, only the ``budget`` fraction of its tokens with the highest gates runs each layer.
GATESKIP_RULES = ("threshold", "budget")


@dataclass(frozen=True)
class GateSkipConfig(LayeredDepthConfig):
    """GateSkip, layer-level variant (residual-gated layer skipping; see the module docstring).

    Attributes:
        layers: Gated layers (empty: every layer).
        rule: Inference rule (:data:`GATESKIP_RULES`).
        threshold: ``rule="threshold"``: tokens whose gate is below it skip the layer.
        budget: ``rule="budget"``: fraction of each sequence's tokens in a pass that run each gated layer
            (rounded up, so a single-token decode step always runs; ``1``: all).
        gate_bias: Initial gate bias. At ``>= 100`` the gates are exactly ``1`` and the model is unchanged.
        gate_init_std: Standard deviation of the initial gate weights (``0``: the gate starts at the bias).
        skip_prompt: Gate the prompt too (``False``: the prompt runs every layer).
    """

    technique: ClassVar[str] = "gateskip"
    rule: str = "threshold"
    threshold: float = 0.5
    budget: float = 1.0
    gate_bias: float = 5.0
    gate_init_std: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        check_choice("rule", self.rule, GATESKIP_RULES)
        check_unit_interval("threshold", self.threshold, closed_high=True)
        check_unit_interval("budget", self.budget, closed_low=False, closed_high=True)
        check_number("gate_init_std", self.gate_init_std, low=0)


class ResidualGate(ScalarScore):
    """``sigmoid(w . rms(x) + b)``, one scalar per token."""

    def __init__(self, hidden_size: int, bias: float, std: float) -> None:
        super().__init__(hidden_size, bias, std, activation="sigmoid")


class GateSkipPolicy(ScoredDepthPolicy):
    """Skips the tokens with low gates (see :class:`GateSkipConfig`). In training mode every token runs, the output
    is gated softly, and the ``train_budget`` fraction of each sequence's tokens with the highest gates keeps its
    update (the others are zeroed, as at inference)."""

    def __init__(self, technique: GateSkipTechnique) -> None:
        super().__init__(technique.adapter.num_layers, technique.config.phases)
        self.technique = technique
        self.train_budget = 1.0

    @property
    def gates(self) -> dict[int, torch.Tensor]:
        """The gates of the current pass (the objective reads them for the sparsity loss)."""
        return self.scores

    def score(self, index: int, x: torch.Tensor) -> torch.Tensor | None:
        return self.technique.gate(index, x)

    def select(self, index: int, score: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        config = self.technique.config
        if config.rule == "budget":
            return top_fraction(score, valid, config.budget)
        return score >= config.threshold

    def train_select(self, index: int, score: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return top_fraction(score.detach(), valid, self.train_budget)

    def combine(self, index: int, x: torch.Tensor, out: torch.Tensor, score: torch.Tensor,
                ran: torch.Tensor | None) -> torch.Tensor:
        if ran is not None and self.training:  # dropped tokens keep x (at inference out == x for them already)
            score = torch.where(ran, score, torch.zeros_like(score))
        return gated_residual(x, out, score)


@register_technique
class GateSkipTechnique(DepthTechnique):
    """GateSkip, layer-level variant (see :class:`GateSkipConfig`); trained by the objective ``gateskip``."""

    name: ClassVar[str] = GateSkipConfig.technique
    config_class: ClassVar[type[TechniqueConfig]] = GateSkipConfig
    info = TechniqueInfo(
        family=Family.DYNAMIC_DEPTH,
        saves=Saving.COMPUTE | Saving.LATENCY,
        lossless=False,
        trainable=("gates",),
    )
    config: GateSkipConfig
    order: ClassVar[int] = 30

    def __init__(self, config: GateSkipConfig, adapter: Any) -> None:
        super().__init__(config, adapter)
        self.gates = layer_modules(
            adapter,
            config.layer_set(adapter.num_layers),
            lambda _: ResidualGate(adapter.hidden_size, config.gate_bias, config.gate_init_std),
        )

    def gate(self, index: int, hidden: torch.Tensor) -> torch.Tensor | None:
        """Gates ``[B, S]`` of layer ``index`` for its input ``hidden``, or ``None`` if the layer is not gated."""
        module = module_at(self.gates, index)
        return None if module is None else module(hidden)

    def make_policy(self) -> GateSkipPolicy:
        return GateSkipPolicy(self)
