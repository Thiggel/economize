"""SkipDecode (Del Corro et al. 2023): a layer budget that falls with the generated position, one budget per batch.

The ``t``-th generated token (``t = 0, 1, ...``, counted from the end of its row's prompt) runs ``active(t)`` layers:
``max_layers`` until ``warmup``, then a linear decline to ``min_layers`` over ``decay`` positions, and
``min_layers`` from there on. It skips the ``L - active(t)`` lowest layers (``skip="bottom"``, as in the paper, so
that the last layers always run) or the highest ones (``skip="top"``, an early exit). Since the budget never grows
with the position, every layer a token runs was run by all earlier tokens, and skipped layers need no K/V: SkipDecode
needs neither K/V propagation nor recomputation. The prompt runs every layer.

As in the paper, the decision is uniform over the batch (``batch_uniform=True``) so that a decode step costs one
dense pass: all rows run the largest budget of any real token at that step. Since positions count generated tokens,
rows with different prompt lengths decoded in lockstep are at the same position and share the budget exactly. A
multi-token pass (verification after drafting) takes one budget per column, the largest over the rows, so a single
row gets every token's own budget. ``batch_uniform=False`` gives every token its own budget in every pass, which
makes the output independent of how tokens are grouped into passes and batches (a mixed pass then fills the K/V of
the tokens that skip, unless the runtime supports routed K/V).

Training-free here: the paper fine-tunes the model under the schedule, which economize leaves to the user (any
objective that trains the base model works with the policy in place).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from ...core.config import TechniqueConfig
from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import register_technique
from ...errors import ConfigError
from ...validation import check_choice, check_non_negative, check_positive
from .base import GENERATION_PHASES, DepthPolicy, DepthTechnique, valid_mask

if TYPE_CHECKING:
    from ...runtime import LayerContext

__all__ = ["SKIPDECODE_SIDES", "SkipDecodeConfig", "SkipDecodePolicy", "SkipDecodeTechnique"]

#: Which end of the model a token's skipped layers come from.
SKIPDECODE_SIDES = ("bottom", "top")


@dataclass(frozen=True)
class SkipDecodeConfig(TechniqueConfig):
    """SkipDecode.

    Attributes:
        max_layers: Layers a token runs before ``warmup`` (``0``: all of them).
        min_layers: Layers a token runs from ``warmup + decay`` on (``0``: all of them).
        warmup: Generated position until which tokens run ``max_layers``.
        decay: Generated positions over which the budget falls linearly from ``max_layers`` to ``min_layers``.
        skip: :data:`SKIPDECODE_SIDES`.
        batch_uniform: One budget per step for the whole batch (the paper) rather than per token (see the module
            docstring).
        fill_kv: Write the K/V of skipped layers anyway (not needed, see the module docstring; only for
            experiments that combine SkipDecode with techniques that read every layer's cache).
    """

    technique: ClassVar[str] = "skipdecode"
    max_layers: int = 0
    min_layers: int = 0
    warmup: int = 0
    decay: int = 256
    skip: str = "bottom"
    batch_uniform: bool = True
    fill_kv: bool = False

    def __post_init__(self) -> None:
        check_choice("skip", self.skip, SKIPDECODE_SIDES)
        for name in ("max_layers", "min_layers", "warmup"):
            check_non_negative(name, getattr(self, name))
        check_positive("decay", self.decay)

    def validate(self, adapter: Any) -> None:
        top, bottom = self.budget(adapter.num_layers)
        if not 1 <= bottom <= top <= adapter.num_layers:
            raise ConfigError(
                f"SkipDecode needs 1 <= min_layers <= max_layers <= {adapter.num_layers}, got {bottom} and {top}"
            )

    def budget(self, num_layers: int) -> tuple[int, int]:
        """``(max_layers, min_layers)`` with ``0`` resolved to ``num_layers``."""
        return self.max_layers or num_layers, self.min_layers or num_layers

    def active_layers(self, positions: torch.Tensor, num_layers: int) -> torch.Tensor:
        """Layers run by tokens at generated positions ``positions`` (any shape, long)."""
        top, bottom = self.budget(num_layers)
        progress = ((positions.float() - self.warmup) / self.decay).clamp(0.0, 1.0)
        return (top - (top - bottom) * progress).round().long()


class SkipDecodePolicy(DepthPolicy):
    """Runs the layers within the budget of the current step (or token) and skips the others."""

    def __init__(self, technique: SkipDecodeTechnique) -> None:
        super().__init__(technique.adapter.num_layers, GENERATION_PHASES)
        self.technique = technique
        self.fill_kv = technique.config.fill_kv
        self.budget: torch.Tensor | None = None  # [B, S] (or [1, S] when batch-uniform) layers run by each token

    def begin_pass(self, hidden: torch.Tensor, ctx: LayerContext) -> None:
        config = self.technique.config
        active = config.active_layers(self.generated_positions(hidden, ctx), self.num_layers)
        if config.batch_uniform:
            valid = valid_mask(hidden, ctx).to(active.device)
            active = active.masked_fill(~valid, 0).max(0, keepdim=True).values
            active = active.masked_fill(~valid.any(0, keepdim=True), self.num_layers)
        self.budget = active

    def choose(self, index: int, hidden: torch.Tensor, ctx: LayerContext) -> torch.Tensor | None:
        assert self.budget is not None
        bottom = self.technique.config.skip == "bottom"
        runs = index > self.num_layers - self.budget if bottom else index <= self.budget
        if bool(runs.all()):
            return None
        return runs.expand(hidden.shape[:2])


@register_technique
class SkipDecodeTechnique(DepthTechnique):
    """SkipDecode: a layer budget that falls with the generated position, skipping the lowest layers.

    Training-free; see :class:`SkipDecodeConfig` and the module docstring."""

    name: ClassVar[str] = SkipDecodeConfig.technique
    config_class: ClassVar[type[TechniqueConfig]] = SkipDecodeConfig
    info = TechniqueInfo(family=Family.DYNAMIC_DEPTH, saves=Saving.COMPUTE | Saving.LATENCY, lossless=False)
    config: SkipDecodeConfig
    order: ClassVar[int] = 33

    def make_policy(self) -> SkipDecodePolicy:
        return SkipDecodePolicy(self)
