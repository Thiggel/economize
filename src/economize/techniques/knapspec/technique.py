"""The KnapSpec technique: the current skip set, one draft step with sublayers skipped, and the search."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import AttachContext, Capability, Technique, register_technique
from ...decoding.cost import LayerFootprint
from .config import KnapSpecConfig
from .latency import SublayerLatency
from .search import KnapsackSearch, SkipPlan
from .sublayer import SublayerSkipper

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...runtime import LayerCall, LayerContext, LayerRunner
    from ..layer_skip.policy import SkipLayers
    from .drafter import KnapSpecDrafter

__all__ = ["KnapSpecTechnique"]


@register_technique
class KnapSpecTechnique(Technique):
    """KnapSpec (Cha et al., ICML 2026): self-speculative drafting with an adaptive set of skipped sublayers.

    A draft step embeds the last token and runs the model with the attention of some layers and the MLP of others
    skipped (:class:`~economize.techniques.knapspec.SublayerSkipper`), then reads the next-token logits with the
    model's own norm and head. A skipped attention writes no K/V; the next exact pass recomputes every drafted
    position with all layers. The drafter re-searches the skip set and draft length while decoding
    (:class:`~economize.techniques.knapspec.KnapsackSearch`) and sets :attr:`plan`; the set is not saved with the
    technique, which adds no parameters (a checkpoint is its configuration).

    Attributes:
        plan: The current skip set and draft length (``None`` before the first search: the full model).
        latency: Measured or configured sublayer latencies (``None`` until measured on the first draft).
        skipper: The partial-layer calls of the model.
        search: The knapsack search of the configuration.
    """

    name = KnapSpecConfig.technique
    config_class = KnapSpecConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True)
    capabilities = Capability.DRAFTER
    config: KnapSpecConfig

    def __init__(self, config: KnapSpecConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)
        self.skipper = SublayerSkipper(adapter)
        self.search = KnapsackSearch(self.skipper, config.candidates(adapter.num_layers), config.min_similarity,
                                     config.max_skip_fraction, config.max_draft)
        self.latency: SublayerLatency | None = SublayerLatency(*config.latency) if config.latency else None
        self._plan: SkipPlan | None = None
        self._calls: tuple[dict[int, LayerCall], SkipLayers | None] = ({}, None)

    # ---- the skip set --------------------------------------------------------------------------------------------
    @property
    def plan(self) -> SkipPlan | None:
        return self._plan

    @plan.setter
    def plan(self, plan: SkipPlan | None) -> None:
        self._plan = plan
        self._calls = ({}, None) if plan is None else self.skipper.calls(plan.attention, plan.mlp)

    @property
    def run_layers(self) -> tuple[int, ...]:
        """The layers a draft step runs at least one sublayer of."""
        L = self.adapter.num_layers
        return tuple(range(1, L + 1)) if self._plan is None else self._plan.run_layers(L)

    @property
    def footprint(self) -> LayerFootprint:
        """The layers one draft step runs, for the cost log. A layer with one sublayer skipped counts whole."""
        return LayerFootprint.of(self.run_layers)

    @property
    def layer_span(self) -> tuple[int, int]:
        """First and last layer a draft step runs (read :attr:`footprint` for the cost)."""
        layers = self.run_layers
        return layers[0], layers[-1]

    # ---- the pieces of a draft step ------------------------------------------------------------------------------
    def run(self, runner: LayerRunner, embedding: torch.Tensor, ctx: LayerContext) -> torch.Tensor:
        """Output of the last layer for token embeddings ``[B, S, d]`` with the sublayers of :attr:`plan` skipped."""
        overrides, policy = self._calls
        hidden, _ = runner.run(embedding, ctx, 1, self.adapter.num_layers, overrides=overrides, policy=policy)
        return hidden

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.adapter.logits(hidden)

    # ---- technique interface ------------------------------------------------------------------------------------
    def attach(self, model: nn.Module, adapter: ArchAdapter, context: AttachContext | None = None) -> None:
        """Nothing to prepare: the latencies are measured and the skip set searched while decoding."""

    def drafter(self, runner: LayerRunner) -> KnapSpecDrafter:
        from .drafter import KnapSpecDrafter

        return KnapSpecDrafter(self, runner)

    def extra_repr(self) -> str:
        plan = self._plan
        if plan is None:
            return "plan=None"
        return (f"skip_attention={sorted(plan.attention)}, skip_mlp={sorted(plan.mlp)}, "
                f"draft_length={plan.draft_length}")
