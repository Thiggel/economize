"""The whole-layer skipping technique: its skip set, one draft step and the search at attach time."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from ...core.calibration import calibration_source, token_batches
from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import AttachContext, Capability, Technique, register_technique
from ...decoding.cost import LayerFootprint
from .config import LayerSkipConfig
from .policy import SkipLayers
from .search import SKIP_SEARCHES, SkipScorer

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...runtime import LayerContext, LayerRunner
    from .drafter import LayerSkipDrafter

__all__ = ["LayerSkipTechnique"]


@register_technique
class LayerSkipTechnique(Technique):
    """Self-speculative whole-layer greedy skip: draft tokens run the model without a fixed set of whole layers.

    This approximates Draft & Verify (Zhang et al. 2024), SWIFT (Xia et al. 2025) and CLaSp (Chen et al. 2025):
    the skip set holds whole decoder layers (Draft & Verify skips attention and MLP sub-layers separately), is
    chosen once on calibration data by the registered search (greedy by default; the papers use Bayesian
    optimization or re-optimize the set during decoding) and stays fixed while decoding. A draft step embeds the last
    token, runs every layer outside the skip set at its position and reads the next-token logits with the model's own
    norm and head. Skipped layers write no K/V; the next exact pass recomputes every drafted position with all layers.
    The technique adds no parameters. Its buffer ``skip_mask`` (``[L + 1]``, index 0 unused) holds the skip set, so a
    searched set is saved and restored with the technique.
    """

    name = LayerSkipConfig.technique
    config_class = LayerSkipConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True)
    capabilities = Capability.DRAFTER
    config: LayerSkipConfig
    skip_mask: torch.Tensor  # buffer

    def __init__(self, config: LayerSkipConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)
        mask = torch.zeros(adapter.num_layers + 1, dtype=torch.bool)
        mask[list(config.skip_layers)] = True
        self.register_buffer("skip_mask", mask.to(adapter.device))
        self._policy: SkipLayers | None = None

    # ---- the skip set --------------------------------------------------------------------------------------------
    @property
    def skip_layers(self) -> tuple[int, ...]:
        """The skipped layers (empty until a searched set is found at attach)."""
        return tuple(int(i) for i in self.skip_mask.nonzero().flatten().tolist())

    @property
    def policy(self) -> SkipLayers:
        # Reading the mask syncs with the host, which a CUDA-graph capture forbids: a captured draft step reuses the
        # policy that its warmup call (outside the capture) built.
        if self._policy is not None and self.skip_mask.is_cuda and torch.cuda.is_current_stream_capturing():
            return self._policy
        if self._policy is None or self._policy.layers != frozenset(self.skip_layers):
            self._policy = SkipLayers(self.skip_layers)
        return self._policy

    def search(self, runner: LayerRunner, batches: list[torch.Tensor] | None) -> tuple[int, ...]:
        """Run the configured search and store its result."""
        c = self.config
        search = SKIP_SEARCHES.get(c.search)()
        scorer = SkipScorer(runner, batches, c.metric) if search.needs_data and batches is not None else None
        chosen = search(scorer, c.candidates(self.adapter.num_layers), c.size)
        self.skip_mask.zero_()
        self.skip_mask[list(chosen)] = True
        return chosen

    # ---- the pieces of a draft step ------------------------------------------------------------------------------
    @property
    def run_layers(self) -> tuple[int, ...]:
        """The layers a draft step runs (all layers outside the skip set)."""
        skip = self.skip_mask.tolist()
        return tuple(i for i in range(1, self.adapter.num_layers + 1) if not skip[i])

    @property
    def footprint(self) -> LayerFootprint:
        """The layers one draft step runs, for the cost log (not one range unless the skip set is at an end)."""
        return LayerFootprint.of(self.run_layers)

    @property
    def layer_span(self) -> tuple[int, int]:
        """First and last layer a draft step runs. The layers in between are not all run: read :attr:`footprint`
        for the cost (a :class:`LayerSkipDrafter` declares it)."""
        layers = self.run_layers
        return layers[0], layers[-1]

    def run(self, runner: LayerRunner, embedding: torch.Tensor, ctx: LayerContext) -> torch.Tensor:
        """Output of the last layer for token embeddings ``[B, S, d]`` with the skip set skipped."""
        hidden, _ = runner.run(embedding, ctx, 1, self.adapter.num_layers, policy=self.policy)
        return hidden

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.adapter.logits(hidden)

    # ---- technique interface ------------------------------------------------------------------------------------
    def attach(self, model: nn.Module, adapter: ArchAdapter, context: AttachContext | None = None) -> None:
        """Search the skip set on the calibration data, unless it is configured or was loaded."""
        if self.skip_mask.any():
            return
        from ...runtime import LayerRunner

        needs_data = SKIP_SEARCHES.get(self.config.search).needs_data
        batches = token_batches(calibration_source(context, self.name)) if needs_data else None
        self.search(LayerRunner(adapter), batches)

    def drafter(self, runner: LayerRunner) -> LayerSkipDrafter:
        from .drafter import LayerSkipDrafter

        if not self.skip_mask.any():
            raise RuntimeError("the layer_skip technique has no skip set yet: attach it first")
        return LayerSkipDrafter(self, runner)

    def extra_repr(self) -> str:
        return f"skip_layers={list(self.skip_layers)}"
