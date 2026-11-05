"""The MagicDec and TriForce techniques: lossless self-speculation with the base model on a sparse view of its cache."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from ...core.metadata import Family, KernelChoice, Saving, Simulation, TechniqueInfo
from ...core.technique import AttachContext, Capability, Technique, register_technique
from .config import MagicDecConfig, TriForceConfig
from .path import SparseKVPath, sparse_layers, streaming_budget, streaming_keep
from .views import DRAFT_VIEWS, DraftBudget, MaskedView

if TYPE_CHECKING:
    import torch

    from ...arch import ArchAdapter
    from ...runtime import LayerContext, LayerRunner
    from .drafter import MagicDecDrafter, TriForceDrafter

__all__ = ["MagicDecTechnique", "StreamingPath", "TriForceTechnique", "sparse_layers"]


def _skips_masked_keys(adapter: ArchAdapter) -> bool:
    """Whether the model's attention kernel skips the key blocks a mask hides (economize's decode kernel)."""
    return getattr(adapter.config, "_attn_implementation", None) == "economize"


def _latent_cache(adapter: ArchAdapter) -> bool:
    """Whether the model caches one latent per position for all heads (multi-head latent attention, DeepSeek-V2/V3
    and Kimi-K2), which cannot give each KV head its own retrieved chunks."""
    return bool(getattr(adapter.config, "kv_lora_rank", None))


#: How TriForce simulates its saving on a latent cache: the gathered view falls back to the masked one there.
LATENT_READS = Simulation(Saving.LATENCY, "masks the KV entries outside each head's retrieved chunks, since the "
                          "multi-head latent attention cache shares every entry between the heads; the attention "
                          "kernel still reads them")


class _SparseDraftTechnique(Technique):
    """What MagicDec and TriForce share: no parameters, nothing to attach, a draft view chosen by ``config.view``.

    The default view gathers the budget's KV entries and realizes the saving on every attention implementation; the
    masked reference view simulates it (:data:`~.views.MASKED_READS`) except on economize attention, whose decode
    kernel skips the key blocks the mask hides.
    """

    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True,
                         kernel=KernelChoice("view", DRAFT_VIEWS))
    capabilities = Capability.DRAFTER
    config: MagicDecConfig | TriForceConfig

    def simulation(self) -> Simulation | None:
        """``None`` for the gathered view, and for the masked view on economize attention; else the masked reads."""
        masked = DRAFT_VIEWS.get(self.config.view) is MaskedView
        return None if masked and _skips_masked_keys(self.adapter) else super().simulation()

    def attach(self, model: nn.Module, adapter: ArchAdapter, context: AttachContext | None = None) -> None:
        """Nothing to install: the draft path wraps the full-attention layers per step."""


class StreamingPath(SparseKVPath):
    """MagicDec's draft path: full-attention layers read the attention sinks and a recent window
    (:func:`~.path.streaming_keep`)."""

    layer_independent = True

    def __init__(self, adapter: ArchAdapter, sinks: int, window: int, view: str = "gathered") -> None:
        super().__init__(adapter, view)
        self.sinks, self.window = sinks, window

    def keep(self, index: int, mask: torch.Tensor, ctx: LayerContext) -> torch.Tensor | None:
        return streaming_keep(ctx, mask.shape[-1], self.sinks, self.window, mask.device)

    def budget(self, index: int, mask: torch.Tensor, ctx: LayerContext) -> DraftBudget | None:
        return streaming_budget(ctx, self.sinks, self.window, mask.device)


@register_technique
class MagicDecTechnique(_SparseDraftTechnique):
    """MagicDec: the base model drafts on a StreamingLLM view of its own KV cache.

    A draft step runs every layer with the base weights. Each full-attention layer attends to the first
    ``sinks`` positions and the last ``budget - sinks`` positions up to the query (and, in a draft tree, to the
    query's ancestors); sliding-window, chunked and recurrent layers run unchanged. The drafted positions write
    their K/V into the shared cache, where the next exact pass, which reads the whole cache, overwrites them, so
    greedy decoding reproduces the base model's output exactly.

    MagicDec (Sadhukhan et al., 2025) keeps a separate, compacted StreamingLLM cache for the drafter. Here the draft
    reads the model's own cache: by default each full-attention layer gathers the ``budget`` entries of its view
    (:class:`~.views.GatheredView`), so its attention FLOPs and KV reads per draft token stay at ``budget`` entries
    whatever the context length, on any attention implementation. No KV memory is saved. ``view="masked"`` masks the
    other entries instead (the numerical reference), which saves their reads only on economize attention. The
    technique adds no parameters.
    """

    name = MagicDecConfig.technique
    config_class = MagicDecConfig
    config: MagicDecConfig

    def __init__(self, config: MagicDecConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)

    def drafter(self, runner: LayerRunner) -> MagicDecDrafter:
        from .drafter import MagicDecDrafter

        path = StreamingPath(self.adapter, self.config.sinks, self.config.window, self.config.view)
        return MagicDecDrafter(path, runner)

    def extra_repr(self) -> str:
        return f"budget={self.config.budget}, sinks={self.config.sinks}, view={self.config.view!r}"


@register_technique
class TriForceTechnique(_SparseDraftTechnique):
    """TriForce: the base model drafts on the prompt chunks its query attends to.

    The first draft step of a generation runs the full model and picks, per full-attention layer and KV head, the
    first prompt chunk and the ``budget // chunk_size - 1`` chunks whose mean keys score highest against the step's
    query (see :func:`~.retrieval.retrieval_keep`). Every later draft step reads only those chunks, the prompt slots
    after the last whole chunk and the generated positions. The exact pass reads the whole cache, so greedy decoding
    reproduces the base model's output exactly.

    TriForce adds a small draft model with a StreamingLLM cache below this level; that hierarchy is not built here
    (combine the retrieval drafter with a tree policy for longer drafts instead). The selection is made once per
    generation from the first generated token's query, while TriForce rebuilds its cache from the last prompt
    token's query (TriForce's retrieval level, Sun et al., 2024). By default the later draft steps gather the selected
    chunks and the tail (:class:`~.views.GatheredView`), so their attention reads ``budget`` prompt entries plus the
    tail whatever the prompt length, on any attention implementation; ``view="masked"`` masks the other entries
    instead (the numerical reference, a saving only on economize attention). The technique adds no parameters.
    """

    name = TriForceConfig.technique
    config_class = TriForceConfig
    config: TriForceConfig

    def __init__(self, config: TriForceConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)

    def simulation(self) -> Simulation | None:
        """As for MagicDec, except that on a latent cache (multi-head latent attention) the gathered view masks too
        (:data:`LATENT_READS`): the selection is per KV head, and the cache shares every entry between the heads."""
        if _latent_cache(self.adapter) and not _skips_masked_keys(self.adapter):
            return LATENT_READS
        return super().simulation()

    def drafter(self, runner: LayerRunner) -> TriForceDrafter:
        from .drafter import TriForceDrafter
        from .retrieval import RetrievalPath

        path = RetrievalPath(self.adapter, self.config.chunk_size, self.config.select_sets, self.config.view)
        return TriForceDrafter(path, runner)

    def extra_repr(self) -> str:
        return f"budget={self.config.budget}, chunk_size={self.config.chunk_size}, view={self.config.view!r}"
