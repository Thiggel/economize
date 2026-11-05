"""The MagicDec and TriForce techniques: lossless self-speculation with the base model on a sparse view of its cache."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from ...core.metadata import Family, Saving, Simulation, TechniqueInfo
from ...core.technique import AttachContext, Capability, Technique, register_technique
from .config import MagicDecConfig, TriForceConfig
from .path import SparseKVPath, sparse_layers, streaming_keep

if TYPE_CHECKING:
    import torch

    from ...arch import ArchAdapter
    from ...runtime import LayerContext, LayerRunner
    from .drafter import MagicDecDrafter, TriForceDrafter

__all__ = ["MagicDecTechnique", "StreamingPath", "TriForceTechnique", "sparse_layers"]

#: How the Hugging Face attention paths simulate a sparse draft: they mask the dropped entries and still read them.
_MASKED_READS = Simulation(Saving.LATENCY, "masks the KV entries outside the draft budget, which the attention "
                           "kernel still reads")


def _skips_masked_keys(adapter: ArchAdapter) -> bool:
    """Whether the model's attention kernel skips the key blocks a mask hides (economize's decode kernel)."""
    return getattr(adapter.config, "_attn_implementation", None) == "economize"


class _SparseDraftTechnique(Technique):
    """What MagicDec and TriForce share: no parameters, nothing to attach, a saving only a key-skipping kernel has."""

    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True, simulated=_MASKED_READS)
    capabilities = Capability.DRAFTER

    def simulation(self) -> Simulation | None:
        """``None`` on economize attention (its decode kernel skips the hidden key blocks), else the masked reads."""
        return None if _skips_masked_keys(self.adapter) else super().simulation()

    def attach(self, model: nn.Module, adapter: ArchAdapter, context: AttachContext | None = None) -> None:
        """Nothing to install: the draft path wraps the full-attention layers per step."""


class StreamingPath(SparseKVPath):
    """MagicDec's draft path: full-attention layers read the attention sinks and a recent window
    (:func:`~.path.streaming_keep`)."""

    layer_independent = True

    def __init__(self, adapter: ArchAdapter, sinks: int, window: int) -> None:
        super().__init__(adapter)
        self.sinks, self.window = sinks, window

    def keep(self, index: int, mask: torch.Tensor, ctx: LayerContext) -> torch.Tensor | None:
        return streaming_keep(ctx, mask.shape[-1], self.sinks, self.window, mask.device)


@register_technique
class MagicDecTechnique(_SparseDraftTechnique):
    """MagicDec: the base model drafts on a StreamingLLM view of its own KV cache.

    MagicDec (Sadhukhan et al., 2025): the base model drafts on a StreamingLLM view of its own KV cache.

    A draft step runs every layer with the base weights. Each full-attention layer attends to the first
    ``sinks`` positions and the last ``budget - sinks`` positions up to the query (and, in a draft tree, to the
    query's ancestors); sliding-window, chunked and recurrent layers run unchanged. The drafted positions write
    their K/V into the shared cache, where the next exact pass, which reads the whole cache, overwrites them, so
    greedy decoding reproduces the base model's output exactly.

    MagicDec keeps a separate, compacted StreamingLLM cache for the drafter; here the draft reads the model's own cache
    through a mask, which saves no KV memory and realizes the bandwidth saving only with economize attention, whose
    decode kernel skips every block of keys the mask hides. The technique adds no parameters.
    """

    name = MagicDecConfig.technique
    config_class = MagicDecConfig
    config: MagicDecConfig

    def __init__(self, config: MagicDecConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)

    def drafter(self, runner: LayerRunner) -> MagicDecDrafter:
        from .drafter import MagicDecDrafter

        return MagicDecDrafter(StreamingPath(self.adapter, self.config.sinks, self.config.window), runner)

    def extra_repr(self) -> str:
        return f"budget={self.config.budget}, sinks={self.config.sinks}"


@register_technique
class TriForceTechnique(_SparseDraftTechnique):
    """TriForce: the base model drafts on the prompt chunks its query attends to.

    TriForce's retrieval level (Sun et al., 2024): the base model drafts on the prompt chunks its query attends to.

    The first draft step of a generation runs the full model and picks, per full-attention layer and KV head, the
    first prompt chunk and the ``budget // chunk_size - 1`` chunks whose mean keys score highest against the step's
    query (see :func:`~.retrieval.retrieval_keep`). Every later draft step reads only those chunks, the prompt slots
    after the last whole chunk and the generated positions. The exact pass reads the whole cache, so greedy decoding
    reproduces the base model's output exactly.

    TriForce adds a small draft model with a StreamingLLM cache below this level; that hierarchy is not built here
    (combine the retrieval drafter with a tree policy for longer drafts instead). The selection is made once per
    generation from the first generated token's query, while TriForce rebuilds its cache from the last prompt
    token's query. The saving is realized with economize attention (see :class:`MagicDecTechnique`); chunks of at
    least the kernel's key block (64 slots) let it skip the unselected chunks. The technique adds no parameters.
    """

    name = TriForceConfig.technique
    config_class = TriForceConfig
    config: TriForceConfig

    def __init__(self, config: TriForceConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)

    def drafter(self, runner: LayerRunner) -> TriForceDrafter:
        from .drafter import TriForceDrafter
        from .retrieval import RetrievalPath

        path = RetrievalPath(self.adapter, self.config.chunk_size, self.config.select_sets)
        return TriForceDrafter(path, runner)

    def extra_repr(self) -> str:
        return f"budget={self.config.budget}, chunk_size={self.config.chunk_size}"
