"""Long-context self-speculation: the base model drafts on a sparse view of its own KV cache (MagicDec, TriForce).

    import economize
    from economize import DecodeConfig
    from economize.techniques.sparse_kv_draft import MagicDecConfig, TriForceConfig

    model = economize.apply(hf_model, MagicDecConfig(budget=512, sinks=4))           # StreamingLLM draft view
    model = economize.apply(hf_model, TriForceConfig(budget=4096, chunk_size=64))    # retrieval draft view
    result = model.decode(long_prompt, 256, decoding=DecodeConfig(cycle_length=5))

A draft step runs every layer with the base weights; the full-attention layers read only the entries of the draft
budget, the other layers run unchanged. Verification reads the whole cache, so greedy decoding is lossless. By
default (``view="gathered"``) a draft layer gathers the budget's K/V entries and attends over those alone, so its
attention FLOPs and KV reads scale with the budget on every attention implementation. ``view="masked"`` is the
numerical reference: it hides the other entries in the mask, which saves their reads only on economize attention.
New budgets subclass :class:`SparseKVPath`, new ways of reading them :class:`DraftView` (registered in
:data:`DRAFT_VIEWS`).
"""

from __future__ import annotations

from .config import MagicDecConfig, TriForceConfig
from .drafter import MagicDecDrafter, TriForceDrafter
from .path import SparseKVPath, restrict, sparse_layers, streaming_budget, streaming_keep
from .retrieval import RetrievalPath, chunk_slots, retrieval_keep, tail_width
from .technique import MagicDecTechnique, StreamingPath, TriForceTechnique
from .views import DRAFT_VIEWS, DraftBudget, DraftView, GatheredView, MaskedView, gather_entries

__all__ = [
    "DRAFT_VIEWS",
    "DraftBudget",
    "DraftView",
    "GatheredView",
    "MagicDecConfig",
    "MagicDecDrafter",
    "MagicDecTechnique",
    "MaskedView",
    "RetrievalPath",
    "SparseKVPath",
    "StreamingPath",
    "TriForceConfig",
    "TriForceDrafter",
    "TriForceTechnique",
    "chunk_slots",
    "gather_entries",
    "restrict",
    "retrieval_keep",
    "sparse_layers",
    "streaming_budget",
    "streaming_keep",
    "tail_width",
]
