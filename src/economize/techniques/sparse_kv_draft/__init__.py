"""Long-context self-speculation: the base model drafts on a sparse view of its own KV cache (MagicDec, TriForce).

    import economize
    from economize import DecodeConfig
    from economize.techniques.sparse_kv_draft import MagicDecConfig, TriForceConfig

    model = economize.apply(hf_model, MagicDecConfig(budget=512, sinks=4))           # StreamingLLM draft view
    model = economize.apply(hf_model, TriForceConfig(budget=4096, chunk_size=64))    # retrieval draft view
    result = model.decode(long_prompt, 256, decoding=DecodeConfig(cycle_length=5))

A draft step runs every layer with the base weights; the full-attention layers read only the entries of the view,
the other layers run unchanged. Verification reads the whole cache, so greedy decoding is lossless. Load the model
with ``attn_implementation="economize"`` to turn the hidden entries into skipped reads; the sdpa and eager paths
mask them and only simulate the saving. New views subclass :class:`SparseKVPath`.
"""

from __future__ import annotations

from .config import MagicDecConfig, TriForceConfig
from .drafter import MagicDecDrafter, TriForceDrafter
from .path import SparseKVPath, restrict, sparse_layers, streaming_keep
from .retrieval import RetrievalPath, retrieval_keep
from .technique import MagicDecTechnique, StreamingPath, TriForceTechnique

__all__ = [
    "MagicDecConfig",
    "MagicDecDrafter",
    "MagicDecTechnique",
    "RetrievalPath",
    "SparseKVPath",
    "StreamingPath",
    "TriForceConfig",
    "TriForceDrafter",
    "TriForceTechnique",
    "restrict",
    "retrieval_keep",
    "sparse_layers",
    "streaming_keep",
]
