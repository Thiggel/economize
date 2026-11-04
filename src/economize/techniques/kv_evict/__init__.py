"""KV-cache eviction: StreamingLLM, H2O (during decoding or once at prefill), SnapKV and PyramidKV.

    import economize
    from economize.techniques.kv_evict import KVEvictConfig

    model = economize.apply(hf_model, KVEvictConfig(method="h2o", recent=128, heavy=128))
    result = model.decode(prompt, 64)
    model.techniques["kv_evict"].cost_summary(result)  # layer applications plus KV bytes held and read

The technique wraps every generation's cache, evicts only committed entries at the boundaries of exact passes, and
masks evicted entries out of every attention call, drafts included. New eviction rules are registered in
:data:`EVICTION_METHODS` (see :mod:`.policies`). Lossy and training-free.
"""

from __future__ import annotations

from .policies import (
    EVICTION_METHODS,
    EvictionMethod,
    H2OEviction,
    H2OPrefillEviction,
    PassInfo,
    PyramidKVEviction,
    SnapKVEviction,
    StreamingEviction,
    candidates,
    keep_top,
)
from .tap import AttentionSink, AttentionTap, attention_probs, iter_attention_probs
from .technique import POOLINGS, EvictionCache, KVEvictConfig, KVEvictTechnique, PendingScores, WindowEvictionCache

__all__ = [
    "EVICTION_METHODS",
    "POOLINGS",
    "AttentionSink",
    "AttentionTap",
    "EvictionCache",
    "EvictionMethod",
    "H2OEviction",
    "H2OPrefillEviction",
    "KVEvictConfig",
    "KVEvictTechnique",
    "PassInfo",
    "PendingScores",
    "PyramidKVEviction",
    "SnapKVEviction",
    "StreamingEviction",
    "WindowEvictionCache",
    "attention_probs",
    "candidates",
    "iter_attention_probs",
    "keep_top",
]
