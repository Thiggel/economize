"""The base shared by the techniques that wrap the KV cache (``kv_quant``, ``kv_evict``).

* :mod:`.wrapper`: :class:`CacheWrapper` (a cache that forwards to an inner cache and acts at the boundaries of
  exact passes, with the memory accounting of its layers), :class:`SlotStore` (per-slot side tensors) and the
  per-row :class:`KVStats`.
* :mod:`.base`: :class:`KVCacheTechnique`, the technique that wraps every generation's cache, plus the config
  checks :func:`check_positional` and :func:`layer_indices`.

A new KV-cache technique subclasses both and registers the technique with a ``TechniqueInfo`` of its family; two
techniques of one family (two quantizers, two evictions) cannot be combined.
"""

from __future__ import annotations

from .base import KVCacheTechnique, check_positional, layer_indices
from .wrapper import (
    CacheWrapper,
    KVStats,
    LayerBytes,
    RowStats,
    SlotStore,
    find_wrapper,
    frontiers,
    innermost,
    write_slots,
)

__all__ = [
    "CacheWrapper",
    "KVCacheTechnique",
    "KVStats",
    "LayerBytes",
    "RowStats",
    "SlotStore",
    "check_positional",
    "find_wrapper",
    "frontiers",
    "innermost",
    "layer_indices",
    "write_slots",
]
