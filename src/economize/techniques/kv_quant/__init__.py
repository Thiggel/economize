"""Quantized KV caches: KIVI (2/4-bit, keys per channel, values per token) and an fp8 cache.

    import economize
    from economize.techniques.kv_quant import KIVIConfig

    model = economize.apply(hf_model, KIVIConfig(bits=2, group_size=32, residual_length=128))
    result = model.decode(prompt, 64)
    model.techniques["kivi"].cost_summary(result)  # layer applications plus KV bytes held and read

Both techniques wrap the cache of every generation (:class:`~economize.techniques._kv.CacheWrapper`), act only on
committed entries, and compose with drafters, dynamic depth and weight techniques. They are lossy and
training-free. The shared base (:mod:`economize.techniques._kv`) also serves the KV eviction techniques of
:mod:`economize.techniques.kv_evict`; it is re-exported here.
"""

from __future__ import annotations

from .._kv import (  # re-exported: the shared KV-cache base lives in economize.techniques._kv
    CacheWrapper,
    KVCacheTechnique,
    KVStats,
    RowStats,
    SlotStore,
    check_positional,
    find_wrapper,
    frontiers,
    innermost,
    layer_indices,
    write_slots,
)
from .fp8 import FP8_FORMATS, FP8_SCALES, FP8Cache, FP8KVConfig, FP8KVTechnique, fp8_round_trip
from .kivi import KIVI_BITS, KIVICache, KIVIConfig, KIVITechnique, fake_quantize, kivi_bytes, quantize

__all__ = [
    "FP8_FORMATS",
    "FP8_SCALES",
    "KIVI_BITS",
    "CacheWrapper",
    "FP8Cache",
    "FP8KVConfig",
    "FP8KVTechnique",
    "KIVICache",
    "KIVIConfig",
    "KIVITechnique",
    "KVCacheTechnique",
    "KVStats",
    "RowStats",
    "SlotStore",
    "check_positional",
    "fake_quantize",
    "find_wrapper",
    "fp8_round_trip",
    "frontiers",
    "innermost",
    "kivi_bytes",
    "layer_indices",
    "quantize",
    "write_slots",
]
