"""Common base of the techniques that wrap the KV cache (quantized, evicting)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ...core.technique import Capability, Technique
from ...errors import ConfigError, UnsupportedModelError
from ...runtime import KVCache
from .wrapper import CacheWrapper, KVStats

__all__ = ["KVCacheTechnique", "check_positional", "layer_indices"]


class KVCacheTechnique(Technique, ABC):
    """A technique whose whole effect is a :class:`~.wrapper.CacheWrapper` around every generation's cache.

    It keeps the memory statistics of the last generation (:meth:`kv_stats`) and merges them into a result's cost
    summary (:meth:`cost_summary`). Subclasses implement :meth:`make_wrapper` and declare their family
    (``Family.KV_QUANTIZATION``, ``Family.KV_EVICTION``, ...): two techniques of one family cannot be combined.
    """

    capabilities: ClassVar[Capability] = Capability.KV_CACHE
    #: Whether the wrapper works under CUDA graphs: every tensor it hands to a layer stays at a fixed address while
    #: the cache lives, what recorded kernels read beyond that is reported by ``own_graph_state``, and the wrapper can
    #: :meth:`~.wrapper.CacheWrapper.restart` (:meth:`wrap_cache` reuses it). Its pass boundaries then run outside the
    #: graph (``external_pass``).
    graph_safe: ClassVar[bool] = False
    #: The :class:`~economize.attention.PackedKV` format the wrapper attaches to the keys it returns, if any. A key
    #: tensor carries one attachment, so a technique that attaches its own (``quest``) refuses the others.
    packed_format: ClassVar[str | None] = None

    def __init__(self, config: Any, adapter: Any) -> None:
        super().__init__(config, adapter)
        self.last_stats = KVStats()

    @abstractmethod
    def make_wrapper(self, cache: KVCache) -> CacheWrapper:
        """The wrapper of one generation's cache."""

    def wrap_cache(self, cache: KVCache, previous: KVCache | None = None) -> KVCache:
        """The wrapper of a generation's ``cache``. A graph-safe technique :meth:`restarts
        <.wrapper.CacheWrapper.restart>` its wrapper of the previous generation over the same static cache
        (``previous``) instead of building a new one, so that graphs recorded over it stay valid."""
        if self.graph_safe and isinstance(previous, CacheWrapper) and previous.inner is cache:
            previous.restart()
            wrapper = previous
        else:
            wrapper = self.make_wrapper(cache)
        self.last_stats = wrapper.stats
        return wrapper

    def kv_stats(self, row: int = 0) -> dict[str, float]:
        """Memory statistics of row ``row`` of the last generation (see :class:`~.wrapper.RowStats`)."""
        return self.last_stats.row(row).as_dict()

    def cost_summary(self, result: Any, row: int = 0) -> dict[str, Any]:
        """``result.cost_summary()`` of a :class:`~economize.decoding.DecodeResult` of the last generation (row
        ``row`` of its batch) together with this technique's memory statistics."""
        return {**result.cost_summary(), **self.kv_stats(row)}


def check_positional(adapter: Any, technique: str) -> None:
    """Raise :class:`~economize.UnsupportedModelError` for models with recurrent layers (hybrid caches)."""
    if adapter is not None and getattr(adapter, "cache_kind", "positional") != "positional":
        raise UnsupportedModelError(f"{technique} needs attention layers only; this model keeps recurrent states")


def layer_indices(layers: tuple[int, ...], adapter: Any) -> frozenset[int] | None:
    """Cache indices (0-based ``layer_idx``) of the 1-indexed ``layers``, or ``None`` for every layer."""
    if not layers:
        return None
    if adapter is not None and any(i > adapter.num_layers for i in layers):
        raise ConfigError(f"layers {layers} outside 1..{adapter.num_layers}")
    return frozenset(i - 1 for i in layers)
