"""The KV-cache wrapping of a decoder: every KV-cache technique wraps each generation's cache, in attach order."""

from __future__ import annotations

from collections.abc import Sequence

from ..runtime import KVCache, StaticKVCache
from .technique import Technique

__all__ = ["CacheWrapping"]


class CacheWrapping:
    """Wraps the cache of every generation of one decoder with ``techniques`` (innermost first).

    A decoder that records CUDA graphs runs every generation on the same static cache. When a generation brings
    that cache again, each technique gets its wrapper of the previous generation (``previous`` of
    :meth:`~economize.core.Technique.wrap_cache`), so that graph-safe wrappers are restarted and the graphs recorded
    over them stay valid. Wrappers of other caches are not kept: an eager generation's cache is freed with its
    generation.

    Attributes:
        graph_safe: Whether every technique allows CUDA graphs over its wrapper (read by
            :class:`~economize.decoding.CycleDecoder`).
    """

    def __init__(self, techniques: Sequence[Technique]) -> None:
        self.techniques = tuple(techniques)
        self.graph_safe = all(t.graph_safe for t in self.techniques)
        self._base: KVCache | None = None
        self._wrappers: list[KVCache] = []

    def __call__(self, cache: KVCache) -> KVCache:
        previous: Sequence[KVCache | None] = self._wrappers if cache is self._base else [None] * len(self.techniques)
        wrappers: list[KVCache] = []
        inner = cache
        for technique, before in zip(self.techniques, previous, strict=True):
            inner = technique.wrap_cache(inner) if before is None else technique.wrap_cache(inner, previous=before)
            wrappers.append(inner)
        keep = self.graph_safe and isinstance(cache, StaticKVCache)
        self._base, self._wrappers = (cache, wrappers) if keep else (None, [])
        return inner
