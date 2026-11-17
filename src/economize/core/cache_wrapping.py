"""The KV-cache wrapping of a decoder: every KV-cache technique wraps each generation's cache, in attach order."""

from __future__ import annotations

from collections.abc import Sequence

from ..runtime import KVCache, StaticKVCache
from .technique import Technique

__all__ = ["CacheWrapping", "NestedCacheWrapping"]


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
        return self._wrap(cache, reuse=cache is self._base, static=isinstance(cache, StaticKVCache))

    def _wrap(self, cache: KVCache, reuse: bool, static: bool) -> KVCache:
        """Wrap ``cache`` with every technique, handing each its wrapper of the previous generation if ``reuse``,
        and keep the new wrappers for the next generation if ``static`` (the cache of CUDA graphs)."""
        previous: Sequence[KVCache | None] = self._wrappers if reuse else [None] * len(self.techniques)
        wrappers: list[KVCache] = []
        inner = cache
        for technique, before in zip(self.techniques, previous, strict=True):
            inner = technique.wrap_cache(inner) if before is None else technique.wrap_cache(inner, previous=before)
            wrappers.append(inner)
        keep = self.graph_safe and static
        self._base, self._wrappers = (cache, wrappers) if keep else (None, [])
        return inner


class NestedCacheWrapping(CacheWrapping):
    """The cache wrapping of techniques that another technique holds (the techniques of a draft view, the drafters
    of a wrapping drafter), as that technique's :meth:`~economize.core.Technique.wrap_cache`.

    The holder is one technique to the decoder's :class:`CacheWrapping`, which passes it its own wrapper of the
    previous generation as ``previous``. When that wrapper was built on the same cache, each held technique gets its
    wrapper from then, as :class:`CacheWrapping` gives it at the top level. Held techniques that wrap no cache
    (:attr:`~economize.core.Technique.wraps_cache` false) are left out.
    """

    def __init__(self, techniques: Sequence[Technique]) -> None:
        super().__init__([t for t in techniques if t.wraps_cache])
        self._output: KVCache | None = None

    def __call__(self, cache: KVCache, previous: KVCache | None = None) -> KVCache:
        """Wrap ``cache`` with every held technique (innermost first) and return the outermost wrapper."""
        reuse = previous is not None and previous is self._output and cache is self._base
        output = self._wrap(cache, reuse, static=isinstance(_innermost(cache), StaticKVCache))
        self._output = output if self._base is not None else None
        return output


def _innermost(cache: KVCache) -> KVCache:
    """The cache under every wrapper of ``cache`` (a wrapper forwards to its ``inner`` cache)."""
    while (inner := getattr(cache, "inner", None)) is not None:
        cache = inner
    return cache
