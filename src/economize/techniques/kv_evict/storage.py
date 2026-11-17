"""How an eviction cache stores the entries it keeps (:data:`EVICTION_STORAGES`).

``"compact"`` (the default for positional methods) keeps the sinks and a ring of the recent entries of every
covered layer in buffers bounded by the budget (:class:`~.compact.CompactWindowCache`), so evicted entries free
their memory. ``"masked"`` keeps every entry in the dense cache and hides the evicted ones from attention: the
reference that score-based methods need, simulated (:class:`~economize.Simulation`). A new storage is registered
as an :class:`EvictionStorage`::

    EVICTION_STORAGES.register("paged", EvictionStorage("...", None, make_paged_cache, positional_only=True))
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...core.metadata import Saving, Simulation
from ...registry import Registry

if TYPE_CHECKING:
    from ...runtime import KVCache
    from .technique import EvictionCache, KVEvictTechnique

__all__ = ["EVICTION_STORAGES", "EvictionStorage"]


@dataclass(frozen=True)
class EvictionStorage:
    """One way of storing the kept entries.

    Attributes:
        description: What the storage keeps.
        simulation: How it simulates the saving, ``None`` when evicted entries free their memory.
        factory: The cache of one generation: ``factory(inner, owner)``.
        positional_only: Whether only :attr:`positional <.policies.EvictionMethod.positional>` methods can use it.
    """

    description: str
    simulation: Simulation | None
    factory: Callable[[KVCache, KVEvictTechnique], EvictionCache]
    positional_only: bool = False


def _masked(inner: KVCache, owner: KVEvictTechnique) -> EvictionCache:
    from .technique import EvictionCache, WindowEvictionCache

    return WindowEvictionCache(inner, owner) if owner.method.positional else EvictionCache(inner, owner)


def _compact(inner: KVCache, owner: KVEvictTechnique) -> EvictionCache:
    from .compact import CompactWindowCache

    return CompactWindowCache(inner, owner)


EVICTION_STORAGES: Registry[EvictionStorage] = Registry("KV eviction storage", "economize.kv_eviction_storages")
EVICTION_STORAGES.register(
    "masked",
    EvictionStorage(
        "every entry stays in the dense cache; attention masks the evicted ones",
        Simulation(Saving.KV_MEMORY | Saving.COMPUTE, "masks the evicted entries, which stay in the cache"),
        _masked,
    ),
)
EVICTION_STORAGES.register(
    "compact",
    EvictionStorage(
        "the sinks and a ring of the recent entries in buffers bounded by the budget",
        None,
        _compact,
        positional_only=True,
    ),
)
