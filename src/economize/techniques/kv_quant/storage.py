"""How a KIVI cache stores its entries (:data:`KIVI_STORAGES`).

``"packed"`` (the default) keeps the quantized entries of every covered layer as packed integer codes with fp16
scales and zero points, and the full-precision residual in a ring of ``residual_length + headroom`` slots
(:class:`~.packed.PackedKIVICache`): the inner cache never allocates a dense buffer for these layers, so the
memory the cache holds shrinks by the quantization ratio. ``"simulated"`` quantizes in place inside the dense
buffers (:class:`~.kivi.KIVICache`): the same numerics with the full dense memory, the reference that other KV
techniques can wrap. A new storage is registered as a :class:`KIVIStorage`::

    KIVI_STORAGES.register("paged", KIVIStorage("...", None, make_paged_cache))
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...core.metadata import Saving, Simulation
from ...registry import Registry

if TYPE_CHECKING:
    from ...runtime import KVCache
    from .kivi import KIVICache, KIVITechnique

__all__ = ["KIVI_STORAGES", "KIVIStorage"]


@dataclass(frozen=True)
class KIVIStorage:
    """One way of storing a KIVI cache.

    Attributes:
        description: What the storage keeps.
        simulation: How it simulates the saving, ``None`` when the cache really holds the quantized bytes.
        factory: The cache of one generation: ``factory(inner, owner)``.
    """

    description: str
    simulation: Simulation | None
    factory: Callable[[KVCache, KIVITechnique], KIVICache]


def _simulated(inner: KVCache, owner: KIVITechnique) -> KIVICache:
    from .._kv import layer_indices
    from .kivi import KIVICache

    return KIVICache(inner, owner.config, layer_indices(owner.config.layers, owner.adapter))


def _packed(inner: KVCache, owner: KIVITechnique) -> KIVICache:
    from .._kv import layer_indices
    from .packed import PackedKIVICache

    config = owner.config
    layers = layer_indices(config.layers, owner.adapter)
    return PackedKIVICache(inner, config, layers, reads_packed=owner.reads_packed())


KIVI_STORAGES: Registry[KIVIStorage] = Registry("KIVI storage", "economize.kivi_storages")
KIVI_STORAGES.register(
    "simulated",
    KIVIStorage(
        "dense buffers; the quantized entries are written back dequantized",
        Simulation(Saving.KV_MEMORY, "writes the dequantized entries back into the dense cache"),
        _simulated,
    ),
)
KIVI_STORAGES.register(
    "packed",
    KIVIStorage(
        "packed low-bit codes with fp16 scales and zero points, and a full-precision ring of the recent entries",
        None,
        _packed,
    ),
)
