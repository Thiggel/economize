"""A static KV cache whose smaller batch sizes are views into one pool (batched serving under CUDA graphs).

:class:`~economize.runtime.StaticKVCache` keeps one full buffer set per batch size. In batched decoding rows leave the
batch as they finish, so a generation of ``B`` rows visits up to ``B`` batch sizes and the cache grows to about
``B (B + 1) / 2`` rows, which does not fit next to a large model already at ``B = 16``.

:class:`PooledStaticKVCache` allocates every layer once for the largest batch size it has seen (the *pool*) and
serves a batch of ``k`` rows from the views ``pool[:k]``. A view of a fixed pool keeps its storage, so the CUDA
graphs recorded on it stay valid; graphs of different batch sizes share the pool. A batch larger than the pool
allocates a new pool and keeps the old one alive for the graphs recorded on it.

The variant is opt-in: :func:`use_pooled_static_cache` registers :class:`PooledPositionalKVCache` (an eager cache
that behaves exactly like :class:`~economize.runtime.PositionalKVCache`) under ``"positional"``, whose
:attr:`~economize.runtime.KVCache.static_variant` is the pooled cache. Decoders built afterwards use it.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from ..errors import ConfigError
from ..validation import check_positive
from .cache import CACHES, CacheSpec, KVCache, PositionalKVCache, StaticKVCache

__all__ = ["PooledPositionalKVCache", "PooledStaticKVCache", "use_pooled_static_cache"]

_Pool = tuple[int, dict[int, torch.Tensor], dict[int, torch.Tensor]]  # rows, keys, values


class PooledStaticKVCache(StaticKVCache):
    """A :class:`~economize.runtime.StaticKVCache` that serves every batch size from prefix views of one pool.

    Memory is one buffer set of the largest batch size, independent of how many batch sizes a generation visits.
    """

    def __init__(self, spec: CacheSpec) -> None:
        super().__init__(spec)
        self._pools: list[_Pool] = [(spec.batch_size, {}, {})]

    @property
    def pool_rows(self) -> int:
        """Rows of the current (largest) pool."""
        return self._pools[-1][0]

    def activate(self, batch_size: int) -> None:
        """Use the first ``batch_size`` rows of the pool from now on (a larger batch gets a new pool)."""
        check_positive("batch size", batch_size)
        if batch_size > self.pool_rows:
            self._pools.append((batch_size, {}, {}))  # the old pool stays alive for its graphs
            self._sets = {}
        _, pool_keys, pool_values = self._pools[-1]
        keys, values = self._sets.setdefault(batch_size, ({}, {}))
        for layer in pool_keys:
            if layer not in keys:
                keys[layer] = pool_keys[layer][:batch_size]
                values[layer] = pool_values[layer][:batch_size]
        self.keys, self.values = keys, values
        self.batch_size = batch_size

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx not in self.keys and self._rows is None:
            self._bind(layer_idx, key_states, value_states)
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)

    def reorder(self, rows: torch.Tensor) -> None:
        if len(rows) == self.batch_size:
            super().reorder(rows)  # same size: copies in place
            return
        selected = [
            (layer, keys.index_select(0, index), self.values[layer].index_select(0, index))
            for layer, keys in self.keys.items()
            for index in (rows.to(keys.device),)
        ]  # copies first: the smaller views overlap the current ones
        self.activate(len(rows))
        for layer, keys, values in selected:
            if layer not in self.keys:
                self._bind(layer, keys, values)
            self.keys[layer].copy_(keys)
            self.values[layer].copy_(values)

    def _bind(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """Allocate ``layer_idx`` in the pool (shaped like these K/V) and add its views to every set."""
        rows, pool_keys, pool_values = self._pools[-1]
        if key_states.shape[0] != self.batch_size:
            raise ConfigError(f"the cache holds {self.batch_size} rows but a layer wrote {key_states.shape[0]}")
        pool_keys[layer_idx] = self._pooled(key_states, rows)
        pool_values[layer_idx] = self._pooled(value_states, rows)
        for size, (keys, values) in self._sets.items():
            keys[layer_idx] = pool_keys[layer_idx][:size]
            values[layer_idx] = pool_values[layer_idx][:size]

    def _pooled(self, like: torch.Tensor, rows: int) -> torch.Tensor:
        return like.new_zeros(rows, like.shape[1], self.num_slots, *like.shape[3:])


class PooledPositionalKVCache(PositionalKVCache):
    """:class:`~economize.runtime.PositionalKVCache` whose CUDA-graph variant is :class:`PooledStaticKVCache`."""

    copyable: ClassVar[bool] = True
    static_variant: ClassVar[type[KVCache] | None] = PooledStaticKVCache


PooledStaticKVCache.static_variant = PooledStaticKVCache


def use_pooled_static_cache(enabled: bool = True) -> None:
    """Make graphed decoders of ``"positional"`` caches use :class:`PooledStaticKVCache` (``False`` restores the
    default). Decoders built before the call keep their cache class."""
    CACHES.register("positional", PooledPositionalKVCache if enabled else PositionalKVCache, override=True)
