"""The polymorphic copy and state-edit interface of :class:`~economize.runtime.KVCache`.

* :meth:`~economize.runtime.KVCache.detached` and :meth:`~economize.runtime.KVCache.clone` rebuild a cache of the same
  class from every tensor store (:meth:`~economize.runtime.KVCache.tensor_stores`), so the RL teacher's
  ``detached_cache`` needs no ``isinstance`` check on concrete cache classes.
* A class opts in through the ``copyable`` capability; subclasses do not inherit it (a static cache or a technique's
  cache wrapper holds more than its stores), so they raise :class:`TypeError`.
* :meth:`~economize.runtime.KVCache.put_recurrent_state` replaces a layer's recurrent states on caches that declare
  ``supports_state_edit`` and raises :class:`TypeError` elsewhere.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from economize.runtime.cache import CacheSpec, HybridCache, KVCache, PositionalKVCache, StaticKVCache
from economize.techniques._kv import CacheWrapper
from economize.train.rl import detached_cache

BATCH, CAPACITY = 2, 6


def _positional() -> PositionalKVCache:
    """A positional cache with two layers written at slots 0..2, its buffers requiring gradients."""
    cache = PositionalKVCache(CacheSpec(CAPACITY, BATCH))
    for layer in (0, 1):
        keys = torch.randn(BATCH, 1, 3, 4, requires_grad=True)
        with cache.writing_at(torch.arange(3)):
            cache.update(keys * 1.0, keys * 2.0, layer)
    return cache


def _hybrid() -> HybridCache:
    """A hybrid cache with an attention layer (0) and a recurrent layer (1) whose states require gradients."""
    cache = HybridCache(CacheSpec(CAPACITY, BATCH, "hybrid"))
    keys = torch.randn(BATCH, 1, 3, 4, requires_grad=True)
    with cache.writing_at(torch.arange(3)):
        cache.update(keys * 1.0, keys * 2.0, 0)
    conv = torch.randn(BATCH, 3, 4, requires_grad=True)
    recurrent = torch.randn(BATCH, 2, 2, requires_grad=True)
    cache.put_recurrent_state(1, conv * 1.0, recurrent * 1.0)
    return cache


CACHES: dict[str, Callable[[], KVCache]] = {"positional": _positional, "hybrid": _hybrid}


def _stores(cache: KVCache) -> list[tuple[int, int, torch.Tensor]]:
    """``(store index, layer, tensor)`` of every tensor of ``cache``."""
    return [(i, layer, t) for i, store in enumerate(cache.tensor_stores()) for layer, t in sorted(store.items())]


@pytest.mark.parametrize("make", CACHES.values(), ids=CACHES.keys())
def test_detached_shares_every_tensor_without_history(make: Callable[[], KVCache]) -> None:
    cache = make()
    copy = cache.detached()
    assert type(copy) is type(cache) and copy.spec == cache.spec and copy.batch_size == cache.batch_size
    pairs = list(zip(_stores(cache), _stores(copy), strict=True))
    assert len(pairs) >= 4
    for (i, layer, original), (j, copy_layer, detached) in pairs:
        assert (i, layer) == (j, copy_layer)
        assert original.requires_grad and not detached.requires_grad
        assert detached.data_ptr() == original.data_ptr()  # no copy


@pytest.mark.parametrize("make", CACHES.values(), ids=CACHES.keys())
def test_clone_copies_every_tensor_and_keeps_the_graph(make: Callable[[], KVCache]) -> None:
    cache = make()
    copy = cache.clone()
    assert type(copy) is type(cache) and copy.spec == cache.spec
    for (_, _, original), (_, _, cloned) in zip(_stores(cache), _stores(copy), strict=True):
        assert torch.equal(cloned, original) and cloned.data_ptr() != original.data_ptr()
        assert cloned.grad_fn is not None  # gradients flow back to the original
    stored = next(iter(copy.tensor_stores()[0].values()))
    stored.zero_()
    assert next(iter(cache.tensor_stores()[0].values())).abs().sum() > 0


def test_hybrid_copies_include_the_recurrent_states() -> None:
    cache = _hybrid()
    for copy in (cache.detached(), cache.clone()):
        conv, recurrent = copy.recurrent_state(1)
        expected_conv, expected_recurrent = cache.recurrent_state(1)
        assert conv is not None and recurrent is not None
        assert torch.equal(conv, expected_conv) and torch.equal(recurrent, expected_recurrent)
        assert copy.recurrent_state(0) == (None, None)


def test_detached_cache_delegates_to_the_cache() -> None:
    cache = _hybrid()
    copy = detached_cache(cache)
    assert type(copy) is HybridCache and all(not t.requires_grad for _, _, t in _stores(copy))


def test_the_copy_capability_is_not_inherited() -> None:
    class Derived(PositionalKVCache):
        pass

    assert PositionalKVCache.copyable and HybridCache.copyable
    assert not KVCache.copyable and not StaticKVCache.copyable and not Derived.copyable


def test_static_caches_and_wrappers_refuse_to_copy() -> None:
    static = StaticKVCache(CacheSpec(CAPACITY, BATCH))
    wrapper = CacheWrapper(_positional())
    for cache in (static, wrapper):
        with pytest.raises(TypeError, match="copyable"):
            cache.detached()
        with pytest.raises(TypeError, match="copyable"):
            cache.clone()


def test_put_recurrent_state_replaces_and_drops_states() -> None:
    cache = _hybrid()
    assert HybridCache.supports_state_edit and HybridCache.keeps_recurrent_state
    new = torch.ones(BATCH, 2, 2)
    cache.put_recurrent_state(1, None, new)
    conv, recurrent = cache.recurrent_state(1)
    assert conv is None and recurrent is new
    assert 1 not in cache.conv_states


def test_put_recurrent_state_inside_selecting_wins_over_the_view() -> None:
    """A state put while rows are selected replaces the handed-out view, which is not written back."""
    cache = _hybrid()
    with cache.selecting(torch.tensor([0])):
        _, view = cache.recurrent_state(1)
        assert view is not None
        view.fill_(5.0)
        cache.put_recurrent_state(1, None, torch.zeros(BATCH, 2, 2))
    assert torch.equal(cache.recurrent_states[1], torch.zeros(BATCH, 2, 2))


def test_caches_without_recurrent_states_refuse_state_edits() -> None:
    cache = _positional()
    assert not PositionalKVCache.supports_state_edit
    assert cache.recurrent_state(0) == (None, None)
    with pytest.raises(TypeError, match="no recurrent states"):
        cache.put_recurrent_state(0, None, torch.zeros(BATCH, 2))
