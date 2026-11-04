"""Row-wise snapshots and row selection of :class:`~economize.runtime.HybridCache` on layers without a state.

A recurrent layer's state exists for the whole batch once the layer has seen tokens. Restoring the snapshot for some
rows, or writing a state inside :meth:`~economize.runtime.HybridCache.selecting`, cannot create the state of the other
rows, so both raise a clear error instead of replacing every row or failing with a ``KeyError``.
"""

from __future__ import annotations

import pytest
import torch

from economize.runtime.cache import CacheSpec, HybridCache


def _cache(batch: int = 3) -> HybridCache:
    return HybridCache(CacheSpec(8, batch, "hybrid"))


def test_restore_of_some_rows_keeps_the_other_rows() -> None:
    cache = _cache()
    cache.update_recurrent_state(torch.zeros(3, 2), 0)
    snapshot = cache.snapshot()
    cache.update_recurrent_state(torch.ones(3, 2), 0)
    cache.restore(snapshot, torch.tensor([1]))
    assert cache.recurrent_states[0][:, 0].tolist() == [1.0, 0.0, 1.0]


def test_restore_of_some_rows_zeroes_a_layer_the_snapshot_had_not_seen() -> None:
    cache = _cache()
    snapshot = cache.snapshot()
    cache.update_recurrent_state(torch.ones(3, 2), 0)
    cache.restore(snapshot, torch.tensor([0, 2]))
    assert cache.recurrent_states[0][:, 0].tolist() == [0.0, 1.0, 0.0]


def test_restore_of_some_rows_raises_for_a_layer_the_cache_lacks() -> None:
    cache = _cache()
    cache.update_recurrent_state(torch.ones(3, 2), 0)
    snapshot = cache.snapshot()
    cache.restore({}, None)  # every state dropped
    with pytest.raises(ValueError, match="layer 0"):
        cache.restore(snapshot, torch.tensor([1]))
    cache.restore(snapshot)  # every row: the whole state comes back
    assert torch.equal(cache.recurrent_states[0], torch.ones(3, 2))


def test_writing_a_new_layer_inside_selecting_raises_a_clear_error() -> None:
    cache = _cache()
    with pytest.raises(ValueError, match="whole batch first"), cache.selecting(torch.tensor([0, 2])):
        cache.update_recurrent_state(torch.ones(2, 2), 0)
    with pytest.raises(ValueError, match="whole batch first"), cache.selecting(torch.tensor([1])):
        cache.update_conv_state(torch.ones(1, 4, 3), 1)


def test_writing_a_known_layer_inside_selecting_updates_the_selected_rows() -> None:
    cache = _cache()
    cache.update_recurrent_state(torch.zeros(3, 2), 0)
    with cache.selecting(torch.tensor([0, 2])):
        cache.update_recurrent_state(torch.ones(2, 2), 0)
    assert cache.recurrent_states[0][:, 0].tolist() == [1.0, 0.0, 1.0]
