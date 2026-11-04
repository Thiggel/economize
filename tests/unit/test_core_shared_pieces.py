"""Pieces shared across the runtime, the drafters and the calibrating techniques: the slot layout of K/V buffers,
the recurrent-state interface of every cache, and the calibration observers."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from economize.arch import adapter_for
from economize.core.calibration import (
    AbsMeanObserver,
    FeatureMeanObserver,
    SquaredNormObserver,
    calibration_source,
    layer_device,
    linear_modules,
)
from economize.core.technique import AttachContext
from economize.errors import ConfigError
from economize.modules.kv_cache import DraftKVCache, SlotLayout, write_slots
from economize.runtime import CACHES, HybridCache, LayerRunner, PositionalKVCache, StaticKVCache


def test_write_slots_shares_one_scatter_for_shared_and_per_row_slots() -> None:
    buffer = torch.zeros(2, 1, 5, 3, dtype=torch.float16)
    states = torch.arange(2 * 1 * 2 * 3, dtype=torch.float32).view(2, 1, 2, 3)
    write_slots(buffer, torch.tensor([1, 3]), states)
    torch.testing.assert_close(buffer[:, :, [1, 3]], states.half())
    buffer.zero_()
    write_slots(buffer, torch.tensor([[0, 4], [2, 4]]), states)
    torch.testing.assert_close(buffer[0, :, [0, 4]], states[0].half())
    torch.testing.assert_close(buffer[1, :, [2, 4]], states[1].half())
    assert buffer.dtype is torch.float16


def test_runtime_and_draft_caches_share_the_slot_layout(tiny_model) -> None:
    assert issubclass(PositionalKVCache, SlotLayout) and issubclass(DraftKVCache, SlotLayout)
    cache = LayerRunner(adapter_for(tiny_model)).new_cache(16)
    assert (cache.trash_slot, cache.num_slots) == (16, 17)


def test_every_cache_answers_snapshot_and_restore(tiny_model, prompt: Tensor, monkeypatch) -> None:
    """A cache without recurrent state snapshots nothing, and the runner's replay leaves it alone."""
    assert not PositionalKVCache.keeps_recurrent_state and HybridCache.keeps_recurrent_state
    runner = LayerRunner(adapter_for(tiny_model))
    cache = runner.new_cache(32)
    snapshot = cache.snapshot()
    assert snapshot == {}
    cache.restore(snapshot, torch.tensor([0]))
    monkeypatch.setattr(runner, "forward", lambda *a, **k: pytest.fail("replay ran a pass on a positional cache"))
    runner.replay(cache, snapshot, prompt[None, :4], torch.tensor([0]), torch.tensor([2]))


def test_feature_mean_observers_differ_only_in_their_reduction() -> None:
    module = nn.Linear(4, 2)
    x = torch.randn(3, 5, 4)
    observers = {cls: cls(module) for cls in (SquaredNormObserver, AbsMeanObserver)}
    for observer in observers.values():
        observer.update(x[:2])
        observer.update(x[2:])
    flat = x.reshape(-1, 4)
    torch.testing.assert_close(observers[SquaredNormObserver].mean, flat.square().mean(0))
    torch.testing.assert_close(observers[AbsMeanObserver].mean, flat.abs().mean(0))

    class MaxObserver(FeatureMeanObserver):
        @staticmethod
        def reduce(x: Tensor) -> Tensor:
            return x.clamp(min=0)

    custom = MaxObserver(module)
    custom.update(x)
    torch.testing.assert_close(custom.mean, flat.clamp(min=0).mean(0))
    with pytest.raises(TypeError):
        FeatureMeanObserver(module)  # type: ignore[abstract]


def test_calibration_helpers_ask_the_adapter(tiny_model) -> None:
    adapter = adapter_for(tiny_model)
    assert linear_modules(adapter, adapter.layer(1)) == adapter.linear_modules(adapter.layer(1))
    assert layer_device(adapter, 1) == adapter.layer_device(1)
    data = [torch.zeros(1, 4, dtype=torch.long)]
    assert calibration_source(AttachContext(calibration=data), "prune") is data
    for context in (None, AttachContext()):
        with pytest.raises(ConfigError, match="'prune' needs calibration data"):
            calibration_source(context, "prune")


def test_each_cache_class_names_its_own_static_variant() -> None:
    from economize.decoding.graphs import static_cache_class

    assert PositionalKVCache.static_variant is StaticKVCache and static_cache_class("positional") is StaticKVCache
    assert HybridCache.static_variant is None and static_cache_class("hybrid") is None  # recurrent states: eager

    class Custom(PositionalKVCache):  # may hold state StaticKVCache would drop: not inherited
        pass

    class CustomStatic(Custom):
        def activate(self, batch_size: int) -> None:
            self.batch_size = batch_size

    class Declared(Custom):
        static_variant = CustomStatic

    CACHES.register("custom", Custom)
    CACHES.register("declared", Declared)
    try:
        assert Custom.static_variant is None and static_cache_class("custom") is None
        assert static_cache_class("declared") is CustomStatic
    finally:
        CACHES.unregister("custom")
        CACHES.unregister("declared")
