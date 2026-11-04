"""The layer runner and the positional K/V cache against the model's own forward pass."""

from __future__ import annotations

import pytest
import torch

from economize import UnsupportedModelError
from economize.arch import adapter_for
from economize.runtime import CACHES, CacheSpec, LayerRunner, PositionalKVCache
from economize.runtime.cache import UNSEEN_POSITION

ATOL = 1e-4  # fp32 reduction-order noise is ~1e-5 on logits of size ~3 (different cache sizes)


def _exact(runner: LayerRunner, cache, tokens: torch.Tensor, start: int, capture=()):
    """One exact pass of a single row over ``tokens`` at positions ``start, start + 1, ...``."""
    positions = torch.arange(start, start + tokens.numel())[None]
    return runner.forward(tokens[None], runner.context(cache, positions), capture=capture)


@torch.no_grad()
def test_exact_pass_matches_the_model(tiny_model, prompt: torch.Tensor) -> None:
    runner = LayerRunner(adapter_for(tiny_model))
    cache = runner.new_cache(64)
    exact = _exact(runner, cache, prompt, 0, capture=(0, 2, 6))
    hf = tiny_model(prompt[None], output_hidden_states=True)
    torch.testing.assert_close(runner.logits(exact.final), hf.logits, atol=ATOL, rtol=0)
    torch.testing.assert_close(exact.hidden[0], hf.hidden_states[0], atol=ATOL, rtol=0)
    torch.testing.assert_close(exact.hidden[2], hf.hidden_states[2], atol=ATOL, rtol=0)
    torch.testing.assert_close(exact.hidden[6], exact.final)  # HF's last hidden state is after the final norm
    assert exact.context is not None and exact.context.positions.tolist() == [list(range(prompt.numel()))]


@torch.no_grad()
def test_incremental_passes_match_the_model(tiny_model, prompt: torch.Tensor) -> None:
    runner = LayerRunner(adapter_for(tiny_model))
    cache = runner.new_cache(64)
    head, tail = prompt[:5], prompt[5:]
    rows = [runner.logits(_exact(runner, cache, head, 0).final)[0]]
    rows.append(runner.logits(_exact(runner, cache, tail[:3], 5).final)[0])  # several tokens at once
    for i in range(3, tail.numel()):  # then one at a time
        rows.append(runner.logits(_exact(runner, cache, tail[i : i + 1], 5 + i).final)[0])
    torch.testing.assert_close(torch.cat(rows), tiny_model(prompt[None]).logits[0], atol=ATOL, rtol=0)


@torch.no_grad()
def test_stale_entries_are_overwritten_or_masked(tiny_model, prompt: torch.Tensor) -> None:
    """Garbage K/V written at future positions and later overwritten must not change any result."""
    adapter = adapter_for(tiny_model)
    runner = LayerRunner(adapter)
    cache = runner.new_cache(64)
    _exact(runner, cache, prompt[:6], 0)
    garbage = torch.randn(1, 4, adapter.hidden_size, generator=torch.Generator().manual_seed(0)) * 3
    runner.run(garbage, runner.context(cache, torch.arange(6, 10)[None]), 1, adapter.num_layers)  # rejected drafts
    runner.run(garbage[:, :1], runner.context(cache, torch.tensor([[30]])), 3, 5)  # stale slot, never overwritten
    final = _exact(runner, cache, prompt[6:], 6).final
    torch.testing.assert_close(runner.logits(final)[0], tiny_model(prompt[None]).logits[0, 6:], atol=ATOL, rtol=0)


@torch.no_grad()
def test_padded_rows_leave_the_cache_unchanged(tiny_model, prompt: torch.Tensor) -> None:
    """Padding tokens write to the trash slot only and never change another token's result."""
    adapter = adapter_for(tiny_model)
    runner = LayerRunner(adapter)
    cache = runner.new_cache(32, batch_size=2)
    tokens = torch.stack([prompt[:8], prompt[:8]])
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[1, 5:] = False  # row 1 has only 5 tokens
    exact = runner.forward(tokens, runner.context(cache, torch.arange(8).expand(2, 8), valid))
    expected = tiny_model(prompt[None, :8]).logits[0]
    torch.testing.assert_close(runner.logits(exact.final)[0], expected, atol=ATOL, rtol=0)
    torch.testing.assert_close(runner.logits(exact.final)[1, :5], expected[:5], atol=ATOL, rtol=0)
    keys, _ = cache.read(0)
    assert keys[1, :, 5:8].abs().sum() == 0 and keys[0, :, 5:8].abs().sum() > 0


def test_cache_spec_is_checked() -> None:
    with pytest.raises(ValueError, match="positive"):
        CacheSpec(0)
    assert set(CACHES.names()) >= {"positional", "hybrid"}


def test_cache_requires_slots() -> None:
    cache = PositionalKVCache(CacheSpec(8))
    with pytest.raises(RuntimeError, match="writing_at"):
        cache.update(torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4), 0)
    assert cache.num_slots == 9 and cache.trash_slot == 8
    assert cache.key_positions(torch.device("cpu"))[-1] == UNSEEN_POSITION  # the trash slot is never visible


def test_cache_writes_at_the_given_slots() -> None:
    cache = PositionalKVCache(CacheSpec(5))
    key = torch.arange(4.0).view(1, 1, 2, 2)
    with cache.writing_at(torch.tensor([3, 1])):
        keys, values = cache.update(key, -key, 0)
    assert keys.shape == (1, 1, 6, 2)
    assert keys[0, 0, 3].tolist() == [0.0, 1.0] and keys[0, 0, 1].tolist() == [2.0, 3.0]
    assert values[0, 0, 1].tolist() == [-2.0, -3.0] and keys[0, 0, [0, 2, 4, 5]].abs().sum() == 0


def test_cache_rows_write_reorder_gather_and_select() -> None:
    cache = PositionalKVCache(CacheSpec(4, batch_size=3))
    key = torch.arange(1.0, 7.0).view(3, 1, 2, 1)  # row b writes 2b+1, 2b+2
    with cache.writing_at(torch.tensor([[0, 1], [1, 2], [2, 3]])):  # a different slot pair per row
        cache.update(key, key, 0)
    keys, _ = cache.read(0)
    assert keys[:, 0, :, 0].tolist() == [[1, 2, 0, 0, 0], [0, 3, 4, 0, 0], [0, 0, 5, 6, 0]]
    cache.gather(torch.tensor([2]), torch.tensor([[3]]), torch.tensor([[0]]))  # row 2: slot 3 -> slot 0
    assert cache.read(0)[0][2, 0, :, 0].tolist() == [6, 0, 5, 6, 0]
    with cache.selecting(torch.tensor([1])), cache.writing_at(torch.tensor([4])):
        sub, _ = cache.update(torch.full((1, 1, 1, 1), 9.0), torch.zeros(1, 1, 1, 1), 0)
    assert sub.shape[0] == 1 and cache.read(0)[0][1, 0, 4, 0] == 9 and cache.read(0)[0][0, 0, 4, 0] == 0
    cache.reorder(torch.tensor([2, 0]))
    assert cache.batch_size == 2 and cache.read(0)[0][:, 0, :, 0].tolist() == [[6, 0, 5, 6, 0], [1, 2, 0, 0, 0]]


def test_runner_refresh_forgets_layer_devices(tiny_factory) -> None:
    runner = LayerRunner(adapter_for(tiny_factory()))
    assert runner._device(1) == torch.device("cpu") and runner._devices
    runner.refresh()
    assert runner._devices == {} and not runner.has_recurrent_layers


def test_unsupported_models_are_rejected(tiny_factory) -> None:
    with pytest.raises(UnsupportedModelError, match="no economize adapter"):
        adapter_for(torch.nn.Linear(2, 2))
    model = tiny_factory()
    model.config._attn_implementation = "flash_attention_2"
    with pytest.raises(UnsupportedModelError, match="flash_attention_2"):
        adapter_for(model)


def test_adapter_layer_indexing(tiny_factory) -> None:
    adapter = adapter_for(tiny_factory())
    assert adapter.layer(1) is adapter.layers[0] and adapter.layer(6) is adapter.layers[5]
    with pytest.raises(IndexError):
        adapter.layer(0)
    assert set(adapter.lora_targets(adapter.layer(1))) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }
