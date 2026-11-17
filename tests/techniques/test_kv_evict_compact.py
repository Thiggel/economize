"""Compact storage of positional KV eviction: the sinks and a ring of recent entries in owned buffers decode like
the masked reference (every entry in the dense cache, evicted ones hidden) and free the evicted entries."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import economize
from economize import DecodeConfig
from economize.errors import ConfigError
from economize.runtime import phase
from economize.techniques._kv import innermost
from economize.techniques.kv_evict import CompactWindowCache, KVEvictConfig, KVEvictTechnique, WindowEvictionCache
from economize.techniques.kv_quant import KIVIConfig
from helpers import VOCAB, make_tiny_model

LENGTHS = [24, 17, 30]
#: A ring that wraps several times within the prompts and generations above.
SMALL = dict(recent=8, sinks=3, headroom=8)
TREE = dict(cycle_length=4, draft_policy="dynamic_tree",
            draft_policy_options={"depth": 3, "topk": 2, "total_tokens": 6})


def _prompts() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in (12, 7, 20)]


def _identity_jump(base: torch.nn.Module, config: KVEvictConfig, cycle_length: int) -> economize.EfficientModel:
    jump = economize.JumpConfig(target_layer=0, source_layer=0, adaptation="frozen", lora_rank=2)
    model = economize.apply(base, jump, config, decoding=DecodeConfig(cycle_length=cycle_length))
    for name, parameter in model.techniques["jump"].predictor.named_parameters():
        if name != "out_gain":
            parameter.zero_()
    return model


def _model(attn: str, storage: str, cycle_length: int, **config: Any) -> economize.EfficientModel:
    evict = KVEvictConfig(storage=storage, **{**SMALL, **config})
    base = make_tiny_model(attn=attn)
    return economize.apply(base, evict) if cycle_length == 1 else _identity_jump(base, evict, cycle_length)


def _decode(model: economize.EfficientModel, decoding: DecodeConfig | None = None) -> list[list[int]]:
    prompts = _prompts()
    batched = [r.tokens.tolist() for r in model.decode_batch(prompts, LENGTHS, decoding=decoding)]
    single = model.decode(prompts[2], LENGTHS[2], decoding=decoding).tokens.tolist()
    assert single == batched[2]
    return batched


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
@pytest.mark.parametrize("cycle_length", [1, 4])
@pytest.mark.parametrize("headroom", [8, 64])
def test_compact_decodes_like_masked(attn: str, cycle_length: int, headroom: int) -> None:
    """Greedy tokens and byte counts match the masked reference for plain decoding and lossless drafting
    (verification, rollback, rows finishing at different lengths), with a ring that wraps."""
    compact = _model(attn, "compact", cycle_length, headroom=headroom)
    assert compact.techniques["kv_evict"].storage == "compact"
    tokens = _decode(compact)
    stats = [compact.techniques["kv_evict"].kv_stats(row) for row in range(3)]
    masked = _model(attn, "masked", cycle_length, headroom=headroom)
    assert tokens == _decode(masked)
    assert stats == [masked.techniques["kv_evict"].kv_stats(row) for row in range(3)]


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_compact_trees_decode_like_masked(attn: str) -> None:
    """Dynamic trees gather the accepted path into place inside the ring."""
    decoding = DecodeConfig(**TREE)
    compact, masked = (_model(attn, storage, 4) for storage in ("compact", "masked"))
    assert _decode(compact, decoding) == _decode(masked, decoding)


@pytest.mark.parametrize("chunk", [5, 10])
def test_compact_chunked_prefill_decodes_like_masked(chunk: int) -> None:
    """A chunked prompt is one exact pass: chunks within the headroom write into the ring, longer ones take the
    eager route (concatenate, then keep the window). A prompt longer than the ring raises."""
    decoding = DecodeConfig(cycle_length=1, prefill_chunk=chunk)
    prompts = [torch.randint(1, VOCAB, (37,), generator=torch.Generator().manual_seed(seed)) for seed in (0, 1)]
    tokens = []
    for storage in ("compact", "masked"):
        model = _model("eager", storage, 1, recent=32)
        tokens.append([r.tokens.tolist() for r in model.decode_batch(prompts, [20, 13], decoding=decoding)])
    assert tokens[0] == tokens[1]
    with pytest.raises(ConfigError, match="headroom"):
        _model("eager", "compact", 1).decode_batch(prompts, [20, 13], decoding=decoding)


def test_compact_owns_the_window_and_frees_the_rest() -> None:
    """The inner cache allocates no buffer for the covered layers; the owned buffers hold sinks + ring + trash."""
    model = _model("sdpa", "compact", 1)
    cache = model.techniques["kv_evict"].wrap_cache(model.runner.new_cache(64))
    assert isinstance(cache, CompactWindowCache) and isinstance(cache, WindowEvictionCache)
    seq = torch.randint(1, VOCAB, (1, 40), generator=torch.Generator().manual_seed(0))
    for t in range(seq.shape[1]):
        ctx = model.runner.context(cache, torch.tensor([[t]]), trusted=True)
        with phase("decode"):
            model.runner.forward(seq[:, t : t + 1], ctx)
    assert not innermost(cache).keys
    physical = SMALL["sinks"] + SMALL["recent"] + SMALL["headroom"] + 1
    assert cache.physical_slots == physical
    assert all(buffer.shape[-2] == cache.num_slots for buffer in cache.owned_buffers().values())  # logical stand-ins
    assert all(buffer.shape[-2] == physical for buffer in cache._store.keys.values())


def test_simulation_status_follows_the_storage() -> None:
    assert KVEvictTechnique.simulation_for(KVEvictConfig()) is None
    assert KVEvictTechnique.simulation_for(KVEvictConfig(storage="masked")) is not None
    assert KVEvictTechnique.simulation_for(KVEvictConfig(method="h2o", heavy=4)) is not None  # scores need masks
    with pytest.raises(ConfigError):
        KVEvictConfig(method="h2o", heavy=4, storage="compact")


def test_compact_falls_back_to_masked_next_to_another_kv_technique() -> None:
    model = economize.apply(make_tiny_model(), KIVIConfig(), KVEvictConfig(recent=8))
    assert model.techniques["kv_evict"].storage == "masked"
    with pytest.raises(ConfigError, match="storage='masked'"):
        economize.apply(make_tiny_model(), KIVIConfig(), KVEvictConfig(recent=8, storage="compact"))


def test_a_pass_beyond_the_headroom_raises() -> None:
    """A verify pass wider than the headroom would overwrite entries it still attends to."""
    model = _model("sdpa", "compact", 6, headroom=2)
    with pytest.raises(ConfigError, match="headroom"):
        model.decode(_prompts()[2], 20)
