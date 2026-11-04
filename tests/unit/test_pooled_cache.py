"""The pooled static cache (:mod:`economize.runtime.pooled_cache`): graphed batched decoding gives the same tokens as
the default static cache and as greedy decoding, while all batch sizes share one pool.

Run with ``-m gpu`` on a machine with a GPU; skipped otherwise.
"""

from __future__ import annotations

import pytest
import torch

import economize
from economize import DecodeConfig, JumpConfig
from economize.runtime import CACHES, PositionalKVCache
from economize.runtime.pooled_cache import PooledStaticKVCache, use_pooled_static_cache
from helpers import VOCAB, make_tiny_model, perturb, reference_greedy

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")]

DEVICE = "cuda"


@pytest.fixture
def pooled():
    use_pooled_static_cache()
    yield
    use_pooled_static_cache(False)


def _prompts(lengths: tuple[int, ...]) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(5)
    return [torch.randint(3, VOCAB, (n,), generator=generator).to(DEVICE) for n in lengths]


def _model() -> economize.EfficientModel:
    base = make_tiny_model().to(DEVICE).float()
    model = economize.apply(base, JumpConfig(target_layer=2, source_layer=4))
    perturb(model.techniques["jump"], 0.05)
    return model


@torch.no_grad()
@pytest.mark.parametrize("cycle", [1, 4])
def test_pooled_cache_matches_greedy_with_rows_leaving(pooled, cycle: int) -> None:
    model = _model()
    prompts = _prompts((13, 5, 9, 7, 11, 4))
    limits = [16, 3, 9, 1, 12, 6]  # rows leave the batch one after another: batch sizes 6, 5, ..., 1
    greedy = [reference_greedy(model.base_model, p, n).tolist() for p, n in zip(prompts, limits)]
    decoding = DecodeConfig(cycle_length=cycle, cuda_graphs=True)
    for _ in range(3):  # eager warm-up, capture, replay
        assert [r.tokens.tolist() for r in model.decode_batch(prompts, limits, decoding=decoding)] == greedy
    for size in (4, 2, 1):  # smaller batches later reuse the pool
        sub = model.decode_batch(prompts[:size], limits[:size], decoding=decoding)
        assert [r.tokens.tolist() for r in sub] == greedy[:size]
    cache = model.decoder(decoding).graphs._cache  # type: ignore[union-attr]
    assert isinstance(cache, PooledStaticKVCache) and len(cache.batch_sizes) > 1
    assert len(cache._pools) == 1 and cache.pool_rows == len(prompts)
    pool_ptrs = {t.data_ptr() for t in cache._pools[0][1].values()}
    for keys, _ in cache._sets.values():  # every batch size is a prefix view of the pool
        assert {t.data_ptr() for t in keys.values()} <= pool_ptrs


@torch.no_grad()
def test_pooled_cache_equals_default_static_cache_and_grows(pooled) -> None:
    model = _model()
    prompts = _prompts((13, 5, 9, 7))
    decoding = DecodeConfig(cycle_length=3, cuda_graphs=True)
    pooled_out = [model.decode_batch(prompts[:n], 10, decoding=decoding) for n in (2, 4, 1)]  # 4 > 2: a new pool
    cache = model.decoder(decoding).graphs._cache  # type: ignore[union-attr]
    assert [rows for rows, *_ in cache._pools] == [2, 4]
    use_pooled_static_cache(False)
    assert CACHES.get("positional") is PositionalKVCache
    ref = _model()
    default_out = [ref.decode_batch(prompts[:n], 10, decoding=decoding) for n in (2, 4, 1)]
    for a, b in zip(pooled_out, default_out):
        assert [r.tokens.tolist() for r in a] == [r.tokens.tolist() for r in b]
