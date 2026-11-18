"""Measured KV memory (``kv_bytes_allocated_batch``): the bytes of the tensor storages the cache really holds, next to
the modelled ``kv_bytes``. A path that keeps the dense buffers measures no less than the dense cache, a path whose
saving is real measures less."""

from __future__ import annotations

import pytest
import torch

import economize
from economize.core.config import TechniqueConfig
from economize.core.metadata import Saving
from economize.runtime import StaticKVCache
from economize.techniques._kv import held_bytes
from economize.techniques.kv_evict import KVEvictConfig
from economize.techniques.kv_quant import FP8KVConfig, KIVIConfig
from helpers import VOCAB, make_tiny_model

LENGTHS = [40, 33]
KIVI = dict(bits=2, group_size=4, residual_length=8)
EVICT = dict(recent=8, sinks=3, headroom=8)


def _prompts() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in (12, 20)]


def _stats(config: TechniqueConfig) -> tuple[dict[str, float], bool]:
    model = economize.apply(make_tiny_model(), config)
    model.decode_batch(_prompts(), LENGTHS)
    technique = model.techniques[config.technique]
    simulation = technique.simulation()
    real = simulation is None or not simulation.savings & Saving.KV_MEMORY
    return technique.kv_stats(), real


def _dense() -> int:
    """Measured bytes of the plain cache of the same generation (a wrapper with no side state measures this)."""
    return int(_stats(KIVIConfig(storage="simulated", **KIVI))[0]["kv_bytes_allocated_batch"])


def test_held_bytes_counts_each_storage_once() -> None:
    cache = StaticKVCache.__new__(StaticKVCache)
    shared = torch.zeros(4, 8)
    cache.__dict__.update(a=shared, b=[shared, shared[1:]], c={"x": torch.zeros(3, dtype=torch.float64)})
    assert held_bytes(cache) == 4 * 8 * 4 + 3 * 8


def test_a_packed_ring_as_long_as_the_cache_measures_above_dense() -> None:
    """The measured value exposes what the model hides: with the default headroom the packed KIVI ring spans this
    whole short cache, so the codes come on top of full-precision buffers of the dense size."""
    stats, real = _stats(KIVIConfig(storage="packed", **KIVI))
    assert real and stats["kv_bytes"] < stats["kv_bytes_full"]
    assert stats["kv_bytes_allocated_batch"] > _dense()


@pytest.mark.parametrize(
    "config",
    [
        KIVIConfig(storage="simulated", **KIVI),
        KIVIConfig(storage="packed", headroom=8, **KIVI),  # the default ring (R + 256) exceeds this short cache
        FP8KVConfig(),
        KVEvictConfig(storage="masked", **EVICT),
        KVEvictConfig(storage="compact", **EVICT),
    ],
    ids=["kivi-simulated", "kivi-packed", "fp8", "evict-masked", "evict-compact"],
)
def test_measured_memory_matches_the_declared_simulation(config: TechniqueConfig) -> None:
    dense = _dense()
    stats, real = _stats(config)
    allocated = stats["kv_bytes_allocated_batch"]
    assert stats["kv_bytes_simulated"] == float(not real)
    assert stats["kv_bytes"] < stats["kv_bytes_full"]  # every one models a saving
    if real:
        assert allocated < dense
    else:
        assert allocated >= dense
