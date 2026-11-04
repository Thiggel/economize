"""``kv_evict_fused``: prompt-time eviction whose kept entries the ``"economize"`` kernel reads compacted.

The oracle is ``kv_evict`` with the same settings. On CPU every pass falls back to ``sdpa`` over the masked dense
cache, so the tokens are equal. On a GPU (``-m gpu``) the kernel reads the compact copies and the generated entries,
which agrees with the masked dense kernel up to the summation order; in fp32 the greedy tokens are equal, which the
tests check together with the path log.
"""

from __future__ import annotations

import pytest
import torch

import economize
from economize import DecodeConfig, JumpConfig
from economize.attention import install, kernel_usage
from economize.errors import ConfigError, IncompatibleTechniquesError
from economize.techniques.kv_evict import KVEvictConfig
from economize.techniques.kv_evict_fused import FusedEvictionCache, FusedKVEvictConfig
from economize.techniques.kv_quant import FP8KVConfig, KIVIConfig
from helpers import VOCAB, make_tiny_model, perturb

SETTINGS = [
    {"method": "snapkv", "heavy": 4, "observation_window": 4},
    {"method": "h2o_prefill", "recent": 4, "heavy": 4},
    {"method": "pyramidkv", "heavy": 4, "observation_window": 4, "beta": 4},
]


def _prompts(device: str) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(5)
    return [torch.randint(3, VOCAB, (n,), generator=generator).to(device) for n in (31, 6, 19)]


def _greedy(attn: str, config: object, device: str, decoding: DecodeConfig, jump: bool = False) -> list[list[int]]:
    configs = [config]
    if jump:
        configs.insert(0, JumpConfig(target_layer=2, source_layer=4))
    model = economize.apply(make_tiny_model(attn).to(device), *configs)
    if jump:
        perturb(model.techniques["jump"], 0.05)
    out = [r.tokens.tolist() for r in model.decode_batch(_prompts(device), 16, decoding=decoding)]
    model.detach()
    return out


def test_needs_a_kernel_that_reads_a_compacted_cache() -> None:
    with pytest.raises(IncompatibleTechniquesError, match="compacted"):
        economize.apply(make_tiny_model("sdpa"), FusedKVEvictConfig(**SETTINGS[0]))


def test_needs_a_method_that_selects_once() -> None:
    for method in ("streaming", "h2o"):
        with pytest.raises(ConfigError, match="once"):
            FusedKVEvictConfig(method=method)


def test_is_exclusive_with_kv_quantizers_and_other_evictions() -> None:
    install()
    for other in (KIVIConfig(), FP8KVConfig(), KVEvictConfig(**SETTINGS[0])):
        with pytest.raises(IncompatibleTechniquesError):
            economize.apply(make_tiny_model("economize"), FusedKVEvictConfig(**SETTINGS[0]), other)


@torch.no_grad()
@pytest.mark.parametrize("settings", SETTINGS, ids=[s["method"] for s in SETTINGS])
def test_cpu_fallback_decodes_like_kv_evict(settings: dict) -> None:
    install()
    decoding = DecodeConfig(cycle_length=1)
    expected = _greedy("sdpa", KVEvictConfig(**settings), "cpu", decoding)
    assert _greedy("economize", FusedKVEvictConfig(**settings), "cpu", decoding) == expected


@torch.no_grad()
def test_compact_copies_hold_the_kept_prompt_entries() -> None:
    install()
    model = economize.apply(make_tiny_model("economize"), FusedKVEvictConfig(**SETTINGS[0]))
    seen: list[FusedEvictionCache] = []
    original = FusedEvictionCache.pass_begin

    def spy(self, frontier):
        seen.append(self)
        return original(self, frontier)

    FusedEvictionCache.pass_begin = spy  # type: ignore[method-assign]
    try:
        model.decode(_prompts("cpu")[0], 8)
    finally:
        FusedEvictionCache.pass_begin = original  # type: ignore[method-assign]
    cache = seen[-1]
    assert cache._compact and cache._prompt_end == [31]
    for layer, compact in cache._compact.items():
        keys, values = cache.inner.read(layer)
        keep = cache.keep[layer][..., :31]
        assert compact.dropped == 31 - (4 + 4)  # heavy + observation window kept per head
        for h in range(keys.shape[1]):
            slots = keep[0, h].nonzero().flatten()
            n = int(compact.count[0, h])
            assert torch.equal(compact.index[0, h, :n].long(), slots)
            assert torch.equal(compact.key[0, h, :n], keys[0, h, slots])
            assert torch.equal(compact.value[0, h, :n], values[0, h, slots])
    model.detach()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("settings", SETTINGS, ids=[s["method"] for s in SETTINGS])
@pytest.mark.parametrize("cycle", [1, 4])
@torch.no_grad()
def test_gpu_kernel_decodes_like_kv_evict(settings: dict, cycle: int) -> None:
    install()
    decoding = DecodeConfig(cycle_length=cycle)
    dense = _greedy("economize", KVEvictConfig(**settings), "cuda", decoding, jump=cycle > 1)
    assert dense == _greedy("sdpa", KVEvictConfig(**settings), "cuda", decoding, jump=cycle > 1)
    kernel_usage(reset=True)
    got = _greedy("economize", FusedKVEvictConfig(**settings), "cuda", decoding, jump=cycle > 1)
    assert got == dense
    paths = {row["path"] for row in kernel_usage(reset=True)}
    assert "kernel_gathered" in paths


@torch.no_grad()
def test_pyramidkv_compacts_a_budget_per_layer() -> None:
    """PyramidKV's per-layer budgets reach the compact copies: layer ``l`` of the 31-token prompt keeps
    ``8 - l`` heavy entries (``beta=4``: from 7 down to 2) plus the window of 4."""
    install()
    model = economize.apply(make_tiny_model("economize"), FusedKVEvictConfig(**SETTINGS[2]))
    seen: list[FusedEvictionCache] = []
    original = FusedEvictionCache.pass_begin

    def spy(self, frontier):
        seen.append(self)
        return original(self, frontier)

    FusedEvictionCache.pass_begin = spy  # type: ignore[method-assign]
    try:
        model.decode(_prompts("cpu")[0], 4)
    finally:
        FusedEvictionCache.pass_begin = original  # type: ignore[method-assign]
    cache = seen[-1]
    kept = {layer: int(compact.count[0, 0]) for layer, compact in sorted(cache._compact.items())}
    assert kept == {layer: 7 - layer + 4 for layer in range(6)}
    model.detach()
