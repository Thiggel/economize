"""KV-cache techniques under the chunked prefill (``DecodeConfig.prefill_chunk``) and the window cache.

The chunks of a prompt run inside one pass of the technique's cache wrapper, so a chunked prefill must decode the
tokens of an unchunked one and leave the same memory statistics: SnapKV and PyramidKV observe the same window of
queries, StreamingLLM cuts the same entries, KIVI and FP8 quantize the same prefixes. Over the window cache the
sliding layers keep only a ring, which no technique covers: a technique on every layer then decodes like the same
technique restricted to the full-attention layers over the full cache.
"""

from __future__ import annotations

import pytest
import torch

import economize
from economize import DecodeConfig
from economize.attention import install
from economize.decoding import generation as generation_module
from economize.techniques.duo_attention import DuoAttentionConfig
from economize.techniques.kv_evict import KVEvictConfig
from economize.techniques.kv_evict_fused import FusedKVEvictConfig
from economize.techniques.kv_fp8_fused import FusedFP8KVConfig
from economize.techniques.kv_kivi_fused import FusedKIVIConfig
from economize.techniques.kv_quant import FP8KVConfig, KIVIConfig
from economize.techniques.quest import QuestConfig
from helpers import VOCAB, family_skip_reason, make_family_model, make_tiny_model

CHUNK, LENGTHS, MAX_NEW = 4, (37, 37), [12, 9]  # the chunked prefill takes prompts of one length
KIVI = {"group_size": 4, "residual_length": 4}
# Duo gate values of the 6-layer tiny model: with threshold 0.5 every layer has a streaming head.
DUO = {"pattern": ((0.9, 0.1), (0.2, 0.8), (0.6, 0.3), (0.3, 0.4), (0.95, 0.05), (0.15, 0.85)), "sparsity": None,
       "threshold": 0.5, "sinks": 2, "recent": 4}
TECHNIQUES = {
    "snapkv": (KVEvictConfig(method="snapkv", heavy=4, observation_window=3, kernel_size=3), "sdpa"),
    "pyramidkv": (KVEvictConfig(method="pyramidkv", heavy=4, observation_window=3, beta=4), "sdpa"),
    "h2o": (KVEvictConfig(method="h2o", recent=3, heavy=4), "sdpa"),
    "streaming": (KVEvictConfig(method="streaming", recent=5, sinks=2), "economize"),
    "kv_evict_fused": (FusedKVEvictConfig(method="snapkv", heavy=4, observation_window=3), "economize"),
    "duo": (DuoAttentionConfig(**DUO), "economize"),
    "quest": (QuestConfig(page_size=16, token_budget=32), "economize"),
    "kivi": (KIVIConfig(bits=2, **KIVI), "sdpa"),
    "kivi_fused": (FusedKIVIConfig(bits=2, **KIVI), "economize"),
    "kv_fp8": (FP8KVConfig(), "sdpa"),
    "kv_fp8_fused": (FusedFP8KVConfig(), "economize"),
}


def _prompts() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return [torch.randint(3, VOCAB, (n,), generator=generator) for n in LENGTHS]


def _run(model: economize.EfficientModel, decoding: DecodeConfig) -> tuple[list[list[int]], list[dict[str, float]]]:
    results = model.decode_batch(_prompts(), MAX_NEW, decoding=decoding)
    technique = next(iter(model.techniques.values()))
    return [r.tokens.tolist() for r in results], [technique.kv_stats(b) for b in range(len(LENGTHS))]


@pytest.mark.parametrize("name", list(TECHNIQUES))
def test_chunked_prefill_decodes_like_one_pass(name: str) -> None:
    install()
    config, attn = TECHNIQUES[name]
    model = economize.apply(make_tiny_model(attn), config)
    expected = _run(model, DecodeConfig(cycle_length=1))
    assert _run(model, DecodeConfig(cycle_length=1, prefill_chunk=CHUNK)) == expected


WINDOWED = {
    "snapkv": lambda layers: KVEvictConfig(method="snapkv", heavy=4, observation_window=3, layers=layers),
    "streaming": lambda layers: KVEvictConfig(method="streaming", recent=5, sinks=2, layers=layers),
    "kivi": lambda layers: KIVIConfig(bits=2, layers=layers, **KIVI),
    "kv_fp8": lambda layers: FP8KVConfig(layers=layers),
}


@pytest.mark.parametrize("name", list(WINDOWED))
def test_window_cache_leaves_the_sliding_layers_to_the_ring(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ministral alternates sliding and full-attention layers (1-indexed full layers 2 and 4)."""
    reason = family_skip_reason("ministral")
    if reason is not None:
        pytest.skip(reason)
    monkeypatch.setattr(generation_module, "RING_PAGE", 4)
    full_layers = economize.apply(make_family_model("ministral"), WINDOWED[name]((2, 4)))
    expected, _ = _run(full_layers, DecodeConfig(cycle_length=1))
    every_layer = economize.apply(make_family_model("ministral"), WINDOWED[name](()))
    windowed, _ = _run(every_layer, DecodeConfig(cycle_length=1, prefill_chunk=CHUNK, window_cache=True))
    assert windowed == expected
