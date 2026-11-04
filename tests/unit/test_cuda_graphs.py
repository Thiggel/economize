"""CUDA graphs in the decoding loop (``DecodeConfig.cuda_graphs``): replayed passes match eager decoding.

Run with ``-m gpu`` on a machine with a GPU (``CUDA_VISIBLE_DEVICES`` picks it); skipped otherwise.
"""

from __future__ import annotations

import pytest
import torch

import economize
from economize import DecodeConfig, JumpConfig, TopPSampler
from economize.arch import adapter_for
from economize.compat.tskip import cost_events
from economize.decoding import CycleDecoder
from economize.runtime import LayerRunner, StaticKVCache
from helpers import VOCAB, ScriptedTreeDrafter, make_tiny_model, perturb, reference_greedy

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")]

DEVICE = "cuda"
MAX_NEW = 16
VARIANTS = {
    "frozen": dict(target_layer=2, source_layer=4),
    "lora": dict(target_layer=2, source_layer=4, adaptation="lora", lora_rank=4),
    "full_readout": dict(target_layer=1, source_layer=3, exit_layer=4, adaptation="full"),
    "layer_attention": dict(target_layer=2, source_layer=5, extra_source_layers=(1, 3), mixer="layer_attention",
                            mixer_heads=4),
}


def _prompts(lengths: tuple[int, ...] = (13, 5, 9)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    return [torch.randint(3, VOCAB, (n,), generator=generator).to(DEVICE) for n in lengths]


def _jump_model(variant: str = "frozen") -> economize.EfficientModel:
    model = economize.apply(make_tiny_model().to(DEVICE), JumpConfig(**VARIANTS[variant]))
    perturb(model.techniques["jump"], 0.05)
    return model


@pytest.mark.parametrize("variant", VARIANTS)
@torch.no_grad()
def test_graphed_jump_decoding_is_greedy_and_reuses_its_graphs(variant: str) -> None:
    model = _jump_model(variant)
    prompts = _prompts()
    greedy = [reference_greedy(model.base_model, p, MAX_NEW).tolist() for p in prompts]
    decoding = DecodeConfig(cycle_length=4, cuda_graphs=True)
    for _ in range(3):  # eager warm-up, capture, replay
        results = model.decode_batch(prompts, MAX_NEW, decoding=decoding)
        assert [r.tokens.tolist() for r in results] == greedy
    decoder = model.decoder(decoding)
    assert decoder.graphs is not None and decoder.graphs.num_graphs > 0
    recorded = decoder.graphs.num_graphs
    again = model.decode_batch(prompts, MAX_NEW, decoding=decoding)
    assert [r.tokens.tolist() for r in again] == greedy
    assert decoder.graphs.num_graphs == recorded  # nothing new to record for the same shapes


@torch.no_grad()
def test_graphed_lossy_decoding_matches_eager_bit_for_bit() -> None:
    model = _jump_model("lora")
    prompts = _prompts()
    eager = model.decode_batch(prompts, MAX_NEW, decoding=DecodeConfig(cycle_length=4, lossless=False))
    graphed = DecodeConfig(cycle_length=4, lossless=False, cuda_graphs=True)
    for _ in range(3):
        results = model.decode_batch(prompts, MAX_NEW, decoding=graphed)
        assert [r.tokens.tolist() for r in results] == [r.tokens.tolist() for r in eager]
    assert model.decoder(graphed).graphs.num_graphs > 0  # type: ignore[union-attr]


@torch.no_grad()
def test_graphs_follow_rows_that_leave_the_batch() -> None:
    model = _jump_model()
    prompts = _prompts((13, 5, 9, 7))
    limits = [16, 3, 9, 1]
    greedy = [reference_greedy(model.base_model, p, n).tolist() for p, n in zip(prompts, limits)]
    decoding = DecodeConfig(cycle_length=3, cuda_graphs=True)
    for _ in range(3):
        results = model.decode_batch(prompts, limits, decoding=decoding)
        assert [r.tokens.tolist() for r in results] == greedy
    cache = model.decoder(decoding).graphs._cache  # type: ignore[union-attr]
    assert isinstance(cache, StaticKVCache) and len(cache.batch_sizes) > 1


@torch.no_grad()
def test_plain_decoding_and_a_larger_cache() -> None:
    base = make_tiny_model().to(DEVICE)
    model = economize.apply(base, JumpConfig(target_layer=2, source_layer=4))
    prompt = _prompts()[0]
    decoding = DecodeConfig(cuda_graphs=True)
    for n in (8, 8, 8, 200):  # the last generation outgrows the cache: a new cache and new graphs
        result = model.decode(prompt, n, decoding=decoding)
        assert result.tokens.tolist() == reference_greedy(base, prompt, n).tolist()
    assert model.decoder(decoding).graphs._cache.capacity >= 200  # type: ignore[union-attr]


@torch.no_grad()
def test_graphed_plain_batches_match_eager_with_stops_and_costs() -> None:
    """Graphed plain decoding builds its one-token passes on the device; tokens and cost logs equal eager decoding
    for ragged prompts, own lengths and stops that end inside a read-back window."""
    base = make_tiny_model().to(DEVICE)
    model = economize.apply(base, JumpConfig(target_layer=2, source_layer=4))
    prompts = _prompts((13, 5, 9, 7))
    stop_token = int(reference_greedy(base, prompts[2], 8)[4])
    stops = [None, lambda seq: seq.numel() >= 5 + 6, stop_token, None]
    limits = [3, MAX_NEW, MAX_NEW, 11]
    eager = model.decode_batch(prompts, limits, stops=stops, decoding=DecodeConfig(sync_interval=3))
    graphed = DecodeConfig(sync_interval=3, cuda_graphs=True)
    for _ in range(3):
        results = model.decode_batch(prompts, limits, stops=stops, decoding=graphed)
        assert [r.tokens.tolist() for r in results] == [r.tokens.tolist() for r in eager]
        assert [cost_events(r.cost) for r in results] == [cost_events(r.cost) for r in eager]
    assert results[2].tokens.tolist()[-1] == stop_token
    assert model.decoder(graphed).graphs.num_graphs > 0  # type: ignore[union-attr]


@torch.no_grad()
def test_graphed_speculative_sampling_is_seeded() -> None:
    model = _jump_model()
    prompts = _prompts()
    decoding = DecodeConfig(cycle_length=4, verifier="rejection", cuda_graphs=True)
    runs = [
        model.decode_batch(prompts, MAX_NEW, decoding=decoding, selector=TopPSampler(0.9, 0.95, 7, DEVICE))
        for _ in range(4)
    ]
    eager = model.decode_batch(prompts, MAX_NEW, decoding=DecodeConfig(cycle_length=4, verifier="rejection"),
                               selector=TopPSampler(0.9, 0.95, 7, DEVICE))
    for results in runs:
        assert [r.tokens.tolist() for r in results] == [r.tokens.tolist() for r in eager]


@torch.no_grad()
def test_graphed_draft_tree_passes_are_greedy() -> None:
    base = make_tiny_model().to(DEVICE)
    prompts = _prompts()
    greedy = [reference_greedy(base, p, MAX_NEW + 4) for p in prompts]
    scripts = [torch.cat([p, g]).cpu() for p, g in zip(prompts, greedy)]
    decoder = CycleDecoder(LayerRunner(adapter_for(base)), ScriptedTreeDrafter(scripts, cut=(2,)),
                           DecodeConfig(cycle_length=5, cuda_graphs=True))
    for _ in range(3):  # eager warm-up, capture, replay
        results = decoder.decode_batch(prompts, MAX_NEW)
        assert [r.tokens.tolist() for r in results] == [g[:MAX_NEW].tolist() for g in greedy]
    assert decoder.graphs is not None and decoder.graphs.num_graphs > 0
