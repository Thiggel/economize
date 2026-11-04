"""Batched decoding on a CUDA device: jump drafters, draft trees and speculative sampling.

Run with ``-m gpu`` on a machine with a GPU (``CUDA_VISIBLE_DEVICES`` picks it); skipped otherwise.
"""

from __future__ import annotations

import pytest
import torch

import economize
from economize import DecodeConfig, JumpConfig, TopPSampler
from economize.arch import adapter_for
from economize.decoding import CycleDecoder
from economize.runtime import LayerRunner
from helpers import VOCAB, ScriptedTreeDrafter, make_identity_jump, make_tiny_model, perturb, reference_greedy

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


def _prompts() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    return [torch.randint(3, VOCAB, (n,), generator=generator).to(DEVICE) for n in (13, 5, 9)]


@pytest.mark.parametrize("variant", VARIANTS)
@torch.no_grad()
def test_batched_jump_decoding_is_greedy(variant: str) -> None:
    base = make_tiny_model().to(DEVICE)
    model = economize.apply(base, JumpConfig(**VARIANTS[variant]))
    perturb(model.techniques["jump"], 0.05)
    prompts = _prompts()
    greedy = [reference_greedy(base, p, MAX_NEW).tolist() for p in prompts]
    results = model.decode_batch(prompts, MAX_NEW, decoding=DecodeConfig(cycle_length=4))
    assert [r.tokens.tolist() for r in results] == greedy
    assert all(r.tokens.device.type == "cpu" or r.tokens.is_cuda for r in results)
    single = model.decode(prompts[1], MAX_NEW, decoding=DecodeConfig(cycle_length=4))
    assert single.tokens.tolist() == greedy[1]


@torch.no_grad()
def test_draft_trees_are_greedy() -> None:
    base = make_tiny_model().to(DEVICE)
    prompts = _prompts()
    greedy = [reference_greedy(base, p, MAX_NEW + 4) for p in prompts]
    scripts = [torch.cat([p, g]).cpu() for p, g in zip(prompts, greedy)]
    decoder = CycleDecoder(LayerRunner(adapter_for(base)), ScriptedTreeDrafter(scripts, cut=(2,)),
                           DecodeConfig(cycle_length=5))
    results = decoder.decode_batch(prompts, MAX_NEW)
    assert [r.tokens.tolist() for r in results] == [g[:MAX_NEW].tolist() for g in greedy]


@torch.no_grad()
def test_bfloat16_identity_drafter_is_mostly_accepted() -> None:
    base = make_tiny_model().to(DEVICE, torch.bfloat16)
    model = make_identity_jump(base)
    prompts = _prompts()
    lossless = model.decode_batch(prompts, MAX_NEW, decoding=DecodeConfig(cycle_length=4))
    lossy = model.decode_batch(prompts, MAX_NEW, decoding=DecodeConfig(cycle_length=4, lossless=False))
    assert all(r.tokens.numel() == MAX_NEW for r in (*lossless, *lossy))
    drafted = sum(d for r in lossless for d, _ in r.acceptance)
    accepted = sum(a for r in lossless for _, a in r.acceptance)
    assert accepted >= 0.8 * drafted  # bf16 kernels round batched and single-token calls differently
    first = [reference_greedy(base, p, 2)[0].item() for p in prompts]
    assert [int(r.tokens[0]) for r in lossless] == first  # the prefill token is exact


@torch.no_grad()
def test_speculative_sampling_runs_on_the_device() -> None:
    base = make_tiny_model().to(DEVICE)
    model = economize.apply(base, JumpConfig(target_layer=2, source_layer=4))
    perturb(model.techniques["jump"], 0.05)
    prompts = _prompts()
    decoding = DecodeConfig(cycle_length=4, verifier="rejection")
    first = model.decode_batch(prompts, MAX_NEW, decoding=decoding, selector=TopPSampler(0.9, 0.95, 7, DEVICE))
    again = model.decode_batch(prompts, MAX_NEW, decoding=decoding, selector=TopPSampler(0.9, 0.95, 7, DEVICE))
    assert [r.tokens.tolist() for r in first] == [r.tokens.tolist() for r in again]  # seeded
    assert all(r.tokens.numel() == MAX_NEW for r in first) and any(a for r in first for _, a in r.acceptance)
