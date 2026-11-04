"""Every supported family: speculative decoding of a ragged batch equals greedy decoding of each prompt alone.

A batch of three prompts of different lengths is decoded in float32 with drafts that are partly right, so that
cycles accept different numbers of tokens per row and rejected drafts leave stale K/V (and, in hybrid models,
advanced recurrent states) behind. Every row must equal Hugging Face greedy decoding by full recomputation. Draft
trees run on every family as well, where the recurrent layers of hybrid models process them path by path. A jump
drafter with a perturbed predictor runs on every family, so that the draft path's own K/V writes into layers
``m+1..r`` are overwritten correctly too.
"""

from __future__ import annotations

import pytest
import torch

import economize
from economize import DecodeConfig, JumpConfig
from economize.arch import adapter_for
from economize.decoding import CycleDecoder
from economize.runtime import LayerRunner
from helpers import (
    FAMILY_MODELS,
    VOCAB,
    ScriptedDrafter,
    ScriptedTreeDrafter,
    family_skip_reason,
    make_family_model,
    perturb,
    reference_greedy,
)

PROMPTS = (11, 4, 8)
MAX_NEW = 14
FAMILIES = sorted(FAMILY_MODELS)


@pytest.fixture(scope="module", params=FAMILIES)
def family(request):
    """``(model, prompts, greedy continuations, partly corrupted scripts)`` of one family."""
    reason = family_skip_reason(request.param)
    if reason is not None:
        pytest.skip(reason)
    model = make_family_model(request.param)
    generator = torch.Generator().manual_seed(1)
    prompts = [torch.randint(3, VOCAB, (n,), generator=generator) for n in PROMPTS]
    greedy = [reference_greedy(model, p, MAX_NEW + 4) for p in prompts]
    scripts = []
    for b, (prompt, row) in enumerate(zip(prompts, greedy)):
        script = torch.cat([prompt, row])
        wrong = script[prompt.numel() + 2 + b :: 3 + b]  # a different error pattern per row
        script[prompt.numel() + 2 + b :: 3 + b] = (wrong + 1) % VOCAB
        scripts.append(script)
    return request.param, model, prompts, [g[:MAX_NEW].tolist() for g in greedy], scripts


def _decode(model, drafter, prompts, cycle_length: int):
    decoder = CycleDecoder(LayerRunner(adapter_for(model)), drafter, DecodeConfig(cycle_length=cycle_length))
    return decoder.decode_batch(prompts, MAX_NEW)


@pytest.mark.parametrize("cycle_length", [1, 4])
def test_chain_drafts_equal_greedy(family, cycle_length: int) -> None:
    _, model, prompts, greedy, scripts = family
    drafter = ScriptedDrafter(scripts) if cycle_length > 1 else None
    results = _decode(model, drafter, prompts, cycle_length)
    assert [r.tokens.tolist() for r in results] == greedy
    if drafter is not None:
        accepted = [a for r in results for _, a in r.acceptance]
        assert 0 < sum(accepted) < sum(d for r in results for d, _ in r.acceptance)  # partial acceptance


@pytest.mark.parametrize(("cycle_length", "cut"), [(5, (2,)), (5, ()), (2, ())], ids=["cut", "uncut", "one_level"])
def test_tree_drafts_equal_greedy(family, cycle_length: int, cut: tuple[int, ...]) -> None:
    _, model, prompts, greedy, scripts = family
    results = _decode(model, ScriptedTreeDrafter(scripts, cut=cut), prompts, cycle_length)
    assert [r.tokens.tolist() for r in results] == greedy
    assert any(a > 0 for r in results for _, a in r.acceptance)


def test_medusa_tree_equals_greedy(family) -> None:
    """A static Medusa tree verified by the whole model: recurrent layers of hybrids run its root-to-node paths."""
    from economize.techniques.medusa import MedusaConfig

    _, model, prompts, greedy, _ = family
    tree = ((0,), (1,), (0, 0), (0, 1), (1, 0), (0, 0, 0))
    efficient = economize.apply(model, MedusaConfig(num_heads=3, tree=tree))
    perturb(efficient.techniques["medusa"].heads, 0.05, seed=1)
    try:
        results = efficient.decode_batch(prompts, MAX_NEW, decoding=DecodeConfig(cycle_length=4))
        assert [r.tokens.tolist() for r in results] == greedy
    finally:
        efficient.detach()


def test_jump_drafts_equal_greedy(family) -> None:
    _, model, prompts, greedy, _ = family
    num_layers = adapter_for(model).num_layers
    efficient = economize.apply(model, JumpConfig(target_layer=1, source_layer=2, exit_layer=num_layers))
    perturb(efficient.techniques["jump"], 0.02)
    try:
        results = efficient.decode_batch(prompts, MAX_NEW, decoding=DecodeConfig(cycle_length=3))
        assert [r.tokens.tolist() for r in results] == greedy
    finally:
        efficient.detach()
