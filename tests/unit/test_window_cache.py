"""The window cache of sliding layers (:class:`RingSpec`) and the chunked prefill (``DecodeConfig.prefill_chunk``).

A ring of a few slots per sliding layer must give the logits of the full cache, also after the ring has wrapped
several times, and speculative decoding over it must stay exactly greedy in float32 while rejected drafts leave
stale K/V in ring slots. The tiny family models have a window of 5 and the tests use 4-slot pages, so the ring is
much shorter than every sequence.
"""

from __future__ import annotations

import copy

import pytest
import torch

import economize
from economize import DecodeConfig, JumpConfig
from economize.arch import adapter_for
from economize.decoding import CycleDecoder
from economize.decoding import generation as generation_module
from economize.errors import ConfigError
from economize.runtime import LayerRunner, RingSpec
from economize.runtime.cache import UNSEEN_POSITION
from helpers import (
    VOCAB,
    WINDOW,
    ScriptedDrafter,
    ScriptedTreeDrafter,
    family_skip_reason,
    make_family_model,
    make_tiny_model,
    perturb,
    reference_greedy,
)

FAMILIES = ("gemma4", "ministral", "mistral")  # sliding + global + K/V sharing, alternating, every layer sliding
PROMPT, MAX_NEW, CHUNK, PAGE = 23, 14, 4, 4


@pytest.fixture(autouse=True)
def small_pages(monkeypatch):
    monkeypatch.setattr(generation_module, "RING_PAGE", PAGE)


@pytest.fixture(scope="module", params=FAMILIES)
def family(request):
    """``(model, two prompts of equal length, greedy continuations, partly corrupted scripts)``."""
    reason = family_skip_reason(request.param)
    if reason is not None:
        pytest.skip(reason)
    model = make_family_model(request.param)
    generator = torch.Generator().manual_seed(2)
    prompts = [torch.randint(3, VOCAB, (PROMPT,), generator=generator) for _ in range(2)]
    greedy = [reference_greedy(model, p, MAX_NEW + 4) for p in prompts]
    scripts = []
    for b, (prompt, row) in enumerate(zip(prompts, greedy)):
        script = torch.cat([prompt, row])
        script[PROMPT + 1 + b :: 3 + b] = (script[PROMPT + 1 + b :: 3 + b] + 1) % VOCAB
        scripts.append(script)
    return model, prompts, [g[:MAX_NEW].tolist() for g in greedy], scripts


def _ring_specs(monkeypatch) -> list:
    """The ring of every cache the runner builds from now on."""
    specs, build = [], LayerRunner.new_cache

    def spy(self, capacity, batch_size=1, kind=None, ring=None):
        specs.append(ring)
        return build(self, capacity, batch_size, kind, ring)

    monkeypatch.setattr(LayerRunner, "new_cache", spy)
    return specs


# ---- RingSpec --------------------------------------------------------------------------------------------------------
def test_for_pass_sizes_the_ring_in_whole_pages() -> None:
    ring = RingSpec.for_pass(frozenset({0}), window=1024, queries=4096, ahead=9)
    assert ring.slots % 256 == 0 and ring.slots - ring.page >= 4096 + 9 + 1024 - 2
    assert ring.buffer_slots == ring.slots + 256


@pytest.mark.parametrize(("window", "slots", "page"), [(5, 6, 4), (5, 8, 4), (0, 16, 4)])
def test_invalid_rings_raise(window: int, slots: int, page: int) -> None:
    with pytest.raises(ConfigError):
        RingSpec(frozenset({0}), window, slots, page)


def test_physical_slots_wrap_and_trash_goes_past_the_ring() -> None:
    ring = RingSpec(frozenset({0}), 5, 12, 4)
    trash = 99
    assert ring.physical(torch.tensor([0, 11, 12, 30, trash]), trash).tolist() == [0, 11, 0, 6, 12]


def test_key_positions_hold_the_newest_position_up_to_the_pass() -> None:
    ring = RingSpec(frozenset({0}), 5, 12, 4)
    held = ring.key_positions(torch.tensor([[25, 26], [3, 4]]))
    for b, last in enumerate((26, 4)):
        for slot in range(ring.buffer_slots):
            expected = last - (last - slot) % 12 if slot < 12 else -1
            assert held[b, slot].item() == (expected if expected >= 0 else UNSEEN_POSITION)


# ---- logits ----------------------------------------------------------------------------------------------------------
@torch.no_grad()
def test_ring_logits_equal_the_full_cache(family) -> None:
    """In float64, where chunking changes nothing beyond rounding (float32 differs by up to 3e-5 with or without the
    ring, from the summation order of the chunks)."""
    model, prompts, _, _ = family
    runner = LayerRunner(adapter_for(copy.deepcopy(model).double()))
    ring = generation_module.window_ring(runner, DecodeConfig(prefill_chunk=CHUNK, window_cache=True), 1)
    assert ring is not None and ring.buffer_slots < 2 * PROMPT
    seq = torch.cat([prompts[0], prompts[1]])[None]  # 46 tokens: the ring wraps several times
    width = seq.shape[1]
    full = runner.new_cache(width, 1)
    positions = torch.arange(width)[None]
    expected = runner.logits(runner.forward(seq, runner.context(full, positions, None, trusted=True)).final)

    cache = runner.new_cache(width, 1, ring=ring)
    pieces, begin = [], 0
    for size in [CHUNK] * 3 + [1] * (width - 3 * CHUNK):  # chunked prefill, then single-token steps
        span = positions[:, begin : begin + size]
        ctx = runner.context(cache, span, None, trusted=True, consecutive=True)
        pieces.append(runner.logits(runner.forward(seq[:, begin : begin + size], ctx).final))
        begin += size
    torch.testing.assert_close(torch.cat(pieces, 1), expected, rtol=1e-12, atol=1e-12)


# ---- decoding --------------------------------------------------------------------------------------------------------
def _decode(model, drafter, prompts, **config):
    decoder = CycleDecoder(LayerRunner(adapter_for(model)), drafter, DecodeConfig(**config))
    return [r.tokens.tolist() for r in decoder.decode_batch(prompts, MAX_NEW)]


@pytest.mark.parametrize("window_cache", [False, True], ids=["full_cache", "ring"])
@pytest.mark.parametrize("cycle_length", [1, 4])
def test_chunked_decoding_equals_greedy(family, monkeypatch, cycle_length: int, window_cache: bool) -> None:
    model, prompts, greedy, scripts = family
    specs = _ring_specs(monkeypatch)
    drafter = ScriptedDrafter(scripts) if cycle_length > 1 else None
    tokens = _decode(model, drafter, prompts, cycle_length=cycle_length, prefill_chunk=CHUNK, window_cache=window_cache)
    assert tokens == greedy
    assert (specs[-1] is not None) == window_cache


def test_jump_drafter_over_the_ring_equals_greedy(family, monkeypatch) -> None:
    model, prompts, greedy, _ = family
    specs = _ring_specs(monkeypatch)
    num_layers = adapter_for(model).num_layers
    efficient = economize.apply(model, JumpConfig(target_layer=1, source_layer=2, exit_layer=num_layers))
    perturb(efficient.techniques["jump"], 0.02)
    try:
        config = DecodeConfig(cycle_length=3, prefill_chunk=CHUNK, window_cache=True)
        results = efficient.decode_batch(prompts, MAX_NEW, decoding=config)
        assert [r.tokens.tolist() for r in results] == greedy
        assert specs[-1] is not None
        assert any(a < k for r in results for k, a in r.acceptance)  # rejected drafts stayed in ring slots
    finally:
        efficient.detach()


def test_short_sequences_keep_the_full_cache(monkeypatch) -> None:
    model = make_family_model("mistral")
    specs = _ring_specs(monkeypatch)
    monkeypatch.setattr(generation_module, "RING_PAGE", 256)
    prompt = torch.arange(3, 3 + PROMPT)
    tokens = _decode(model, None, [prompt], cycle_length=1, prefill_chunk=CHUNK, window_cache=True)
    assert tokens == [reference_greedy(model, prompt, MAX_NEW).tolist()]
    assert specs[-1] is None


# ---- what the chunked prefill refuses --------------------------------------------------------------------------------
def test_window_cache_needs_a_chunked_prefill() -> None:
    with pytest.raises(ConfigError, match="prefill_chunk"):
        DecodeConfig(window_cache=True)


def test_config_round_trip() -> None:
    config = DecodeConfig(prefill_chunk=4096, window_cache=True)
    assert DecodeConfig.from_dict(config.to_dict()) == config


def test_ragged_prompts_are_refused() -> None:
    model = make_tiny_model()
    with pytest.raises(ConfigError, match="equal length"):
        _decode(model, None, [torch.arange(3, 12), torch.arange(3, 10)], cycle_length=1, prefill_chunk=CHUNK)


class _TreeDrafter(ScriptedTreeDrafter):
    def drafts_trees(self, verifier) -> bool:
        return True


@pytest.mark.parametrize(("drafter_class", "message"), [(_TreeDrafter, "drafts trees"),
                                                        (ScriptedTreeDrafter, "tree passes need the chain layout")],
                         ids=["declared", "undeclared"])
def test_tree_drafter_is_refused_with_the_ring(drafter_class, message: str) -> None:
    """A drafter that declares its trees is refused before decoding, any other at its first tree pass."""
    model = make_family_model("mistral")
    with pytest.raises(ConfigError, match=message):
        _decode(model, drafter_class([torch.arange(3, 40)]), [torch.arange(3, 30)], cycle_length=4,
                prefill_chunk=CHUNK, window_cache=True)


def test_drafter_that_reads_the_prompt_is_refused() -> None:
    class PromptReader(ScriptedDrafter):
        streams_prompt = False

    model = make_tiny_model()
    with pytest.raises(ConfigError, match="PromptReader"):
        _decode(model, PromptReader([torch.arange(3, 40)]), [torch.arange(3, 12)], cycle_length=4, prefill_chunk=CHUNK)


def test_window_is_the_family_window(family) -> None:
    model, *_ = family
    config = DecodeConfig(prefill_chunk=8, window_cache=True)
    ring = generation_module.window_ring(LayerRunner(adapter_for(model)), config, 2)
    assert ring is not None and ring.window == WINDOW
