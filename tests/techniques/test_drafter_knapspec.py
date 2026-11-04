"""KnapSpec: sublayer skipping, the knapsack search, lossless decoding, costs and save/load."""

from __future__ import annotations

import itertools
import math
from typing import Any

import pytest
import torch

import economize
from economize import ConfigError, DecodeConfig, UnsupportedModelError
from economize.arch import adapter_for
from economize.core.technique import TECHNIQUES
from economize.runtime import LayerRunner
from economize.techniques.knapspec import (
    SEARCH_COST,
    KnapsackSearch,
    KnapSpecConfig,
    KnapSpecDrafter,
    KnapSpecTechnique,
    SearchTokens,
    SkipPlan,
    SublayerLatency,
    SublayerSkipper,
    measure_latency,
)
from helpers import VOCAB, family_skip_reason, make_family_model, make_tiny_model, reference_greedy

LAYERS = 6
#: Latencies with attention as cheap as the MLP at short contexts (weights 1 and 1), so a test needs no timing.
FLAT = (1e-4, 0.0, 1e-4)


def _prompts(seed: int = 1, lengths: tuple[int, ...] = (12, 7, 20)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in lengths]


def _model(config: KnapSpecConfig, base: torch.nn.Module | None = None, **decoding: Any) -> Any:
    return economize.apply(base if base is not None else make_tiny_model(), config,
                           decoding=DecodeConfig(**decoding) if decoding else None)


def _prefill(runner: LayerRunner, ids: torch.Tensor):
    """A cache holding the exact K/V of ``ids`` ``[B, S]`` and the context of that pass."""
    cache = runner.new_cache(ids.shape[1] + 8, ids.shape[0])
    ctx = runner.context(cache, torch.arange(ids.shape[1]), None, trusted=True)
    embedding = runner.embed(ids, ctx)
    runner.run(embedding, ctx, 1, runner.adapter.num_layers)
    return cache, ctx, embedding


# ---- configuration -------------------------------------------------------------------------------------------------
def test_registered_and_config_validation() -> None:
    assert TECHNIQUES.get("knapspec") is KnapSpecTechnique
    config = KnapSpecConfig(latency=FLAT, keep_first=1)
    assert KnapSpecConfig.from_dict(config.to_dict()) == config
    assert config.candidates(LAYERS) == (2, 3, 4, 5, 6)
    for bad in (dict(min_similarity=1.5), dict(history=0), dict(max_draft=0), dict(max_skip_fraction=0.0),
                dict(max_skip_fraction=1.5), dict(latency=(1.0, 0.0)), dict(latency=(0.0, 0.0, 1.0)),
                dict(calibration_lengths=(64,)), dict(calibration_lengths=(64, 64)), dict(keep_last=-1)):
        with pytest.raises(ConfigError):
            KnapSpecConfig(**bad)
    with pytest.raises(ConfigError):
        _model(KnapSpecConfig(keep_first=4, keep_last=3))


def test_recurrent_models_are_rejected() -> None:
    reason = family_skip_reason("qwen3_next")
    if reason is not None:
        pytest.skip(reason)
    with pytest.raises(UnsupportedModelError):
        economize.apply(make_family_model("qwen3_next"), KnapSpecConfig(latency=FLAT))


# ---- sublayer skipping ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("family", [None, "gemma2", "olmo2", "gemma4_text", "gemma4_text_moe"])
def test_attention_then_mlp_is_the_whole_layer(family: str | None) -> None:
    if family is not None and (reason := family_skip_reason(family)) is not None:
        pytest.skip(reason)
    base = make_tiny_model() if family is None else make_family_model(family)
    adapter = adapter_for(base)
    runner, skipper = LayerRunner(adapter), SublayerSkipper(adapter)
    ids = torch.randint(1, 100, (2, 9), generator=torch.Generator().manual_seed(0))
    cache = runner.new_cache(16, 2)
    ctx = runner.context(cache, torch.arange(9), None, trusted=True)
    full = partial = runner.embed(ids, ctx)
    for layer in range(1, adapter.num_layers + 1):
        full, _ = runner.run(full, ctx, layer, layer)
        partial, _ = runner.run(partial, ctx, layer, layer, overrides={layer: skipper.attention_only(layer)})
        partial, _ = runner.run(partial, ctx, layer, layer, overrides={layer: skipper.mlp_only(layer)})
    torch.testing.assert_close(partial, full)


def test_skipping_nothing_is_the_full_model_and_skipping_an_attention_is_the_manual_layer() -> None:
    base = make_tiny_model()
    adapter = adapter_for(base)
    runner, skipper = LayerRunner(adapter), SublayerSkipper(adapter)
    ids = torch.randint(1, VOCAB, (1, 7), generator=torch.Generator().manual_seed(2))
    _, ctx, embedding = _prefill(runner, ids)
    full, _ = runner.run(embedding, ctx, 1, LAYERS)
    overrides, policy = skipper.calls(frozenset(), frozenset())
    nothing, _ = runner.run(embedding, ctx, 1, LAYERS, overrides=overrides, policy=policy)
    torch.testing.assert_close(nothing, full)

    overrides, policy = skipper.calls(frozenset({3}), frozenset())
    skipped, _ = runner.run(embedding, ctx, 1, LAYERS, overrides=overrides, policy=policy)
    hidden, _ = runner.run(embedding, ctx, 1, 2)
    layer = adapter.layer(3)
    hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))  # Qwen2: h + MLP(norm(h)), no attention
    manual, _ = runner.run(hidden, ctx, 4, LAYERS)
    torch.testing.assert_close(skipped, manual)
    assert not torch.allclose(skipped, full)


def test_a_skipped_attention_writes_no_kv() -> None:
    base = make_tiny_model()
    adapter = adapter_for(base)
    runner, skipper = LayerRunner(adapter), SublayerSkipper(adapter)
    ids = torch.randint(1, VOCAB, (1, 5), generator=torch.Generator().manual_seed(3))
    cache, _, _ = _prefill(runner, ids)
    stores = (cache.keys, cache.values)
    before = {i: [store[i].clone() for store in stores] for i in (1, 2)}  # HF layer_idx of blocks 2 and 3
    ctx = runner.context(cache, torch.tensor([5]), None, trusted=True)
    overrides, policy = skipper.calls(frozenset({2}), frozenset({4}))
    runner.run(runner.embed(torch.tensor([[7]]), ctx), ctx, 1, LAYERS, overrides=overrides, policy=policy)
    assert all(torch.equal(a, store[1]) for a, store in zip(before[1], stores))
    assert not all(torch.equal(a, store[2]) for a, store in zip(before[2], stores))
    assert skipper.skippable == (True, True) * LAYERS


# ---- the search ----------------------------------------------------------------------------------------------------
def _brute_force(runner, skipper, tokens: SearchTokens, cache, ctx, budget: int, weights: tuple[int, int]
                 ) -> dict[int, float]:
    """Best mean cosine similarity of the last state over all skip sets per skipped weight (no pruning)."""
    L = runner.adapter.num_layers
    full, _ = runner.run(tokens.hidden, ctx, 1, L)
    best: dict[int, float] = {}
    for mask in itertools.product((False, True), repeat=2 * L):
        j = sum(weights[s % 2] for s, m in enumerate(mask) if m)
        if j > budget:
            continue
        attention = frozenset(s // 2 + 1 for s, m in enumerate(mask) if m and s % 2 == 0)
        mlp = frozenset(s // 2 + 1 for s, m in enumerate(mask) if m and s % 2 == 1)
        overrides, policy = skipper.calls(attention, mlp)
        out, _ = runner.run(tokens.hidden, ctx, 1, L, overrides=overrides, policy=policy)
        cos = float(torch.nn.functional.cosine_similarity(out, full, dim=-1).mean())
        best[j] = max(best.get(j, -1.0), cos)
    return best


def test_search_returns_valid_plans_close_to_brute_force() -> None:
    base = make_tiny_model(num_layers=3)
    adapter = adapter_for(base)
    runner, skipper = LayerRunner(adapter), SublayerSkipper(adapter)
    ids = torch.randint(1, VOCAB, (1, 10), generator=torch.Generator().manual_seed(4))
    cache, ctx, embedding = _prefill(runner, ids)
    tokens = SearchTokens(embedding, ctx.positions.expand(1, -1), torch.ones(1, 10, dtype=torch.bool))
    latency = SublayerLatency(2e-4, 0.0, 1e-4)  # weights: attention 1, MLP 2, K = 9
    search = KnapsackSearch(skipper, (1, 2, 3), min_similarity=-1.0, max_skip_fraction=1.0, max_draft=4)
    assert search.weights(latency, 10) == (1, 2, 1e-4)
    plan, stats = search(runner, cache, tokens, latency)
    assert isinstance(plan, SkipPlan) and 1 <= plan.draft_length <= 4
    assert plan.skipped_weight == len(plan.attention) + 2 * len(plan.mlp) <= 9
    assert stats.layer_tokens > 0 and stats.head_tokens > 0
    # the program's similarity per budget is that of a real skip set: at most the brute-force optimum
    best = _brute_force(runner, skipper, tokens, cache, ctx, 9, (1, 2))
    assert plan.similarity <= best[plan.skipped_weight] + 1e-5
    overrides, policy = skipper.calls(plan.attention, plan.mlp)
    out, _ = runner.run(embedding, ctx, 1, 3, overrides=overrides, policy=policy)
    full, _ = runner.run(embedding, ctx, 1, 3)
    cos = float(torch.nn.functional.cosine_similarity(out, full, dim=-1).mean())
    assert math.isclose(cos, plan.similarity, abs_tol=1e-4)


def test_search_respects_the_budget_threshold_and_kept_layers() -> None:
    base = make_tiny_model()
    adapter = adapter_for(base)
    runner, skipper = LayerRunner(adapter), SublayerSkipper(adapter)
    ids = torch.randint(1, VOCAB, (2, 8), generator=torch.Generator().manual_seed(5))
    cache, ctx, embedding = _prefill(runner, ids)
    tokens = SearchTokens(embedding, ctx.positions.expand(2, -1), torch.ones(2, 8, dtype=torch.bool))
    latency = SublayerLatency(*FLAT)
    search = KnapsackSearch(skipper, (3, 4, 5), min_similarity=-1.0, max_skip_fraction=0.25, max_draft=10)
    plan, _ = search(runner, cache, tokens, latency)
    assert plan.skipped_weight <= math.floor(2 * LAYERS * 0.25)
    assert (plan.attention | plan.mlp) <= {3, 4, 5}
    strict, _ = KnapsackSearch(skipper, (1, 2, 3, 4, 5, 6), 1.0 - 1e-9, 0.5, 10)(runner, cache, tokens, latency)
    assert strict.skipped_weight == 0 or strict.similarity >= 1.0 - 1e-6


def test_measured_latencies_are_positive() -> None:
    base = make_tiny_model()
    adapter = adapter_for(base)
    runner, skipper = LayerRunner(adapter), SublayerSkipper(adapter)
    cache, _, _ = _prefill(runner, torch.randint(1, VOCAB, (1, 6)))
    latency = measure_latency(runner, cache, skipper, (2, 8, 64), repeats=3)
    assert latency.mlp > 0 and latency.attention_base > 0 and latency.attention_slope >= 0


# ---- lossless decoding ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cycle_length", [1, 3, 6])
@pytest.mark.parametrize("interval", [4, 128])
def test_greedy_knapspec_equals_plain_greedy(cycle_length: int, interval: int) -> None:
    base = make_tiny_model()
    model = _model(KnapSpecConfig(latency=FLAT, interval=interval, min_similarity=-1.0), base=base,
                   cycle_length=cycle_length)
    prompts = _prompts()
    new = [30, 21, 26]
    results = model.decode_batch(prompts, new)
    for prompt, result, n in zip(prompts, results, new):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, n))


@pytest.mark.parametrize("family", ["gemma2", "olmo2", "gemma4_text", "gemma4_text_moe", "llama"])
def test_greedy_knapspec_equals_plain_greedy_on_families(family: str) -> None:
    if (reason := family_skip_reason(family)) is not None:
        pytest.skip(reason)
    base = make_family_model(family)
    model = economize.apply(base, KnapSpecConfig(latency=FLAT, interval=8, min_similarity=-1.0),
                            decoding=DecodeConfig(cycle_length=4))
    prompt = torch.randint(1, 100, (11,), generator=torch.Generator().manual_seed(6))
    assert torch.equal(model.decode(prompt, 24).tokens, reference_greedy(base, prompt, 24))


def test_the_first_proposal_measures_latencies_and_searches() -> None:
    base = make_tiny_model()
    model = _model(KnapSpecConfig(calibration_lengths=(4, 32), calibration_repeats=2, interval=8), base=base,
                   cycle_length=4)
    technique = model.techniques["knapspec"]
    assert technique.latency is None
    prompt = _prompts()[2]
    result = model.decode(prompt, 30)
    assert torch.equal(result.tokens, reference_greedy(base, prompt, 30))
    assert technique.latency is not None and technique.plan is not None


# ---- costs ---------------------------------------------------------------------------------------------------------
def test_cost_log_holds_the_searches_and_the_layers_of_the_plan() -> None:
    model = _model(KnapSpecConfig(latency=FLAT, interval=10, min_similarity=-1.0), cycle_length=4)
    result = model.decode(_prompts()[0], 40)
    events = result.cost.events
    searches = [e for e in events if e.kind == SEARCH_COST]
    assert len(searches) >= 3 and all((e.first_layer, e.last_layer) == (1, LAYERS) and e.tokens > 0 for e in searches)
    # every draft step logs the layers of the plan active when it ran
    plan_layers = None
    for event in events:
        if event.kind == SEARCH_COST:
            plan_layers = None
        elif event.kind.startswith("draft"):
            if plan_layers is None:
                plan_layers = event.layer_indices
            assert event.layer_indices == plan_layers
    assert sum(1 for e in events if e.kind == "draft") == sum(k for k, _ in result.acceptance)


def test_draft_length_follows_the_plan() -> None:
    model = _model(KnapSpecConfig(latency=FLAT, max_draft=2, min_similarity=-1.0), cycle_length=6)
    result = model.decode(_prompts()[1], 30)
    assert all(k <= 2 for k, _ in result.acceptance)


# ---- save and load -------------------------------------------------------------------------------------------------
def test_config_only_checkpoint_round_trip(tmp_path) -> None:
    from safetensors.torch import save_file

    config = KnapSpecConfig(latency=FLAT, interval=16)
    model = _model(config, cycle_length=4)
    economize.save(model, tmp_path / "ks")
    loaded = economize.load(make_tiny_model(), tmp_path / "ks")
    assert loaded.techniques["knapspec"].config == config
    prompt = _prompts()[1]
    first, second = model.decode(prompt, 20), loaded.decode(prompt, 20)
    assert torch.equal(first.tokens, second.tokens) and first.acceptance == second.acceptance

    # a hand-written directory (configuration and an empty weight file) attaches the same technique
    import json

    manual = tmp_path / "manual"
    manual.mkdir()
    (manual / "economize_config.json").write_text(
        (tmp_path / "ks" / "economize_config.json").read_text())
    save_file({}, str(manual / "economize_model.safetensors"))
    checkpoint = economize.read_checkpoint(manual)
    assert [c.technique for c in checkpoint.configs] == ["knapspec"]
    attached = economize.attach(make_tiny_model(), economize.Checkpoint(list(checkpoint.configs), checkpoint.states))
    assert attached.techniques["knapspec"].config == config
    assert json.loads((manual / "economize_config.json").read_text())


def test_drafter_type() -> None:
    technique = KnapSpecTechnique(KnapSpecConfig(latency=FLAT), adapter_for(make_tiny_model()))
    assert isinstance(technique.drafter(None), KnapSpecDrafter)
    assert technique.layer_span == (1, LAYERS) and "plan=None" in repr(technique)


def test_graphed_steps_survive_evictions_and_refreshes(monkeypatch) -> None:
    """The sequence of xb31_tune_ks (a bench of several decoding configurations, ``refresh()`` between them, a few
    graphed skip sets kept): decoders that record graphs keep their drafter across generations and evict graphed
    steps of older skip sets. Every generation stays equal to plain greedy decoding. On the CPU the graph path runs
    with :class:`~economize.runtime.StaticStep` executing eagerly."""
    from economize.decoding.loop import CycleDecoder
    from economize.techniques._drafting import GraphedStep

    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    monkeypatch.setattr(CycleDecoder, "_graphable", lambda self: True)

    evicted: list[int] = []
    reset = GraphedStep.reset
    monkeypatch.setattr(GraphedStep, "reset", lambda self: (evicted.append(id(self)), reset(self))[1])
    base = make_family_model("gemma4_text")
    model = economize.apply(base, KnapSpecConfig(latency=FLAT, interval=4, min_similarity=-1.0, max_draft=4,
                                                 max_graphs=1, prompt_tokens=8))
    prompts = _prompts(lengths=(40, 9, 25))
    references = [reference_greedy(base, p, 30) for p in prompts]
    for decoding in (DecodeConfig(cycle_length=5, cuda_graphs=True),
                     DecodeConfig(cycle_length=5, cuda_graphs=True, min_confidence=0.3)):
        model.refresh()
        for prompt, reference in zip(prompts, references):
            assert torch.equal(model.decode(prompt, 30, decoding=decoding).tokens, reference)
    assert evicted, "the bench sequence never evicted a graphed step"
