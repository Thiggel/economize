"""Self-speculative layer skipping: lossless decoding, skip-set search, costs, save/load and combination."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import economize
from economize import ConfigError, DecodeConfig, TopPSampler
from economize.arch import adapter_for
from economize.core.calibration import token_batches
from economize.core.technique import TECHNIQUES
from economize.decoding.cost import LayerFootprint
from economize.decoding.protocols import draft_footprint
from economize.runtime import LayerRunner
from economize.techniques.layer_skip import (
    SKIP_METRICS,
    SKIP_SEARCHES,
    GreedySearch,
    LayerSkipConfig,
    LayerSkipDrafter,
    LayerSkipTechnique,
    SkipLayers,
    SkipMetric,
    SkipScorer,
    SkipSearch,
)
from helpers import VOCAB, make_tiny_model, reference_greedy

LAYERS = 6


def _prompts(seed: int = 1, lengths: tuple[int, ...] = (12, 7, 20)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in lengths]


def _model(config: LayerSkipConfig, base: torch.nn.Module | None = None, calibration: Any = None,
           **decoding: Any) -> Any:
    return economize.apply(base if base is not None else make_tiny_model(), config,
                        decoding=DecodeConfig(**decoding) if decoding else None, calibration=calibration)


# ---- configuration -------------------------------------------------------------------------------------------------
def test_registered_and_config_validation() -> None:
    assert TECHNIQUES.get("layer_skip") is LayerSkipTechnique
    config = LayerSkipConfig(skip_layers=(2, 4))
    assert LayerSkipConfig.from_dict(config.to_dict()) == config and config.size == 2
    searched = LayerSkipConfig(num_skip=3, metric="mismatch", keep_first=2)
    assert LayerSkipConfig.from_dict(searched.to_dict()) == searched
    assert list(searched.candidates(LAYERS)) == [3, 4, 5]
    for bad in (dict(), dict(skip_layers=(0,)), dict(skip_layers=(2, 2)), dict(num_skip=0),
                dict(skip_layers=(2,), num_skip=2), dict(num_skip=2, metric="ce"), dict(num_skip=2, keep_last=-1)):
        with pytest.raises(ConfigError):
            LayerSkipConfig(**bad)
    for bad_config in (LayerSkipConfig(skip_layers=(LAYERS + 1,)), LayerSkipConfig(skip_layers=tuple(range(1, 7))),
                       LayerSkipConfig(num_skip=5), LayerSkipConfig(num_skip=2, search="bayes")):
        with pytest.raises(ConfigError):
            _model(bad_config, calibration=torch.zeros(1, 8, dtype=torch.long))


def test_policy_skips_exactly_the_set() -> None:
    """Running the model with the policy equals running it layer by layer without the skipped layers."""
    base = make_tiny_model()
    runner = LayerRunner(adapter_for(base))
    ids = torch.randint(0, VOCAB, (2, 10), generator=torch.Generator().manual_seed(0))
    scorer = SkipScorer(runner, [ids])
    skipped = scorer.log_probs(ids, frozenset({2, 5}))
    cache = runner.new_cache(10, 2)
    ctx = runner.context(cache, torch.arange(10), None, trusted=True)
    hidden = runner.embed(ids, ctx)
    for layer in (1, 3, 4, 6):
        hidden, _ = runner.run(hidden, ctx, layer, layer)
    torch.testing.assert_close(skipped, runner.logits(hidden).float().log_softmax(-1))
    assert repr(SkipLayers((5, 2))) == "SkipLayers([2, 5])"


# ---- lossless decoding ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("skip", [(3,), (2, 4), (2, 3, 4, 5)])
@pytest.mark.parametrize("cycle_length", [1, 3, 6])
def test_greedy_layer_skip_equals_plain_greedy(skip: tuple[int, ...], cycle_length: int) -> None:
    base = make_tiny_model()
    model = _model(LayerSkipConfig(skip_layers=skip), base=base, cycle_length=cycle_length)
    prompts = _prompts()
    new = [30, 21, 26]
    results = model.decode_batch(prompts, new)
    for prompt, result, n in zip(prompts, results, new):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, n))
    if cycle_length > 1:
        assert sum(k for r in results for k, _ in r.acceptance) > 0


def test_skipping_the_last_layer_is_the_shared_early_exit_before_it() -> None:
    """Skipping layer ``L`` drafts with layers ``1..L-1`` and the model's head, the shared early exit at ``L - 1``."""
    from economize.techniques.early_exit import EarlyExitConfig

    base = make_tiny_model()
    skip = _model(LayerSkipConfig(skip_layers=(LAYERS,)), base=base, cycle_length=4)
    exit_ = _model(EarlyExitConfig(exit_layer=LAYERS - 1, head="shared"), base=base, cycle_length=4)
    prompt = _prompts()[2]
    first, second = skip.decode(prompt, 40), exit_.decode(prompt, 40)
    assert torch.equal(first.tokens, reference_greedy(base, prompt, 40))
    assert first.acceptance == second.acceptance and first.cost.events == second.cost.events
    accepted = sum(a for _, a in first.acceptance)
    assert accepted / sum(k for k, _ in first.acceptance) > 0.2


def test_rejection_at_near_zero_temperature_is_greedy() -> None:
    base = make_tiny_model()
    model = _model(LayerSkipConfig(skip_layers=(2, 5)), base=base)
    prompts = _prompts()
    config = DecodeConfig(cycle_length=4, verifier="rejection")
    results = model.decode_batch(prompts, 24, decoding=config, selector=TopPSampler(1e-6, seed=0))
    for prompt, result in zip(prompts, results):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 24))


def test_cost_events_count_the_layers_a_draft_runs() -> None:
    model = _model(LayerSkipConfig(skip_layers=(2, 3, 5)), cycle_length=5)
    result = model.decode(_prompts()[0], 30)
    drafts = [e for e in result.cost.events if e.kind.startswith("draft")]
    assert drafts and all((e.tokens, e.first_layer, e.last_layer, e.layer_tokens) == (1, 1, 6, 3) for e in drafts)
    assert all(e.layer_indices == (1, 4, 6) for e in drafts)
    assert all((e.first_layer, e.last_layer) == (1, LAYERS) for e in result.cost.events if e.kind == "exact")
    assert sum(1 for e in drafts if e.kind == "draft") == sum(k for k, _ in result.acceptance)


# ---- search --------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("metric", ["kl", "mismatch"])
def test_greedy_search_is_deterministic_and_respects_the_kept_layers(calib, metric: str) -> None:
    config = LayerSkipConfig(num_skip=2, metric=metric, keep_first=2, keep_last=1)
    first = _model(config, calibration=calib).techniques["layer_skip"].skip_layers
    second = _model(config, calibration=calib).techniques["layer_skip"].skip_layers
    assert first == second and len(first) == 2 and set(first) <= {3, 4, 5}


def test_greedy_search_picks_the_least_harmful_layer(calib) -> None:
    """With one layer to skip, the greedy choice is the argmin of the scored single-layer sets."""
    base = make_tiny_model()
    technique = _model(LayerSkipConfig(num_skip=1), base=base, calibration=calib).techniques["layer_skip"]
    runner = LayerRunner(adapter_for(base))
    scorer = SkipScorer(runner, token_batches(calib))
    candidates = list(range(2, LAYERS))
    scores = scorer.score([frozenset({c}) for c in candidates])
    assert technique.skip_layers == (candidates[min(range(len(scores)), key=scores.__getitem__)],)
    assert scorer.passes == len(token_batches(calib)) * (1 + len(candidates))
    assert min(scores) >= 0


def test_uniform_search_needs_no_data() -> None:
    technique = _model(LayerSkipConfig(num_skip=2, search="uniform")).techniques["layer_skip"]
    assert technique.skip_layers == (3, 5)  # the middles of the halves of the candidates 2..5


def test_search_without_calibration_raises() -> None:
    with pytest.raises(ConfigError):
        _model(LayerSkipConfig(num_skip=2))


def test_registered_search_extends_the_registry(calib) -> None:
    @SKIP_SEARCHES.register("last_layers")
    class LastLayers(SkipSearch):
        needs_data = False

        def __call__(self, scorer, candidates, size):
            return tuple(candidates[-size:])

    try:
        assert _model(LayerSkipConfig(num_skip=2, search="last_layers")).techniques["layer_skip"].skip_layers == (4, 5)
    finally:
        SKIP_SEARCHES.unregister("last_layers")


# ---- save, load and combination ------------------------------------------------------------------------------------
def test_save_and_load_keep_the_searched_set_without_calibration(tmp_path, calib) -> None:
    base = make_tiny_model()
    model = _model(LayerSkipConfig(num_skip=2), base=base, calibration=calib, cycle_length=4)
    economize.save(model, tmp_path / "ls")
    loaded = economize.load(make_tiny_model(), tmp_path / "ls")
    assert loaded.techniques["layer_skip"].skip_layers == model.techniques["layer_skip"].skip_layers
    prompt = _prompts()[1]
    first, second = model.decode(prompt, 20), loaded.decode(prompt, 20)
    assert torch.equal(first.tokens, second.tokens) and first.acceptance == second.acceptance


def test_drafter_needs_a_skip_set() -> None:
    technique = LayerSkipTechnique(LayerSkipConfig(num_skip=1, search="uniform"),
                                   adapter_for(make_tiny_model()))
    with pytest.raises(RuntimeError):
        technique.drafter(None)


def test_layer_skip_on_a_pruned_model_equals_its_greedy(calib) -> None:
    from economize.techniques.prune import PruneConfig

    base = make_tiny_model()
    model = economize.apply(base, PruneConfig(method="wanda", sparsity="2:4"), LayerSkipConfig(num_skip=2),
                         decoding=DecodeConfig(cycle_length=4), calibration=calib)
    assert list(model.techniques) == ["prune", "layer_skip"]
    prompts = _prompts()
    for prompt, result in zip(prompts, model.decode_batch(prompts, 24)):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 24))


def test_greedy_search_trace_exists_before_and_resets_per_search() -> None:
    from economize.techniques.layer_skip import GreedySearch

    class _Scorer:
        def score(self, sets: list[frozenset[int]]) -> list[float]:
            return [float(sum(s)) for s in sets]

    search = GreedySearch()
    assert search.trace == []
    assert search(_Scorer(), [3, 1, 2], 2) == (1, 2)  # type: ignore[arg-type]
    assert search.trace == [(1, 1.0), (2, 3.0)]
    assert search(_Scorer(), [4, 5], 1) == (4,)  # type: ignore[arg-type]
    assert search.trace == [(4, 4.0)]


# ---- the cost footprint, the cached reference and the metric registry ----
def test_footprint_names_the_layers_a_draft_runs() -> None:
    model = economize.apply(make_tiny_model(), LayerSkipConfig(skip_layers=(2, 3, 5)))
    technique = model.techniques["layer_skip"]
    assert technique.run_layers == (1, 4, 6)
    assert technique.footprint == LayerFootprint(1, 6, None, (1, 4, 6))
    assert technique.layer_span == (1, 6)
    drafter = technique.drafter(LayerRunner(technique.adapter))
    assert isinstance(drafter, LayerSkipDrafter) and draft_footprint(drafter) == technique.footprint
    top = economize.apply(make_tiny_model(), LayerSkipConfig(skip_layers=(5, 6))).techniques["layer_skip"]
    assert top.footprint == LayerFootprint(1, 4) and top.layer_span == (1, 4)


def test_relative_cost_counts_the_skipped_layers() -> None:
    model = economize.apply(make_tiny_model(), LayerSkipConfig(skip_layers=(2, 3, 5)),
                         decoding=DecodeConfig(cycle_length=4))
    result = model.decode(torch.arange(1, 9), 16)
    drafts = [e for e in result.cost.events if e.kind.startswith("draft")]
    assert drafts and all(e.layer_tokens == 3 for e in drafts)


def test_scorer_caches_the_full_model_once_per_batch(calib) -> None:
    base = make_tiny_model()
    batches = token_batches(calib)
    scorer = SkipScorer(LayerRunner(adapter_for(base)), batches, "mismatch")
    first = scorer.score([frozenset({2}), frozenset({3})])
    second = scorer.score([frozenset({2}), frozenset({3})])
    assert first == second
    assert scorer.passes == len(batches) * (1 + 2 + 2)
    assert all(ref.dtype == torch.long for ref in scorer._references.values())  # mismatch keeps greedy tokens only
    uncached = SkipScorer(LayerRunner(adapter_for(base)), batches, "kl", cache_reference=False)
    uncached.score([frozenset({2})])
    uncached.score([frozenset({2})])
    assert uncached.passes == len(batches) * 4


def test_registered_metric_drives_the_search(calib) -> None:
    @SKIP_METRICS.register("test-neg-layer")
    class PreferHigh(SkipMetric):
        """Scores every set by the negated entropy of the skipped model, enough to check the plumbing."""

        def reference(self, log_probs):  # type: ignore[no-untyped-def]
            return log_probs.argmax(-1)

        def __call__(self, reference, log_probs):  # type: ignore[no-untyped-def]
            return (log_probs.exp() * log_probs).sum(-1)

    try:
        config = LayerSkipConfig(num_skip=1, metric="test-neg-layer")
        model = economize.apply(make_tiny_model(), config, calibration=calib)
        assert len(model.techniques["layer_skip"].skip_layers) == 1
    finally:
        SKIP_METRICS.unregister("test-neg-layer")
    with pytest.raises(ConfigError):
        LayerSkipConfig(num_skip=1, metric="test-neg-layer")


def test_greedy_search_without_a_scorer_raises() -> None:
    with pytest.raises(ConfigError):
        GreedySearch()(None, (2, 3), 1)
