"""The draft-and-refresh loop with scripted drafters, independent of any technique."""

from __future__ import annotations

import gc
from pathlib import Path

import pytest
import torch

import economize
from economize import ConfigError, DecodeConfig, GreedySelector, TopPSampler
from economize.arch import adapter_for
from economize.compat.tskip import cost_events
from economize.decoding import CycleDecoder, Drafter
from economize.decoding.capabilities import SeededSelector, selector_generator
from economize.decoding.cost import CostKind, LayerFootprint
from economize.decoding.generation import Generation
from economize.decoding.graphs import DecodeGraphs, static_cache_class
from economize.decoding.protocols import draft_footprint
from economize.runtime import KVCache, LayerRunner, StaticKVCache
from economize.techniques.draft_model import CATCHUP, DraftModelConfig
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.early_exit import EarlyExitConfig
from economize.techniques.kv_evict import KVEvictConfig
from economize.techniques.kv_quant import KIVIConfig
from helpers import VOCAB, ScriptedDrafter, ScriptedTreeDrafter, make_tiny_model, perturb, reference_greedy

MAX_NEW = 13


@pytest.fixture
def setup(tiny_model, prompt):
    greedy = reference_greedy(tiny_model, prompt, MAX_NEW + 8)
    script = torch.cat([prompt, greedy])
    runner = LayerRunner(adapter_for(tiny_model))

    def decode(drafter, max_new=MAX_NEW, selector=None, stop=None, **options):
        decoder = CycleDecoder(runner, drafter, DecodeConfig(**options))
        return decoder.decode(prompt, max_new, selector=selector, stop=stop)

    return decode, greedy[:MAX_NEW], script


def test_scripted_drafter_satisfies_the_protocol() -> None:
    assert isinstance(ScriptedDrafter(torch.zeros(1, dtype=torch.long)), Drafter)


def test_plain_decoding_is_greedy(setup) -> None:
    decode, greedy, _ = setup
    result = decode(None, cycle_length=1)
    assert result.tokens.tolist() == greedy.tolist()
    assert result.cost.count("exact") == MAX_NEW and result.cost.count("draft") == 0
    assert result.cost_summary()["relative_layer_tokens"] == pytest.approx(1.0)


def test_perfect_drafts_are_all_accepted(setup) -> None:
    decode, greedy, script = setup
    drafter = ScriptedDrafter(script)
    result = decode(drafter, cycle_length=4)
    assert result.tokens.tolist() == greedy.tolist()
    # 1 token from the prompt pass, then 3 accepted drafts + 1 exact token per cycle: 1 + 4 * 3 = 13
    assert result.acceptance == [(3, 3)] * 3 and result.mean_accepted == 3.0
    assert result.cost.count("exact") == 4 and result.cost.count("draft") == 9
    assert [e.tokens for e in result.cost.events if e.kind == "exact"] == [12, 4, 4, 4]
    # each cycle's drafter starts from the exact states of the last token that went through the exact pass
    assert drafter.synced == [[11, 15, 19]]


def test_wrong_drafts_are_all_rejected(setup) -> None:
    decode, greedy, script = setup
    result = decode(ScriptedDrafter(script, offset=1), cycle_length=4)
    assert result.tokens.tolist() == greedy.tolist()
    assert all(accepted == 0 for _, accepted in result.acceptance) and result.mean_accepted == 0.0
    assert result.cost.count("exact") == MAX_NEW


def test_partially_right_drafts(setup, prompt) -> None:
    """Drafts right at even positions only: output stays greedy, acceptance stops at the first wrong draft."""
    decode, greedy, script = setup
    corrupted = script.clone()
    corrupted[prompt.numel() + 3 :: 2] = (corrupted[prompt.numel() + 3 :: 2] + 1) % VOCAB
    result = decode(ScriptedDrafter(corrupted), cycle_length=5)
    assert result.tokens.tolist() == greedy.tolist()
    assert all(accepted < drafted for drafted, accepted in result.acceptance)
    assert any(accepted > 0 for _, accepted in result.acceptance)


def test_lossy_decoding_keeps_the_drafts(setup, prompt) -> None:
    decode, greedy, script = setup
    assert decode(ScriptedDrafter(script), cycle_length=4, lossless=False).tokens.tolist() == greedy.tolist()
    result = decode(ScriptedDrafter(script, offset=1), cycle_length=4, lossless=False)
    tokens = result.tokens.tolist()
    # cycle: one exact token, then 3 drafts shifted by one; the first token is exact
    assert tokens[0] == int(greedy[0]) and tokens[1] == (int(script[prompt.numel() + 1]) + 1) % VOCAB
    assert result.acceptance == [] and result.cost.count("exact") == 4
    assert [e.tokens for e in result.cost.events if e.kind == "exact"] == [12, 4, 4, 4]


def test_low_confidence_stops_drafting(setup) -> None:
    decode, greedy, script = setup
    # max probability of a flat-ish distribution over 128 tokens is far below 0.5
    result = decode(ScriptedDrafter(script, scale=0.1), cycle_length=4, min_confidence=0.5)
    assert result.tokens.tolist() == greedy.tolist()
    assert all(drafted == 1 for drafted, _ in result.acceptance)  # the first draft of a cycle is always made
    assert result.cost.count("draft_wasted") == result.cost.count("draft") == len(result.acceptance)
    confident = decode(ScriptedDrafter(script, scale=100.0), cycle_length=4, min_confidence=0.5)
    assert confident.cost.count("draft_wasted") == 0


@pytest.mark.parametrize("cycle_length", [1, 4])
def test_stop_token_and_predicate(setup, cycle_length: int) -> None:
    decode, greedy, script = setup
    stop_token = int(greedy[6])
    expected = greedy[: greedy.tolist().index(stop_token) + 1].tolist()
    drafter = ScriptedDrafter(script) if cycle_length > 1 else None
    assert decode(drafter, stop=stop_token, cycle_length=cycle_length).tokens.tolist() == expected
    assert decode(drafter, stop=[stop_token, -1], cycle_length=cycle_length).tokens.tolist() == expected
    length = decode(drafter, stop=lambda seq: seq.numel() >= 12 + 5, cycle_length=cycle_length).tokens
    assert length.tolist() == greedy[:5].tolist()


@pytest.mark.parametrize("max_new", [0, 1, 2, 5])
def test_output_length(setup, max_new: int) -> None:
    decode, greedy, script = setup
    for lossless in (True, False):
        result = decode(ScriptedDrafter(script), max_new=max_new, cycle_length=4, lossless=lossless)
        assert result.tokens.tolist() == greedy[:max_new].tolist()
    assert len(decode(None, max_new=0, cycle_length=1).cost) == 0


def test_sampling_requires_lossy_decoding(setup) -> None:
    decode, _, script = setup
    with pytest.raises(ConfigError, match="lossless=False"):
        decode(ScriptedDrafter(script), selector=TopPSampler(), cycle_length=4)


def test_drafting_requires_a_drafter(tiny_model) -> None:
    with pytest.raises(ConfigError, match="needs a drafting technique"):
        CycleDecoder(LayerRunner(adapter_for(tiny_model)), None, DecodeConfig(cycle_length=4))


def test_sampling(setup) -> None:
    decode, greedy, script = setup
    near_greedy = decode(None, selector=TopPSampler(top_p=1e-9), cycle_length=1, lossless=False)
    assert near_greedy.tokens.tolist() == greedy.tolist()  # top_p -> 0 keeps only the most likely token
    first = decode(ScriptedDrafter(script), selector=TopPSampler(1.5, 0.95, seed=3), cycle_length=4, lossless=False)
    again = decode(ScriptedDrafter(script), selector=TopPSampler(1.5, 0.95, seed=3), cycle_length=4, lossless=False)
    other = decode(None, selector=TopPSampler(1.5, 1.0, seed=4), cycle_length=1, lossless=False)
    assert first.tokens.tolist() == again.tokens.tolist()
    assert other.tokens.tolist() != greedy.tolist()


def test_cost_events_record_positions_and_layers(setup) -> None:
    decode, _, script = setup
    result = decode(ScriptedDrafter(script), cycle_length=3, max_new=5)
    assert cost_events(result.cost) == [
        ("exact", 12, 0), ("jump", 1, 12), ("jump", 1, 13), ("exact", 3, 12), ("jump", 1, 15), ("jump", 1, 16),
        ("exact", 3, 15),
    ]
    summary = result.cost_summary()
    assert summary["layer_tokens"] == (12 + 3 + 3) * 6 + 4 * 1
    assert summary["relative_layer_tokens"] == pytest.approx(summary["layer_tokens"] / ((12 + 5 - 1) * 6))


# ---- batches ---------------------------------------------------------------------------------------------------
@pytest.fixture
def batch(tiny_model):
    """Three prompts of different lengths, their greedy continuations and a decoder factory."""
    gen = torch.Generator().manual_seed(5)
    prompts = [torch.randint(0, VOCAB, (n,), generator=gen) for n in (11, 4, 8)]
    greedy = [reference_greedy(tiny_model, p, MAX_NEW + 8) for p in prompts]
    scripts = [torch.cat([p, g]) for p, g in zip(prompts, greedy)]
    runner = LayerRunner(adapter_for(tiny_model))

    def decode(drafter, max_new=MAX_NEW, selector=None, stop=None, **options):
        decoder = CycleDecoder(runner, drafter, DecodeConfig(**options))
        return decoder.decode_batch(prompts, max_new, selector=selector, stop=stop)

    return decode, [g[:MAX_NEW] for g in greedy], scripts, prompts


def _corrupt(scripts, prompts, every: int):
    out = []
    for script, prompt in zip(scripts, prompts):
        script = script.clone()
        script[prompt.numel() + 2 :: every] = (script[prompt.numel() + 2 :: every] + 1) % VOCAB
        out.append(script)
    return out


@pytest.mark.parametrize("cycle_length", [1, 3, 4])
@pytest.mark.parametrize("every", [1, 2, 3, 100])
def test_ragged_batch_equals_greedy_per_row(batch, cycle_length: int, every: int) -> None:
    decode, greedy, scripts, prompts = batch
    drafter = ScriptedDrafter(_corrupt(scripts, prompts, every)) if cycle_length > 1 else None
    results = decode(drafter, cycle_length=cycle_length)
    assert [r.tokens.tolist() for r in results] == [g.tolist() for g in greedy]
    assert [r.prompt_length for r in results] == [p.numel() for p in prompts]


def test_batch_rows_match_single_prompt_decoding(batch, tiny_model) -> None:
    """Every row of a batch makes the same calls as the same prompt decoded alone."""
    decode, _, scripts, prompts = batch
    corrupted = _corrupt(scripts, prompts, 3)
    together = decode(ScriptedDrafter(corrupted), cycle_length=4)
    runner = LayerRunner(adapter_for(tiny_model))
    for script, prompt, result in zip(corrupted, prompts, together):
        alone = CycleDecoder(runner, ScriptedDrafter([script]), DecodeConfig(cycle_length=4)).decode(prompt, MAX_NEW)
        assert result.tokens.tolist() == alone.tokens.tolist()
        assert result.acceptance == alone.acceptance
        assert cost_events(result.cost) == cost_events(alone.cost)


def test_rows_finish_at_their_own_stop_token(batch) -> None:
    decode, greedy, scripts, _ = batch
    stop = int(greedy[1][3])
    results = decode(ScriptedDrafter(scripts), stop=stop, cycle_length=4)
    for result, row in zip(results, greedy):
        tokens = row.tolist()
        assert result.tokens.tolist() == (tokens[: tokens.index(stop) + 1] if stop in tokens else tokens)


def test_lossy_batch_keeps_the_drafts(batch) -> None:
    decode, greedy, scripts, prompts = batch
    results = decode(ScriptedDrafter(scripts, offset=1), cycle_length=4, lossless=False)
    for result, row, script, prompt in zip(results, greedy, scripts, prompts):
        tokens = result.tokens.tolist()
        assert tokens[0] == int(row[0]) and tokens[1] == (int(script[prompt.numel() + 1]) + 1) % VOCAB
        assert len(tokens) == MAX_NEW and result.acceptance == []


@pytest.mark.parametrize("cut", [(), (0,), (1,), (2,)])
def test_tree_drafts_equal_greedy(batch, cut: tuple[int, ...]) -> None:
    decode, greedy, scripts, _ = batch
    results = decode(ScriptedTreeDrafter(scripts, cut=cut), cycle_length=4)
    assert [r.tokens.tolist() for r in results] == [g.tolist() for g in greedy]
    for b, result in enumerate(results):
        depth = 3 - b % 2  # rows with an odd index draft one level less
        if not cut:
            assert all(accepted == min(depth, drafted // 2) for drafted, accepted in result.acceptance[:-1])
        else:
            assert all(accepted <= min(cut) for _, accepted in result.acceptance)
        assert all(drafted % 2 == 0 for drafted, _ in result.acceptance)


def test_trees_need_a_tree_verifier(batch) -> None:
    decode, _, scripts, _ = batch
    with pytest.raises(ConfigError, match="draft trees"):
        decode(ScriptedTreeDrafter(scripts), cycle_length=4, lossless=False)


# ---- per-row selectors, stop conditions and lengths -------------------------------------------------------------
@pytest.mark.parametrize("tree", [False, True], ids=["chain", "tree"])
def test_rows_use_their_own_selector(batch, tiny_model, tree: bool) -> None:
    """A default greedy row, a sampled row and a second greedy selector in one batch: the greedy rows follow the
    model's argmax and the sampled row equals the same prompt decoded alone with the same seed."""
    _, greedy, scripts, prompts = batch
    runner = LayerRunner(adapter_for(tiny_model))
    config = DecodeConfig(cycle_length=4, verifier="rejection")

    def drafter(rows):
        chosen = [scripts[r] for r in rows]
        return ScriptedTreeDrafter(chosen) if tree else ScriptedDrafter(chosen, scale=2.0)

    selectors = [None, TopPSampler(seed=3), GreedySelector()]
    together = CycleDecoder(runner, drafter([0, 1, 2]), config).decode_batch(prompts, MAX_NEW, selectors=selectors)
    assert together[0].tokens.tolist() == greedy[0].tolist()
    assert together[2].tokens.tolist() == greedy[2].tolist()
    if tree:  # the tree drafter's shape depends on the row index: decode the sampled row as the second of two
        alone = CycleDecoder(runner, drafter([0, 1]), config).decode_batch(
            prompts[:2], MAX_NEW, selectors=[None, TopPSampler(seed=3)])[1]
    else:
        alone = CycleDecoder(runner, drafter([1]), config).decode(prompts[1], MAX_NEW, selector=TopPSampler(seed=3))
    assert together[1].tokens.tolist() == alone.tokens.tolist()
    assert together[1].acceptance == alone.acceptance


@pytest.mark.parametrize("cycle_length", [1, 4])
def test_rows_use_their_own_length_and_stop(batch, tiny_model, cycle_length: int) -> None:
    _, greedy, scripts, prompts = batch
    prompts, scripts, greedy = [*prompts, prompts[0]], [*scripts, scripts[0]], [*greedy, greedy[0]]
    stop = int(greedy[2][4])
    drafter = ScriptedDrafter(scripts) if cycle_length > 1 else None
    decoder = CycleDecoder(LayerRunner(adapter_for(tiny_model)), drafter, DecodeConfig(cycle_length=cycle_length))
    results = decoder.decode_batch(prompts, [2, 9, MAX_NEW, 0], stops=[None, None, stop, None])
    assert results[0].tokens.tolist() == greedy[0][:2].tolist()
    assert results[1].tokens.tolist() == greedy[1][:9].tolist()
    row = greedy[2].tolist()
    assert results[2].tokens.tolist() == row[: row.index(stop) + 1]
    assert results[3].tokens.tolist() == [] and results[3].prompt_length == prompts[3].numel()


def test_per_row_arguments_are_checked(batch) -> None:
    decode, _, scripts, prompts = batch
    del decode
    decoder = CycleDecoder(LayerRunner(adapter_for(make_tiny_model())), ScriptedDrafter(scripts),
                           DecodeConfig(cycle_length=4))
    with pytest.raises(ValueError, match="selectors"):
        decoder.decode_batch(prompts, 4, selectors=[None])
    with pytest.raises(ValueError, match="max_new_tokens"):
        decoder.decode_batch(prompts, [4, 4])
    with pytest.raises(ConfigError, match="deterministic"):  # the greedy verifier cannot verify a sampled row
        decoder.decode_batch(prompts, 4, selectors=[None, TopPSampler(), None])


# ---- plain decoding with a read-back window ----------------------------------------------------------------------
@pytest.mark.parametrize("interval", [2, 3, 8, 64])
def test_read_back_window_changes_nothing(batch, tiny_model, interval: int) -> None:
    """Plain decoding reads tokens back every ``sync_interval`` steps; outputs and cost logs equal step-by-step
    decoding, also for rows with their own lengths, stop tokens and stop predicates that end inside a window."""
    _, greedy, _, prompts = batch
    runner = LayerRunner(adapter_for(tiny_model))
    stop_token = int(greedy[2][4])
    stops = [None, lambda seq: seq.numel() >= 4 + 6, stop_token, None]
    prompts = [*prompts, prompts[0]]

    def run(sync_interval: int):
        decoder = CycleDecoder(runner, None, DecodeConfig(cycle_length=1, sync_interval=sync_interval))
        return decoder.decode_batch(prompts, [3, MAX_NEW, MAX_NEW, 0], stops=stops)

    windowed, stepwise = run(interval), run(1)
    assert [r.tokens.tolist() for r in windowed] == [r.tokens.tolist() for r in stepwise]
    assert [cost_events(r.cost) for r in windowed] == [cost_events(r.cost) for r in stepwise]
    assert windowed[1].tokens.tolist() == greedy[1][:6].tolist()
    assert windowed[2].tokens.tolist()[-1] == stop_token
    sampled = [CycleDecoder(runner, None, DecodeConfig(cycle_length=1, lossless=False, sync_interval=s))
               .decode_batch(prompts[:3], MAX_NEW, selectors=[TopPSampler(seed=b) for b in range(3)])
               for s in (interval, 1)]
    assert [r.tokens.tolist() for r in sampled[0]] == [r.tokens.tolist() for r in sampled[1]]


def test_sync_interval_is_checked() -> None:
    with pytest.raises(ConfigError, match="sync_interval"):
        DecodeConfig(sync_interval=0)


# ---- lossy two-model drafting and the top-p sampler ----
def _short_prompts(lengths: tuple[int, ...] = (12, 7)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in lengths]


@pytest.fixture(scope="module")
def draft_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("audit_draft")
    make_tiny_model(seed=5, num_layers=2).save_pretrained(path)
    return path


@pytest.mark.parametrize("max_new", range(2, 12))
@torch.no_grad()
def test_lossy_draft_model_logs_one_call_per_token(draft_model_dir: Path, max_new: int) -> None:
    """Lossy decoding commits every draft, so exact calls plus draft steps equal the generated tokens, also when
    the row ends inside a cycle that began with a draft-model catch-up call."""
    model = economize.apply(make_tiny_model(), DraftModelConfig(model=str(draft_model_dir)))
    result = model.decode(_short_prompts()[0], max_new, decoding=DecodeConfig(cycle_length=4, lossless=False))
    assert result.tokens.numel() == max_new
    assert result.cost.count("exact") + result.cost.count("draft") == max_new
    assert result.cost.count(CATCHUP) >= 1


@torch.no_grad()
def test_lossy_draft_model_stop_inside_drafts_keeps_its_draft_step(draft_model_dir: Path) -> None:
    model = economize.apply(make_tiny_model(), DraftModelConfig(model=str(draft_model_dir)))
    prompt = _short_prompts()[0]
    decoding = DecodeConfig(cycle_length=4, lossless=False)
    tokens = model.decode(prompt, 16, decoding=decoding).tokens.tolist()
    # a token first produced by a draft step after the first cycle (index 1 + 4k + j, j in 1..2)
    index = next(i for i in range(5, 16) if (i - 1) % 4 in (1, 2) and tokens[i] not in tokens[:i])
    stopped = model.decode(prompt, 16, decoding=decoding, stop=tokens[index])
    assert stopped.tokens.tolist() == tokens[: index + 1]
    assert stopped.cost.count("exact") + stopped.cost.count("draft") == index + 1


@pytest.mark.parametrize("kwargs", [dict(temperature=0.0), dict(temperature=-1.0), dict(top_p=0.0),
                                    dict(top_p=1.5)])
def test_top_p_sampler_rejects_settings_without_a_distribution(kwargs: dict[str, float]) -> None:
    with pytest.raises(ConfigError):
        TopPSampler(**kwargs)


def test_top_p_sampler_call_draws_like_select_many() -> None:
    logits = torch.randn(VOCAB, generator=torch.Generator().manual_seed(2))
    one, many = TopPSampler(0.7, 0.8, seed=9), TopPSampler(0.7, 0.8, seed=9)
    assert [one(logits, torch.empty(0)) for _ in range(20)] == [int(many.select_many(logits[None])[0])
                                                               for _ in range(20)]


# ---- sampler generators, captured layers and the drafter's cost footprint ----
# ---- capture -------------------------------------------------------------------------------------------------------
class ObservingDrafter(ScriptedDrafter):
    observed_layers = (2,)


# ---- cost footprint ------------------------------------------------------------------------------------------------
class SplitDrafter(ScriptedDrafter):
    footprint = LayerFootprint.of([1, 3], module=None)


# ---- TopPSampler ----------------------------------------------------------------------------------------------------
def test_top_p_sampler_draws_on_the_logits_device_reproducibly() -> None:
    logits = torch.randn(64, 20, generator=torch.Generator().manual_seed(0))
    a, b = TopPSampler(seed=3), TopPSampler(seed=3)
    assert a.generator is None  # created lazily, on the device of the first logits
    assert torch.equal(a.select_many(logits), b.select_many(logits))
    assert a.generator is not None and a.generator.device == logits.device
    assert isinstance(a, SeededSelector)
    assert selector_generator(a, logits.device) is a.generator_for(torch.device("cpu"))
    assert not torch.equal(TopPSampler(seed=4).select_many(logits), b.select_many(logits))


def test_ordinary_decoding_captures_nothing_and_runs_no_drafter() -> None:
    """With ``cycle_length=1`` the drafter never runs, so its observed layers are not captured either."""
    model = make_tiny_model(seed=1)
    prompt = torch.randint(1, VOCAB, (7,), generator=torch.Generator().manual_seed(1))
    drafter = ObservingDrafter([torch.cat([prompt, reference_greedy(model, prompt, 10)])])
    decoder = CycleDecoder(LayerRunner(adapter_for(model)), drafter, DecodeConfig(cycle_length=1))
    assert decoder.active_drafter is None and decoder.capture == ()
    generation = Generation(decoder, [prompt], [6], [TopPSampler()], [lambda *a: False],
                            TopPSampler())
    assert generation.drafter is None and generation.capture == ()
    result = decoder.decode(prompt, 6)
    assert result.tokens.tolist() == reference_greedy(model, prompt, 6).tolist()
    assert drafter.synced == [[]]  # never observed

    chain = CycleDecoder(LayerRunner(adapter_for(model)), drafter, DecodeConfig(cycle_length=3))
    assert chain.active_drafter is drafter and chain.capture == (2,)


def test_drafting_logs_the_drafter_footprint() -> None:
    model = make_tiny_model(seed=1)
    prompt = torch.randint(1, VOCAB, (7,), generator=torch.Generator().manual_seed(1))
    script = torch.cat([prompt, reference_greedy(model, prompt, 12)])
    assert draft_footprint(ScriptedDrafter([script])) == LayerFootprint(1, 1)
    assert draft_footprint(SplitDrafter([script])) is SplitDrafter.footprint
    result = CycleDecoder(LayerRunner(adapter_for(model)), SplitDrafter([script]), DecodeConfig(cycle_length=4)) \
        .decode(prompt, 8)
    drafts = [e for e in result.cost.events if e.kind == CostKind.DRAFT]
    assert drafts and all(e.layer_indices == (1, 3) for e in drafts)
    exact = [e for e in result.cost.events if e.kind == CostKind.EXACT]
    assert exact and all(e.kind == "exact" for e in exact)  # CostKind values are the strings logs always used


@torch.no_grad()
def test_graphed_exact_passes_keep_their_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drafters that read the exact pass's context (EAGLE-3, vendor MTP) decode with ``cuda_graphs=True``.

    The graph path is exercised on the CPU, where :class:`~economize.runtime.StaticStep` runs eagerly.
    """
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    monkeypatch.setattr(CycleDecoder, "_graphable", lambda self: True)
    model = economize.apply(make_tiny_model(), Eagle3Config())
    perturb(model.techniques["eagle3"], 0.05, seed=1)
    prompts = _short_prompts()
    eager = model.decode_batch(prompts, 10, decoding=DecodeConfig(cycle_length=4))
    graphed_config = DecodeConfig(cycle_length=4, cuda_graphs=True)
    graphed = model.decode_batch(prompts, 10, decoding=graphed_config)
    assert model.decoder(graphed_config).graphs is not None
    assert [r.tokens.tolist() for r in graphed] == [r.tokens.tolist() for r in eager]


# ---- graphs --------------------------------------------------------------------------------------------------------
def test_static_cache_variants() -> None:
    assert static_cache_class("positional") is StaticKVCache
    assert static_cache_class("hybrid") is None
    assert static_cache_class("no such kind") is None
    assert DecodeGraphs.supports_cache("positional") and not DecodeGraphs.supports_cache("hybrid")


# ---- memory --------------------------------------------------------------------------------------------------------
_JUMP = economize.JumpConfig(target_layer=2, source_layer=3, exit_layer=4, adaptation="full", predictor="linear")


@pytest.mark.parametrize(
    "configs",
    [
        (_JUMP,),
        (EarlyExitConfig(exit_layer=3, adaptation="full"),),
        (Eagle3Config(),),
        (_JUMP, KIVIConfig(bits=4, group_size=2, residual_length=4)),
        (_JUMP, KVEvictConfig(method="streaming", recent=6, sinks=2)),
        (_JUMP, KVEvictConfig(method="snapkv", heavy=4, observation_window=2, kernel_size=3)),
    ],
    ids=["jump", "early_exit", "eagle3", "jump+kivi", "jump+streaming", "jump+snapkv"],
)
@torch.no_grad()
def test_eager_decoding_leaves_no_reference_cycles(configs) -> None:
    """An eager decode builds a new drafter and KV cache per call. If either sat in a reference cycle, every call
    would leave its whole KV cache allocated until the cyclic garbage collector happened to run, which ran a 31B
    model out of memory after a few prompts."""
    model = economize.apply(make_tiny_model(), *configs)
    prompt = torch.randint(0, VOCAB, (24,), generator=torch.Generator().manual_seed(0))
    decoding = DecodeConfig(cycle_length=4)
    model.decode(prompt, 8, decoding=decoding)  # builds the lazily created state outside the measurement
    gc.collect()
    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        model.decode(prompt, 8, decoding=decoding)
        gc.collect()
        leaked = sorted({type(o).__qualname__ for o in gc.garbage if isinstance(o, (KVCache, Drafter))})
    finally:
        gc.set_debug(0)
        gc.garbage.clear()
    assert leaked == []
