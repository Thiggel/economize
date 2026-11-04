"""KV-cache eviction: StreamingLLM (sinks and a recent window), H2O (heavy hitters, during decoding or once at
prefill) and SnapKV (prompt compression from an observation window)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import economize
from economize import DecodeConfig
from economize.errors import ConfigError, UnknownNameError
from economize.runtime import phase
from economize.techniques.kv_evict import (
    EVICTION_METHODS,
    EvictionCache,
    KVEvictConfig,
    KVEvictTechnique,
    PendingScores,
    WindowEvictionCache,
)
from economize.techniques.kv_evict.policies import keep_top
from helpers import VOCAB, make_tiny_model, reference_greedy

PREFIX, TOTAL = 16, 48
#: Bytes of one token's K and V over all layers of the tiny model (6 layers, 2 KV heads of width 16, fp32).
TOKEN_BYTES = 6 * 2 * 2 * 16 * 4


def _sequence(seed: int = 0) -> torch.Tensor:
    return torch.randint(0, VOCAB, (1, TOTAL), generator=torch.Generator().manual_seed(seed))


def _prompts() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in (12, 7, 20)]


def _teacher_forced(
    model: economize.EfficientModel, seq: torch.Tensor, wrap: bool = True
) -> tuple[torch.Tensor, object]:
    """Logits of ``seq`` from a prefill of :data:`PREFIX` tokens and single-token decode steps, and the cache."""
    runner = model.runner
    cache = runner.new_cache(64)
    if wrap:
        cache = model.techniques["kv_evict"].wrap_cache(cache)
    ctx = runner.context(cache, torch.arange(PREFIX)[None], trusted=True)
    with phase("prefill"):
        out = [runner.logits(runner.forward(seq[:, :PREFIX], ctx).final)[0]]
    for t in range(PREFIX, seq.shape[1]):
        ctx = runner.context(cache, torch.tensor([[t]]), trusted=True)
        with phase("decode"):
            out.append(runner.logits(runner.forward(seq[:, t : t + 1], ctx).final)[0])
    return torch.cat(out).float(), cache


def _kl(logits: torch.Tensor, reference: torch.Tensor) -> float:
    return F.kl_div(logits.log_softmax(-1), reference.log_softmax(-1), log_target=True, reduction="batchmean").item()


def _identity_jump(base: torch.nn.Module, *configs: object, cycle_length: int = 4) -> economize.EfficientModel:
    jump = economize.JumpConfig(target_layer=0, source_layer=0, adaptation="frozen", lora_rank=2)
    model = economize.apply(base, jump, *configs, decoding=DecodeConfig(cycle_length=cycle_length))
    for name, parameter in model.techniques["jump"].predictor.named_parameters():
        if name != "out_gain":
            parameter.zero_()
    return model


TRIVIAL = [
    KVEvictConfig(recent=1000),
    KVEvictConfig(method="h2o", recent=1000),
    KVEvictConfig(method="h2o_prefill", recent=1000),
    KVEvictConfig(method="snapkv", heavy=1000, observation_window=4),
    KVEvictConfig(method="pyramidkv", heavy=1000, observation_window=4),
]
#: Methods that compress the prompt once and keep every generated entry.
PROMPT_ONLY = ("snapkv", "h2o_prefill", "pyramidkv")
LOSSY = [
    KVEvictConfig(recent=8),
    KVEvictConfig(recent=4, sinks=0),
    KVEvictConfig(method="h2o", recent=4, heavy=8),
    KVEvictConfig(method="h2o", recent=4, heavy=0),
    KVEvictConfig(method="h2o_prefill", recent=4, heavy=4),
    KVEvictConfig(method="snapkv", heavy=4, observation_window=4),
    KVEvictConfig(method="snapkv", heavy=4, observation_window=4, pooling="avg", kernel_size=3),
    KVEvictConfig(method="pyramidkv", heavy=4, observation_window=4, beta=4),
]


def test_methods_registered() -> None:
    assert {"streaming", "h2o", "h2o_prefill", "snapkv", "pyramidkv"} <= set(EVICTION_METHODS.names())
    assert not EVICTION_METHODS.get("streaming").needs_scores and EVICTION_METHODS.get("h2o").needs_scores


@pytest.mark.parametrize(
    "kwargs",
    [dict(method="lru"), dict(sinks=-1), dict(recent=-1), dict(heavy=-1), dict(observation_window=0),
     dict(kernel_size=4), dict(pooling="median"), dict(layers=(0,))],
)
def test_config_validation(kwargs: dict) -> None:
    with pytest.raises((ConfigError, UnknownNameError)):
        KVEvictConfig(**kwargs)


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
@pytest.mark.parametrize("config", TRIVIAL, ids=lambda c: c.method)
def test_trivial_settings_are_exact(attn: str, config: KVEvictConfig) -> None:
    seq = _sequence()
    base = make_tiny_model(attn=attn)
    plain = economize.apply(base)
    expected, _ = _teacher_forced(plain, seq, wrap=False)
    plain.detach()
    model = economize.apply(base, config)
    logits, cache = _teacher_forced(model, seq)
    assert torch.equal(logits, expected) and not cache.evicted
    assert model.techniques["kv_evict"].kv_stats()["kv_compression"] == 1.0
    assert torch.equal(model.decode(seq[0, :12], 16).tokens, reference_greedy(base, seq[0, :12], 16))


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
@pytest.mark.parametrize("config", LOSSY, ids=lambda c: f"{c.method}-{c.recent}-{c.heavy}-{c.sinks}-{c.pooling}")
def test_eviction_changes_the_output_and_saves_memory(attn: str, config: KVEvictConfig) -> None:
    seq = _sequence()
    base = make_tiny_model(attn=attn)
    reference = base(seq).logits[0].float()
    model = economize.apply(base, config)
    logits, cache = _teacher_forced(model, seq)
    stats = model.techniques["kv_evict"].kv_stats()
    assert cache.evicted and _kl(logits, reference) > 1e-3
    prompt_only = config.method in PROMPT_ONLY
    assert stats["kv_compression"] > (1.1 if prompt_only else 1.5)  # prompt-only methods: 16 prompt entries to 8
    assert stats["kv_bytes_read"] < stats["kv_bytes_read_full"]


def test_streaming_keeps_sinks_and_recent_window() -> None:
    model = economize.apply(make_tiny_model(), KVEvictConfig(recent=8))
    _, cache = _teacher_forced(model, _sequence())
    assert isinstance(cache, WindowEvictionCache)
    keep = cache.kept(0)[0, 0, : TOTAL - 1]
    assert keep[:4].all() and keep[-8:].all() and not keep[4:-8].any()
    stats = model.techniques["kv_evict"].kv_stats()
    assert stats["kv_bytes"] == (4 + 8) * TOKEN_BYTES
    assert stats["kv_bytes_full"] == (TOTAL - 1) * TOKEN_BYTES


def _streaming_runs(attn: str, cycle: int) -> tuple[torch.Tensor, list[list[int]], list[dict], torch.Tensor]:
    config = KVEvictConfig(recent=8, sinks=3)
    base = make_tiny_model(attn=attn)
    model = economize.apply(base, config) if cycle == 1 else _identity_jump(base, config, cycle_length=cycle)
    logits, cache = _teacher_forced(model, _sequence())
    kept = torch.stack([cache.kept(layer) for layer in range(6)])
    results = model.decode_batch(_prompts(), [24, 17, 30])
    stats = [model.techniques["kv_evict"].kv_stats(row) for row in range(3)]
    return logits, [r.tokens.tolist() for r in results], stats, kept


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
@pytest.mark.parametrize("cycle", [1, 4])
def test_streaming_window_cache_decodes_like_keep_masks(attn: str, cycle: int, monkeypatch) -> None:
    """The window cache of StreamingLLM (one cut per row) gives the logits, tokens, kept entries and byte counts of
    the keep-mask cache, with and without drafting."""
    window = _streaming_runs(attn, cycle)
    # The same method on the general keep-mask cache.
    monkeypatch.setattr(KVEvictTechnique, "make_wrapper", lambda self, cache: EvictionCache(cache, self))
    masks = _streaming_runs(attn, cycle)
    assert torch.equal(window[0], masks[0])
    assert window[1] == masks[1] and window[2] == masks[2]
    assert torch.equal(window[3], masks[3])


def test_eviction_restricted_to_layers() -> None:
    seq = _sequence()
    every = economize.apply(make_tiny_model(), KVEvictConfig(recent=8))
    _teacher_forced(every, seq)
    one = economize.apply(make_tiny_model(), KVEvictConfig(recent=8, layers=(3,)))
    _teacher_forced(one, seq)
    full = every.techniques["kv_evict"].kv_stats()["kv_bytes_full"]
    stored_every = every.techniques["kv_evict"].kv_stats()["kv_bytes"]
    stored_one = one.techniques["kv_evict"].kv_stats()["kv_bytes"]
    assert stored_every < stored_one < full
    assert stored_one == full - (TOTAL - 1 - 12) * TOKEN_BYTES / 6


def test_snapkv_compresses_the_prompt_only() -> None:
    model = economize.apply(make_tiny_model(), KVEvictConfig(method="snapkv", heavy=4, observation_window=4))
    _, cache = _teacher_forced(model, _sequence())
    kept = cache.keep[0][0, :, :PREFIX].sum(-1)
    assert (kept == 4 + 4).all()  # heavy entries plus the observation window, per KV head
    assert cache.keep[0][0, :, PREFIX : TOTAL - 1].all()  # generated entries stay


def _pyramid_budget(q_len: int, capacity: int, window: int, beta: int, layer: int, layers: int) -> int:
    """Prompt entries outside the window that layer ``layer`` keeps, transcribed from ``PyramidKVCluster.
    update_kv`` of KVCache-Factory (``max_capacity_prompt = capacity``, ``window_size = window``)."""
    min_num = (capacity - window) // beta
    max_num = (capacity - window) * 2 - min_num
    if max_num >= q_len - window:
        max_num = q_len - window
        min_num = (capacity - window) * 2 - max_num
    steps = (max_num - min_num) // (layers - 1)
    max_capacity_prompt = max_num - layer * steps
    if q_len < capacity:
        return q_len - window  # returned uncompressed
    if q_len < (capacity - window) * 2:
        return capacity - window  # SnapKV branch
    return max_capacity_prompt


@pytest.mark.parametrize("q_len", [5, 9, 13, 16, 40, 200, 4096])
@pytest.mark.parametrize(("heavy", "window", "beta"), [(6, 2, 4), (4, 4, 20), (2048 - 8, 8, 20), (96, 32, 5)])
def test_pyramidkv_budget_matches_kvcache_factory(q_len: int, heavy: int, window: int, beta: int) -> None:
    config = KVEvictConfig(method="pyramidkv", heavy=heavy, observation_window=window, beta=beta)
    method = EVICTION_METHODS.get("pyramidkv")(config)
    for layers in (6, 32, 60):
        for layer in range(layers):
            expected = max(_pyramid_budget(q_len, heavy + window, window, beta, layer, layers), 0)
            assert method.budget(q_len, layer, layers) == expected


def test_pyramidkv_keeps_snapkv_choice_at_each_layer_budget() -> None:
    """Every layer keeps exactly what SnapKV keeps with ``heavy`` set to that layer's PyramidKV budget, so the
    scores and pooling are SnapKV's and only the budget varies with depth (11, 9, 7, 5, 3, 1 here; 6 on average)."""
    seq = _sequence()
    settings = {"observation_window": 2, "kernel_size": 3}
    model = economize.apply(make_tiny_model(), KVEvictConfig(method="pyramidkv", heavy=6, beta=4, **settings))
    _, cache = _teacher_forced(model, seq)
    budgets = [11, 9, 7, 5, 3, 1]
    for layer, budget in enumerate(budgets):
        keep = cache.keep[layer][0]
        assert (keep[:, :PREFIX].sum(-1) == budget + 2).all()
        assert keep[:, PREFIX : TOTAL - 1].all()  # generated entries stay
        snap = economize.apply(make_tiny_model(), KVEvictConfig(method="snapkv", heavy=budget, **settings))
        _, reference = _teacher_forced(snap, seq)
        assert torch.equal(keep[:, :PREFIX], reference.keep[layer][0, :, :PREFIX])
    assert sum(budgets) == 6 * len(budgets)


def test_keep_top_takes_a_budget_per_row() -> None:
    generator = torch.Generator().manual_seed(0)
    scores = torch.rand(3, 2, 10, generator=generator)
    candidate = torch.ones(3, 2, 10, dtype=torch.bool)
    candidate[..., 8:] = False
    budgets = [0, 3, 12]
    keep = torch.ones(3, 2, 10, dtype=torch.bool)
    keep_top(keep, candidate, scores, torch.tensor(budgets))
    for row, budget in enumerate(budgets):
        expected = torch.ones(1, 2, 10, dtype=torch.bool)
        keep_top(expected, candidate[row : row + 1], scores[row : row + 1], budget)
        assert torch.equal(keep[row : row + 1], expected)


def test_h2o_prefill_compresses_the_prompt_only() -> None:
    model = economize.apply(make_tiny_model(), KVEvictConfig(method="h2o_prefill", recent=4, heavy=4))
    _, cache = _teacher_forced(model, _sequence())
    for keep in cache.keep.tensors.values():
        assert (keep[0, :, :PREFIX].sum(-1) == 4 + 4).all()  # heavy hitters plus the recent prompt entries
        assert keep[0, :, PREFIX - 4 : TOTAL - 1].all()  # the recent window and every generated entry stay


def test_h2o_prefill_selects_the_heavy_hitters_of_the_prompt_attention() -> None:
    """The prompt entries kept per KV head are KVCache-Factory's ``H2OKVCluster`` choice: the ``heavy`` entries
    before the recent window with the largest attention summed over all prompt queries (and over the query heads of
    the KV head's group), plus the window."""
    seq = _sequence()
    base = make_tiny_model(attn="eager")
    attentions = base(seq[:, :PREFIX], output_attentions=True).attentions
    model = economize.apply(base, KVEvictConfig(method="h2o_prefill", recent=4, heavy=4))
    _, cache = _teacher_forced(model, seq)
    for layer, probs in enumerate(attentions):
        kv_heads = cache.keep[layer].shape[1]
        received = probs[0].float().sum(1).view(kv_heads, -1, PREFIX).sum(1)  # [H_kv, keys]
        heavy = received[:, : PREFIX - 4].topk(4, dim=-1).indices
        expected = torch.zeros(kv_heads, PREFIX, dtype=torch.bool)
        expected.scatter_(-1, heavy, True)
        expected[:, PREFIX - 4 :] = True
        assert torch.equal(cache.keep[layer][0, :, :PREFIX], expected)


# ---- composition ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [KVEvictConfig(method="snapkv", heavy=4, observation_window=4), KVEvictConfig(recent=8),
     KVEvictConfig(method="h2o", recent=4, heavy=4), KVEvictConfig(method="h2o_prefill", recent=4, heavy=4)],
    ids=lambda c: c.method,
)
def test_eviction_with_jump(config: KVEvictConfig) -> None:
    """At cycle length 1 every method decodes as without the drafter. SnapKV and prefill-only H2O act on the
    prompt only, so lossless drafting keeps their output for any cycle; StreamingLLM and H2O evict at exact passes,
    whose spacing depends on the cycle, so their output changes with it (documented in docs/depth_kv.md)."""
    base = make_tiny_model()
    prompts, lengths = _prompts(), [24, 17, 30]
    plain = economize.apply(base, config, decoding=DecodeConfig(cycle_length=1))
    expected = [plain.decode(p, n).tokens for p, n in zip(prompts, lengths)]
    batched = [r.tokens for r in plain.decode_batch(prompts, lengths)]
    plain.detach()
    assert all(torch.equal(a, b) for a, b in zip(expected, batched))
    model = _identity_jump(base, config, cycle_length=1)
    assert all(torch.equal(r.tokens, e) for r, e in zip(model.decode_batch(prompts, lengths), expected))
    model.detach()
    model = _identity_jump(base, config, cycle_length=4)
    results = model.decode_batch(prompts, lengths)
    assert [len(r.tokens) for r in results] == lengths
    if config.method in PROMPT_ONLY:
        assert all(torch.equal(r.tokens, e) for r, e in zip(results, expected))
    compression = [model.techniques["kv_evict"].kv_stats(row)["kv_compression"] for row in range(3)]
    assert compression[0] > 1.0 and compression[2] > 1.0
    if config.method in PROMPT_ONLY:
        assert compression[1] == 1.0  # a 7-token prompt fits the budget (4 heavy entries plus a window of 4)


def test_streaming_bytes_count_committed_entries_after_rollback() -> None:
    """Rejected drafts are rolled back; the accounting of each row counts sinks plus the recent committed window."""
    base = make_tiny_model()
    prompts, lengths = _prompts(), [24, 17, 30]
    model = _identity_jump(base, KVEvictConfig(recent=8))
    results = model.decode_batch(prompts, lengths)
    assert any(drafted > accepted for r in results for drafted, accepted in r.acceptance)
    for row in range(3):
        assert model.techniques["kv_evict"].kv_stats(row)["kv_bytes"] == (4 + 8) * TOKEN_BYTES


def test_eviction_with_prune(calib) -> None:
    from economize.techniques.prune import PruneConfig

    prune = PruneConfig(method="wanda", sparsity="2:4")
    prompt = _prompts()[0]
    expected = economize.apply(make_tiny_model(), prune, calibration=calib).decode(prompt, 20).tokens
    model = economize.apply(make_tiny_model(), prune, KVEvictConfig(method="h2o", recent=1000), calibration=calib)
    assert torch.equal(model.decode(prompt, 20).tokens, expected)
    model = economize.apply(make_tiny_model(), prune, KVEvictConfig(method="h2o", recent=4, heavy=4), calibration=calib)
    model.decode(prompt, 20)
    assert model.techniques["kv_evict"].kv_stats()["kv_compression"] > 1.5


# ---- per-method options, SnapKV's tie-breaking and the head geometry ----
@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"method": "streaming", "heavy": 8}, "heavy"),
        ({"method": "streaming", "pooling": "avg"}, "pooling"),
        ({"method": "h2o", "observation_window": 8}, "observation_window"),
        ({"method": "h2o", "kernel_size": 3}, "kernel_size"),
        ({"method": "snapkv", "recent": 8}, "recent"),
        ({"method": "snapkv", "beta": 8}, "beta"),
        ({"method": "pyramidkv", "recent": 8}, "recent"),
    ],
)
def test_kv_evict_rejects_knobs_the_method_does_not_read(kwargs: dict, name: str) -> None:
    with pytest.raises(ConfigError, match=name):
        KVEvictConfig(**kwargs)


def test_kv_evict_method_options() -> None:
    assert EVICTION_METHODS.get("streaming").options == {"recent"}
    assert EVICTION_METHODS.get("h2o").options == {"recent", "heavy"}
    assert EVICTION_METHODS.get("h2o_prefill").options == {"recent", "heavy"}
    assert EVICTION_METHODS.get("snapkv").options == {"heavy", "observation_window", "kernel_size", "pooling"}
    assert EVICTION_METHODS.get("pyramidkv").options == EVICTION_METHODS.get("snapkv").options | {"beta"}
    assert KVEvictConfig(method="pyramidkv").beta == 20
    config = KVEvictConfig(method="snapkv")
    assert (config.observation_window, config.kernel_size, config.pooling) == (32, 7, "max")


def test_keep_top_breaks_ties_by_slot_whatever_the_buffer_length() -> None:
    for slots in (8, 16, 64):
        keep = torch.ones(1, 1, slots, dtype=torch.bool)
        candidate = torch.zeros_like(keep)
        candidate[..., :6] = True
        scores = torch.zeros(1, 1, slots)
        scores[..., 1:5] = 1.0  # a plateau, as max pooling makes
        keep_top(keep, candidate, scores, 2)
        assert keep[0, 0, :6].tolist() == [False, True, True, False, False, False]
        assert keep[0, 0, 6:].all()


def test_kv_evict_reads_the_query_group_from_the_head_geometry() -> None:
    model = economize.apply(make_tiny_model(), KVEvictConfig(recent=8))
    technique = model.techniques["kv_evict"]
    geometry = technique.adapter.head_geometry(1)
    assert technique.query_group(0) == geometry.heads // geometry.kv_heads


def test_kv_cache_techniques_clash_by_family() -> None:
    from economize.core.metadata import Family
    from economize.errors import IncompatibleTechniquesError
    from economize.techniques.kv_quant import FP8KVConfig, KIVIConfig, KVCacheTechnique

    assert KVCacheTechnique.__abstractmethods__ == {"make_wrapper"}
    with pytest.raises(IncompatibleTechniquesError, match="quantize the KV cache"):
        economize.apply(make_tiny_model(), KIVIConfig(), FP8KVConfig())
    families = {name: economize.TECHNIQUES.get(name).info.family for name in ("kivi", "kv_fp8", "kv_evict")}
    quantization, eviction = Family.KV_QUANTIZATION, Family.KV_EVICTION
    assert families == {"kivi": quantization, "kv_fp8": quantization, "kv_evict": eviction}
    model = economize.apply(make_tiny_model(), KIVIConfig(), KVEvictConfig(recent=8))  # different families combine
    assert {"kivi", "kv_evict"} <= set(model.techniques)


# ---- H2O's attention tap and pending scores ----
VERIFY_PREFIX = 12  # prompt length before the verifying pass


def _h2o_scores_after_a_verify_pass(attn: str, verified: int, accepted: int) -> tuple[dict, dict]:
    """Scores and keep masks of an H2O cache at the start of the pass that follows a verifying pass over
    ``verified`` tokens of which ``accepted`` were accepted (the next pass's frontier)."""
    model = economize.apply(make_tiny_model(attn=attn), KVEvictConfig(method="h2o", recent=2, heavy=4))
    runner = model.runner
    cache = model.techniques["kv_evict"].wrap_cache(runner.new_cache(64))
    seq = torch.randint(0, VOCAB, (1, VERIFY_PREFIX + 4), generator=torch.Generator().manual_seed(2))
    with phase("prefill"):
        runner.forward(seq[:, :VERIFY_PREFIX], runner.context(cache, torch.arange(VERIFY_PREFIX)[None], trusted=True))
    positions = torch.arange(VERIFY_PREFIX, VERIFY_PREFIX + verified)[None]
    with phase("decode"):
        runner.forward(seq[:, VERIFY_PREFIX : VERIFY_PREFIX + verified], runner.context(cache, positions, trusted=True))
    with phase("decode"), cache.writing_at(torch.tensor([VERIFY_PREFIX + accepted])):
        scores = {i: s[..., : VERIFY_PREFIX + accepted].clone() for i, s in cache.scores.tensors.items()}
        keep = {i: k[..., : VERIFY_PREFIX + accepted].clone() for i, k in cache.keep.tensors.items()}
    model.detach()
    return scores, keep


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_h2o_ignores_the_attention_of_rejected_draft_queries(attn: str) -> None:
    """A verifying pass over 3 tokens of which 1 is accepted leaves the scores (and the evictions they cause) of a
    pass over the accepted token alone."""
    scores, keep = _h2o_scores_after_a_verify_pass(attn, verified=3, accepted=1)
    expected_scores, expected_keep = _h2o_scores_after_a_verify_pass(attn, verified=1, accepted=1)
    for layer, expected in expected_scores.items():
        torch.testing.assert_close(scores[layer], expected, atol=1e-5, rtol=1e-5)  # eager: matmul rounding
        assert torch.equal(keep[layer], expected_keep[layer])


def test_attention_tap_is_active_exactly_during_exact_passes() -> None:
    from torch.overrides import _get_current_function_mode_stack

    from economize.techniques.kv_evict.tap import AttentionTap

    def tapped() -> bool:
        return any(isinstance(mode, AttentionTap) for mode in _get_current_function_mode_stack())

    model = economize.apply(make_tiny_model(), KVEvictConfig(method="h2o", recent=2, heavy=4))
    cache = model.techniques["kv_evict"].wrap_cache(model.runner.new_cache(64))
    with phase("prefill"):
        model.runner.forward(torch.ones(1, 4, dtype=torch.long),
                             model.runner.context(cache, torch.arange(4)[None], trusted=True))
    assert not tapped()
    with pytest.raises(RuntimeError, match="boom"), phase("decode"), cache.writing_at(torch.tensor([4])):
        assert tapped()
        raise RuntimeError("boom")
    assert not tapped()
    with phase("draft"), cache.writing_at(torch.tensor([4])):
        assert not tapped()
    model.detach()


def test_pending_scores_follow_tree_compaction_and_row_reorder() -> None:
    """Compaction moves a pending query (and the key columns) from ``src`` to ``dst``; a query overwritten in place
    can no longer be accepted; a reorder keeps the given rows."""
    trash = 99
    received = torch.arange(2 * 1 * 3 * 8, dtype=torch.float).view(2, 1, 3, 8)
    entry = PendingScores(0, torch.tensor([[4, 5, 6], [4, 5, 6]]), received.clone())
    # row 0: the tree branch at slot 6 is accepted and moves to slot 5 (overwriting the rejected query there)
    entry.gather(torch.tensor([0]), torch.tensor([[6]]), torch.tensor([[5]]), trash)
    assert entry.query_slots.tolist() == [[4, trash, 5], [4, 5, 6]]
    assert torch.equal(entry.received[0, :, :, 5], received[0, :, :, 6])
    assert torch.equal(entry.received[1], received[1])
    accepted = entry.accepted(torch.tensor([6, 5]))
    assert accepted.tolist() == [[1.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    entry.reorder(torch.tensor([1]))
    assert entry.query_slots.tolist() == [[4, 5, 6]] and torch.equal(entry.received, received[1:])


# ---- attention tap -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mask", ["causal", "bool", "additive", "broadcast", "none"])
@pytest.mark.parametrize("query_start", [0, 5, 13])
def test_attention_probs_in_chunks_equal_the_whole_matrix(mask: str, query_start: int) -> None:
    from economize.techniques.kv_evict import attention_probs, iter_attention_probs

    generator = torch.Generator().manual_seed(0)
    query, key = torch.randn(2, 4, 13, 8, generator=generator), torch.randn(2, 2, 17, 8, generator=generator)
    allowed = torch.rand(2, 1, 13, 17, generator=generator) > 0.3
    allowed[:, :, 3] = False  # a fully masked query row
    kwargs: dict = {"is_causal": mask == "causal"}
    if mask in ("bool", "additive", "broadcast"):
        attn_mask = allowed[:, :, :1] if mask == "broadcast" else allowed
        kwargs["attn_mask"] = attn_mask if mask == "bool" else torch.zeros(attn_mask.shape).masked_fill(
            ~attn_mask, float("-inf"))
    whole = attention_probs(query, key, **kwargs)
    for chunk_bytes in (1, 3 * 2 * 4 * 17 * 4, None):
        chunks = list(iter_attention_probs(query, key, **kwargs, query_start=query_start, chunk_bytes=chunk_bytes))
        starts = [query_start, *(start + probs.shape[-2] for start, probs in chunks)][: len(chunks)]
        assert [start for start, _ in chunks] == starts  # consecutive row ranges
        rows = 13 - query_start  # a chunk of 3 * 2 * 4 * 17 * 4 bytes holds 3 rows
        assert len(chunks) == {1: rows, None: min(rows, 1)}.get(chunk_bytes, -(-rows // 3))
        got = torch.cat([probs for _, probs in chunks], dim=-2) if chunks else whole[..., :0, :]
        torch.testing.assert_close(got, whole[..., query_start:, :], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "config",
    [KVEvictConfig(method="snapkv", heavy=4, observation_window=4), KVEvictConfig(method="h2o_prefill", recent=4,
                                                                                  heavy=4),
     KVEvictConfig(method="h2o", recent=4, heavy=4)],
    ids=lambda c: c.method,
)
def test_tapped_scores_in_one_row_chunks_equal_eager_attention_scores(config: KVEvictConfig, monkeypatch) -> None:
    """The tap recomputes ``sdpa`` probabilities in bounded chunks of query rows, from the first row with a weight
    on; the scores equal those from the probabilities eager attention returns."""
    from economize.techniques.kv_evict import tap

    seq = _sequence()
    monkeypatch.setattr(tap, "CHUNK_BYTES", 1)
    observed: list[tuple[int, int]] = []
    observe = economize.techniques.kv_evict.EvictionCache.observe

    def spy(self, probs, query_start=0):
        observed.append((query_start, probs.shape[-2]))
        return observe(self, probs, query_start)

    monkeypatch.setattr(economize.techniques.kv_evict.EvictionCache, "observe", spy)
    _, tapped = _teacher_forced(economize.apply(make_tiny_model(attn="sdpa"), config), seq)
    assert observed and all(rows == 1 for _, rows in observed)
    if config.method == "snapkv":  # only the observation window of the prompt is recomputed
        assert {start for start, _ in observed} == set(range(PREFIX - 4, PREFIX))
    observed.clear()
    _, eager = _teacher_forced(economize.apply(make_tiny_model(attn="eager"), config), seq)
    for layer in eager.scores.tensors:
        torch.testing.assert_close(tapped.scores[layer], eager.scores[layer], atol=1e-5, rtol=1e-5)
        assert torch.equal(tapped.keep[layer], eager.keep[layer])


@pytest.mark.parametrize(
    "config",
    [KVEvictConfig(method="snapkv", heavy=4, observation_window=4), KVEvictConfig(method="h2o", recent=4, heavy=4)],
    ids=lambda c: c.method,
)
def test_a_flash_prefill_under_the_tap_scores_like_sdpa(config: KVEvictConfig, monkeypatch) -> None:
    """Under ``"economize"`` the prompt pass runs on FlashAttention although the tap is active (faked here on CPU
    with ``sdpa`` outside the tap); the tap is told about each call and scores and keeps exactly what it does over
    ``sdpa``."""
    from economize.attention import interface

    def fake_flash(query, key, value, mask, scale, softcap=None, window=None):
        assert mask is not None and softcap is None and window is None
        groups = query.shape[1] // key.shape[1]
        with torch._C.DisableTorchFunction():  # the tap must learn of the call through observe_attention only
            out = F.scaled_dot_product_attention(query, key.repeat_interleave(groups, 1),
                                                 value.repeat_interleave(groups, 1),
                                                 attn_mask=mask[..., : key.shape[2]], scale=scale)
        return out.transpose(1, 2), "flash_fake"

    interface.install()
    reason = interface._fallback_reason
    monkeypatch.setattr(interface, "flash_prefill", fake_flash)
    monkeypatch.setattr(
        interface,
        "_fallback_reason",
        lambda module, query, *a: "long_query" if query.shape[2] == PREFIX else reason(module, query, *a),
    )
    seq = _sequence()
    interface.kernel_usage(reset=True)
    _, flashed = _teacher_forced(economize.apply(make_tiny_model(attn="economize"), config), seq)
    paths = {row["path"] for row in interface.kernel_usage(reset=True) if row["queries"] == PREFIX}
    assert paths == {"flash_fake"}
    _, tapped = _teacher_forced(economize.apply(make_tiny_model(attn="sdpa"), config), seq)
    for layer in tapped.scores.tensors:
        torch.testing.assert_close(flashed.scores[layer], tapped.scores[layer], atol=1e-6, rtol=1e-6)
        assert torch.equal(flashed.keep[layer], tapped.keep[layer])


def test_flash_prefill_runs_only_under_attention_observers(monkeypatch) -> None:
    """Any other function mode may intercept ``sdpa``, so a long query under it stays on ``sdpa``."""
    from torch.overrides import TorchFunctionMode

    from economize.attention import AttentionObserver, interface
    from economize.techniques.kv_evict.tap import AttentionTap

    monkeypatch.setattr(interface, "flash_prefill", lambda query, *a, **k: (query.transpose(1, 2), "flash_fake"))
    query = key = value = torch.zeros(1, 2, 8, 4)
    mask = torch.ones(1, 1, 8, 8, dtype=torch.bool).tril()

    class Sink:
        def wants_probs(self) -> bool:
            return False

    assert isinstance(AttentionTap(Sink()), AttentionObserver)
    with TorchFunctionMode():
        assert interface._flash(torch.nn.Module(), query, key, value, mask, 0.0, None, {}) is None
    with AttentionTap(Sink()):
        assert interface._flash(torch.nn.Module(), query, key, value, mask, 0.0, None, {}) is not None
