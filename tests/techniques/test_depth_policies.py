"""Dynamic depth on the exact path: GateSkip, Mixture-of-Depths, FlexiDepth, SkipDecode, CALM and LoRA-Drop.

Every technique has a trivial setting that reproduces the model exactly, skips layer applications otherwise, counts
them in the cost accounting, and composes with drafters (exact passes follow the policy) and weight compression.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn.functional as F

import economize
from economize import DecodeConfig
from economize.errors import ConfigError
from economize.runtime import LayerPolicy, phase
from economize.techniques.dynamic_depth import (
    CalmConfig,
    DepthMeter,
    FlexiDepthConfig,
    GateSkipConfig,
    LoRADropConfig,
    MoDConfig,
    SkipDecodeConfig,
    redundant_layers,
    top_fraction,
)
from helpers import VOCAB, make_tiny_model, perturb, reference_greedy

PREFIX, TOTAL, LAYERS = 12, 40, 6

TRIVIAL = [
    GateSkipConfig(gate_bias=100.0),
    MoDConfig(),
    FlexiDepthConfig(router_bias=100.0),
    SkipDecodeConfig(),
    CalmConfig(threshold=1.0),
    LoRADropConfig(period=1),
]

#: Settings that skip layers on the random tiny model, and whether the technique's parameters must be perturbed
#: (zero-initialized parts) for the setting to have an effect.
LOSSY = [
    (GateSkipConfig(gate_bias=0.0, gate_init_std=0.5), False),
    (GateSkipConfig(rule="budget", budget=0.5), False),
    (MoDConfig(router_bias=0.0, router_init_std=0.3), False),
    (FlexiDepthConfig(router_bias=0.0), True),
    (SkipDecodeConfig(max_layers=6, min_layers=3, warmup=12, decay=10), False),
    (SkipDecodeConfig(min_layers=2, decay=20, skip="top"), False),
    (CalmConfig(confidence="state", threshold=0.5), False),
    (CalmConfig(confidence="classifier", threshold=0.4, decay_temperature=4.0), True),
    (LoRADropConfig(period=2), True),
]


LOSSY_IDS = [
    "gateskip-threshold", "gateskip-budget", "mod", "flexidepth", "skipdecode-bottom", "skipdecode-top",
    "calm-state", "calm-classifier", "loradrop",
]


def _ids(config: object) -> str:
    return config.technique


def _sequence(seed: int = 0) -> torch.Tensor:
    return torch.randint(0, VOCAB, (1, TOTAL), generator=torch.Generator().manual_seed(seed))


def _prompts() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in (12, 7, 20)]


def _teacher_forced(model: economize.EfficientModel, seq: torch.Tensor, policy: object = None) -> torch.Tensor:
    """Logits of ``seq`` from a prefill of :data:`PREFIX` tokens and single-token decode steps under ``policy``."""
    runner = model.runner
    cache = runner.new_cache(64)
    ctx = runner.context(cache, torch.arange(PREFIX)[None], trusted=True)
    with phase("prefill"):
        out = [runner.logits(runner.forward(seq[:, :PREFIX], ctx, policy=policy).final)[0]]
    for t in range(PREFIX, seq.shape[1]):
        ctx = runner.context(cache, torch.tensor([[t]]), trusted=True)
        with phase("decode"):
            out.append(runner.logits(runner.forward(seq[:, t : t + 1], ctx, policy=policy).final)[0])
    return torch.cat(out).float()


def _kl(logits: torch.Tensor, reference: torch.Tensor) -> float:
    return F.kl_div(logits.log_softmax(-1), reference.log_softmax(-1), log_target=True, reduction="batchmean").item()


def _identity_jump(base: torch.nn.Module, *configs: object, cycle_length: int = 4) -> economize.EfficientModel:
    jump = economize.JumpConfig(target_layer=0, source_layer=0, adaptation="frozen", lora_rank=2)
    model = economize.apply(base, jump, *configs, decoding=DecodeConfig(cycle_length=cycle_length))
    for name, parameter in model.techniques["jump"].predictor.named_parameters():
        if name != "out_gain":
            parameter.zero_()
    return model


# ---- configuration -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: GateSkipConfig(rule="topk"),
        lambda: GateSkipConfig(threshold=1.5),
        lambda: GateSkipConfig(budget=0.0),
        lambda: GateSkipConfig(layers=(0,)),
        lambda: MoDConfig(capacity=0.0),
        lambda: MoDConfig(router_init_std=-1.0),
        lambda: FlexiDepthConfig(threshold=1.0),
        lambda: FlexiDepthConfig(router_reduction=0),
        lambda: SkipDecodeConfig(skip="middle"),
        lambda: SkipDecodeConfig(decay=0),
        lambda: SkipDecodeConfig(min_layers=-1),
        lambda: CalmConfig(confidence="entropy"),
        lambda: CalmConfig(threshold=1.5),
        lambda: CalmConfig(min_layer=0),
        lambda: LoRADropConfig(period=0),
        lambda: LoRADropConfig(rank=0),
    ],
)
def test_config_validation(build) -> None:
    with pytest.raises(ConfigError):
        build()


@pytest.mark.parametrize(
    "config",
    [GateSkipConfig(layers=(7,)), SkipDecodeConfig(max_layers=2, min_layers=3), SkipDecodeConfig(max_layers=7),
     CalmConfig(min_layer=6), LoRADropConfig(layers=(9,)), FlexiDepthConfig(layers=(7,)), MoDConfig(layers=(7,))],
    ids=_ids,
)
def test_config_validated_against_the_model(config: object) -> None:
    with pytest.raises(ConfigError):
        economize.apply(make_tiny_model(), config)


def test_skipdecode_schedule() -> None:
    config = SkipDecodeConfig(max_layers=6, min_layers=2, warmup=10, decay=8)
    active = config.active_layers(torch.arange(0, 24), LAYERS)
    assert active[:11].eq(6).all() and active[18:].eq(2).all()
    assert (active[1:] <= active[:-1]).all()  # never grows, so skipped layers need no K/V
    assert config.budget(LAYERS) == (6, 2) and SkipDecodeConfig().budget(LAYERS) == (6, 6)


def test_calm_threshold_decay() -> None:
    generated = torch.tensor([0, 64, 256, 1024])
    assert torch.equal(CalmConfig(threshold=0.8).threshold_at(generated), torch.full((4,), 0.8))
    decayed = CalmConfig(threshold=0.8, decay_temperature=4.0, decay_horizon=256).threshold_at(generated)
    assert float(decayed[0]) == pytest.approx(0.82, abs=1e-6) and (decayed[1:] < decayed[:-1]).all()
    assert float(decayed[-1]) == pytest.approx(0.72, abs=1e-3) and float(decayed[-1]) >= 0.72


def test_top_fraction() -> None:
    scores = torch.tensor([[0.1, 0.9, 0.5, 0.3]])
    valid = torch.tensor([[True, True, True, False]])
    chosen = top_fraction(scores, valid, 0.5)
    assert chosen.dtype == torch.bool and chosen[0, 1] and not chosen[0, 3]
    assert int(chosen.sum()) == 2


def test_depth_meter_summary_without_passes() -> None:
    summary = DepthMeter(LAYERS).summary()
    assert summary["executed_fraction"] == 1.0 and summary["skipped_layer_tokens"] == 0


# ---- the trivial setting is the model ------------------------------------------------------------------------------


@pytest.mark.parametrize("config", TRIVIAL, ids=_ids)
def test_trivial_setting_is_exact(config: object) -> None:
    seq = _sequence()
    base = make_tiny_model()
    model = economize.apply(base, config)
    technique = model.techniques[config.technique]
    policy = technique.layer_policy()
    assert torch.equal(_teacher_forced(model, seq, policy), _teacher_forced(model, seq))
    summary = policy.meter.summary()
    assert summary["executed_fraction"] == 1.0 and summary["skipped_layer_tokens"] == 0
    prompt = seq[0, :PREFIX]
    result = model.decode(prompt, 16)
    assert torch.equal(result.tokens, reference_greedy(base, prompt, 16))
    assert technique.cost_summary(result)["relative_effective_layer_tokens"] == pytest.approx(1.0)


# ---- the mechanisms ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("config, needs_noise", LOSSY, ids=LOSSY_IDS)
def test_mechanism_skips_layers(config: object, needs_noise: bool) -> None:
    seq = _sequence()
    base = make_tiny_model()
    reference = base(seq).logits[0].float()
    model = economize.apply(base, config)
    technique = model.techniques[config.technique]
    if needs_noise:
        perturb(technique, 0.3)
    policy = technique.layer_policy()
    logits = _teacher_forced(model, seq, policy)
    summary = policy.meter.summary()
    assert _kl(logits, reference) > 1e-3
    assert summary["executed_fraction"] < 0.95 and summary["skipped_layer_tokens"] > 0
    assert summary["executed_layer_tokens"] + summary["skipped_layer_tokens"] == summary["policy_layer_tokens"]
    assert len(summary["executed_fraction_per_layer"]) == LAYERS
    if config.technique == "skipdecode":
        assert summary["kv_fill_layer_tokens"] == 0
    elif config.technique in ("gateskip", "mod", "flexidepth", "calm"):
        assert summary["kv_fill_layer_tokens"] == summary["skipped_layer_tokens"]
    if config.technique == "calm":
        assert summary["exits"] > 0
        assert summary.get("confidence_head_tokens", 0) == 0  # the state and classifier measures need no head


def test_gateskip_budget_ranks_the_tokens_of_a_pass() -> None:
    """The budget rule keeps the top fraction of a pass's tokens, so a single-token decode step always runs every
    layer; the prompt and multi-token verification passes skip (docs/depth_kv.md, use ``threshold`` for
    single-token decoding)."""
    model = economize.apply(make_tiny_model(), GateSkipConfig(rule="budget", budget=0.5))
    model.decode(_sequence()[0, :PREFIX], 12)
    phases = model.techniques["gateskip"].depth_stats()["phases"]
    assert phases["decode"]["ran"] == phases["decode"]["entered"] > 0
    assert phases["prefill"]["ran"] < phases["prefill"]["entered"]


def test_calm_softmax_counts_head_calls() -> None:
    model = economize.apply(make_tiny_model(), CalmConfig(threshold=0.3))
    result = model.decode(_sequence()[0, :PREFIX], 16)
    summary = model.techniques["calm"].cost_summary(result)
    assert summary["confidence_head_tokens"] > 0


def test_prompt_runs_every_layer_unless_asked() -> None:
    seq = _sequence()
    for config, prompt_skips in [
        (SkipDecodeConfig(min_layers=2, decay=4), False),
        (LoRADropConfig(period=2), False),
        (LoRADropConfig(period=2, skip_prompt=True), True),
        (MoDConfig(router_bias=0.0, router_init_std=0.3, skip_prompt=False), False),
        (MoDConfig(router_bias=0.0, router_init_std=0.3), True),
    ]:
        model = economize.apply(make_tiny_model(), config)
        policy = model.techniques[config.technique].layer_policy()
        runner = model.runner
        ctx = runner.context(runner.new_cache(64), torch.arange(PREFIX)[None], trusted=True)
        with phase("prefill"):
            runner.forward(seq[:, :PREFIX], ctx, policy=policy)
        assert (policy.meter.summary()["skipped_layer_tokens"] > 0) == prompt_skips, config


def test_skipdecode_budget_is_uniform_over_the_batch_and_counts_generated_tokens() -> None:
    """Rows with different prompt lengths decoded together share one budget per step, the one of their generated
    position, so a step is one dense pass (no K/V fill) and every row decodes as it would alone."""
    config = SkipDecodeConfig(max_layers=6, min_layers=2, warmup=2, decay=6)
    prompts, length = _prompts(), 14
    model = economize.apply(make_tiny_model(), config, decoding=DecodeConfig(cycle_length=1))
    alone = [model.decode(p, length).tokens for p in prompts]
    budgets: list[int] = []
    policy_class = type(model.techniques["skipdecode"].layer_policy())
    original = policy_class.begin_pass

    def spy(self, hidden, ctx):  # type: ignore[no-untyped-def]
        original(self, hidden, ctx)
        if hidden.shape[1] == 1:
            assert self.budget.shape == (1, 1)
            budgets.append(int(self.budget))

    policy_class.begin_pass = spy
    try:
        results = model.decode_batch(prompts, [length] * len(prompts))
    finally:
        policy_class.begin_pass = original
    for result, tokens in zip(results, alone):
        assert torch.equal(result.tokens, tokens)
    assert budgets == config.active_layers(torch.arange(len(budgets)), LAYERS).tolist()
    stats = model.techniques["skipdecode"].depth_stats()
    assert stats["skipped_layer_tokens"] > 0 and stats["kv_fill_layer_tokens"] == 0


def test_cost_summary_counts_skipped_layers() -> None:
    model = economize.apply(make_tiny_model(), SkipDecodeConfig(max_layers=6, min_layers=3, warmup=0, decay=6))
    result = model.decode(_sequence()[0, :PREFIX], 20)
    summary = model.techniques["skipdecode"].cost_summary(result)
    assert summary["effective_layer_tokens"] == summary["layer_tokens"] - summary["skipped_layer_tokens"]
    assert summary["relative_effective_layer_tokens"] < 0.8
    assert summary["skipped_layer_tokens"] > 0 and summary["kv_fill_layer_tokens"] == 0
    assert set(summary["phases"]) >= {"decode"}
    assert model.techniques["skipdecode"].depth_stats() == model.techniques["skipdecode"].last_policy.meter.summary()


def test_redundant_layers(calib) -> None:
    model = economize.apply(make_tiny_model(), LoRADropConfig())
    chosen = redundant_layers(model.adapter, list(calib[:2]), 3)
    assert len(chosen) == 3 and list(chosen) == sorted(chosen) and 1 not in chosen
    with pytest.raises(ConfigError):
        redundant_layers(model.adapter, list(calib[:2]), LAYERS)


def test_loradrop_default_layers_keep_the_first_three_and_the_last() -> None:
    model = economize.apply(make_tiny_model(), LoRADropConfig())
    technique = model.techniques["loradrop"]
    assert [i for i in range(1, LAYERS + 1) if technique.droppable(i)] == list(range(4, LAYERS))


# ---- composition ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config, needs_noise",
    [pytest.param(*case, id=name) for case, name in zip(LOSSY, LOSSY_IDS) if name != "gateskip-budget"],
)
def test_lossless_drafting_keeps_the_depth_output(config: object, needs_noise: bool) -> None:
    """With a jump drafter, exact passes follow the policy and verify the drafts; for a policy whose decisions do
    not depend on how tokens are grouped into passes, the output equals the technique's own greedy decoding, for
    batches of prompts of different lengths too."""
    if config.technique == "skipdecode":  # per-token budgets: independent of the batch (see the module docstring)
        config = dataclasses.replace(config, batch_uniform=False)
    base = make_tiny_model()
    prompts, lengths = _prompts(), [24, 17, 30]
    plain = economize.apply(base, config, decoding=DecodeConfig(cycle_length=1))
    if needs_noise:
        perturb(plain.techniques[config.technique], 0.3)
    expected = [plain.decode(p, n).tokens for p, n in zip(prompts, lengths)]
    state = {k: v.clone() for k, v in plain.techniques[config.technique].state_dict().items()}
    plain.detach()
    model = _identity_jump(base, config)
    model.techniques[config.technique].load_state_dict(state)
    results = model.decode_batch(prompts, lengths)
    for prompt, result, tokens in zip(prompts, results, expected):
        assert torch.equal(result.tokens, tokens)
        assert not torch.equal(result.tokens, reference_greedy(base, prompt, len(tokens))) or config.technique in (
            "calm", "flexidepth", "loradrop"
        )
    assert model.techniques[config.technique].depth_stats()["skipped_layer_tokens"] > 0
    single = model.decode(prompts[1], lengths[1])
    assert torch.equal(single.tokens, expected[1])


@pytest.mark.parametrize("config", TRIVIAL, ids=_ids)
def test_trivial_depth_with_jump_is_lossless(config: object) -> None:
    base = make_tiny_model()
    prompts, lengths = _prompts(), [24, 17, 30]
    model = _identity_jump(base, config)
    for prompt, length, result in zip(prompts, lengths, model.decode_batch(prompts, lengths)):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, length))
        assert sum(accepted for _, accepted in result.acceptance) > 0


@pytest.mark.parametrize("config", [TRIVIAL[0], TRIVIAL[3], TRIVIAL[4]], ids=_ids)
def test_depth_with_prune_and_lowrank(config: object, calib) -> None:
    from economize.techniques.lowrank import LowRankConfig
    from economize.techniques.prune import PruneConfig

    prompt = _prompts()[0]
    for weights in (PruneConfig(method="wanda", sparsity="2:4"), LowRankConfig(method="svd", compression=0.3)):
        expected = economize.apply(make_tiny_model(), weights, calibration=calib).decode(prompt, 20).tokens
        model = economize.apply(make_tiny_model(), weights, config, calibration=calib)
        assert torch.equal(model.decode(prompt, 20).tokens, expected)
    lossy = economize.apply(
        make_tiny_model(), PruneConfig(method="wanda", sparsity="2:4"),
        SkipDecodeConfig(min_layers=3, decay=8), calibration=calib,
    )
    result = lossy.decode(prompt, 20)
    assert lossy.techniques["skipdecode"].cost_summary(result)["relative_effective_layer_tokens"] < 0.85


def test_depth_with_kv_quantization() -> None:
    from economize.techniques.kv_quant import FP8KVConfig

    model = economize.apply(make_tiny_model(), SkipDecodeConfig(min_layers=3, decay=8), FP8KVConfig())
    result = model.decode(_prompts()[0], 20)
    assert model.techniques["skipdecode"].cost_summary(result)["skipped_layer_tokens"] > 0
    assert model.techniques["kv_fp8"].kv_stats()["kv_compression"] == pytest.approx(4.0)


# ---- per-row token budgets in batches ----
def test_gateskip_budget_is_per_row_in_a_batch() -> None:
    config = GateSkipConfig(rule="budget", budget=0.5)
    prompts, lengths = _prompts(), [10, 10, 10]
    single = economize.apply(make_tiny_model(), config, decoding=DecodeConfig(cycle_length=1))
    expected = [single.decode(p, n).tokens for p, n in zip(prompts, lengths)]
    state = {k: v.clone() for k, v in single.techniques["gateskip"].state_dict().items()}
    single.detach()
    model = economize.apply(make_tiny_model(), config, decoding=DecodeConfig(cycle_length=1))
    model.techniques["gateskip"].load_state_dict(state)  # the same gates
    results = model.decode_batch(prompts, lengths)
    decode = model.techniques["gateskip"].depth_stats()["phases"]["decode"]
    assert decode["ran"] == decode["entered"] > 0  # every row of a single-token step keeps its one token
    for result, tokens in zip(results, expected):
        assert torch.equal(result.tokens, tokens)


def test_top_fraction_per_row_ignores_padding() -> None:
    scores = torch.tensor([[5.0, 4.0, 3.0, 2.0], [9.0, 8.0, 0.0, 0.0]])
    valid = torch.tensor([[True, True, True, True], [False, False, True, True]])
    kept = top_fraction(scores, valid, 0.5)
    assert kept.tolist() == [[True, True, False, False], [False, False, True, False]]


def test_top_fraction_breaks_ties_towards_earlier_tokens_regardless_of_padding() -> None:
    """Untrained gates are all equal (``gate_init_std=0``); a padded row must choose like the unpadded prompt."""
    scores = torch.zeros(2, 6)
    valid = torch.tensor([[True] * 6, [False, False, True, True, True, True]])
    kept = top_fraction(scores, valid, 0.5)
    assert kept.tolist() == [[True, True, True, False, False, False], [False, False, True, True, False, False]]
    assert top_fraction(scores[:1], valid[:1], 0.5).tolist() == [[True, True, True, False, False, False]]


def test_redundant_layers_accepts_batches_of_several_rows(calib) -> None:
    model = economize.apply(make_tiny_model(), LoRADropConfig())
    rows = redundant_layers(model.adapter, [calib[:4]], 3)
    one_by_one = redundant_layers(model.adapter, list(calib[:4]), 3)
    assert rows == one_by_one


# ---- per-row policy state follows the rows of a batch ----
CONFIG = CalmConfig(confidence="classifier", threshold=0.4, decay_temperature=8.0, decay_horizon=16)


def _model(sync_interval: int) -> economize.EfficientModel:
    decoding = DecodeConfig(cycle_length=1, sync_interval=sync_interval)
    model = economize.apply(make_tiny_model(), CONFIG, decoding=decoding)
    perturb(model.techniques["calm"], 0.3)
    return model


def _spy(monkeypatch: pytest.MonkeyPatch) -> list[list[int]]:
    """Record the generated position of the first token of every pass (one entry per row) that CALM thresholds."""
    calls: list[list[int]] = []
    original = CalmConfig.threshold_at

    def threshold_at(self: CalmConfig, generated: torch.Tensor) -> torch.Tensor:
        calls.append(generated[:, 0].tolist())
        return original(self, generated)

    monkeypatch.setattr(CalmConfig, "threshold_at", threshold_at)
    return calls


@pytest.mark.parametrize("sync_interval", [1, 8])
def test_calm_keeps_each_rows_prompt_end_after_a_row_finishes(
    monkeypatch: pytest.MonkeyPatch, sync_interval: int
) -> None:
    prompts, lengths = _prompts(), [3, 16, 9]
    calls = _spy(monkeypatch)
    model = _model(sync_interval)
    alone = []
    for prompt, n in zip(prompts, lengths):
        calls.clear()
        alone.append((model.decode(prompt, n).tokens, [c[0] for c in calls]))
    calls.clear()
    results = model.decode_batch(prompts, lengths)
    for b, (result, (tokens, _)) in enumerate(zip(results, alone)):
        assert torch.equal(result.tokens, tokens), b
    # the last passes hold the longest row alone; it measures generated positions from its own prompt end
    tail = [c[0] for c in calls if len(c) == 1]
    assert tail and tail == alone[1][1][-len(tail):]


def test_layer_policy_reorder_is_a_no_op_by_default() -> None:
    class RunAll(LayerPolicy):
        def decide(self, index: int, hidden: torch.Tensor, ctx: object) -> None:  # type: ignore[override]
            return None

    RunAll().reorder(torch.tensor([0]))
