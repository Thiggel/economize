"""Self-speculative early exit: lossless decoding for every head and adaptation, costs, save/load and training."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import economize
from economize import ConfigError, DecodeConfig, TopPSampler
from economize.errors import IncompatibleTechniquesError
from economize.techniques.early_exit import EarlyExitConfig, EarlyExitKL, EarlyExitTechnique
from economize.train import DistillKL, ObjectiveModule, default_objective, fit
from helpers import VOCAB, make_tiny_model, perturb, reference_greedy

LAYERS = 6
HEADS = ("shared", "readout")
ADAPTATIONS = ("frozen", "lora", "full")


def _model(head: str = "readout", adaptation: str = "frozen", exit_layer: int = 3, std: float = 0.05,
           base: torch.nn.Module | None = None, **decoding: Any) -> Any:
    config = EarlyExitConfig(exit_layer=exit_layer, head=head, adaptation=adaptation, lora_rank=2)
    model = economize.apply(base if base is not None else make_tiny_model(), config,
                         decoding=DecodeConfig(**decoding) if decoding else None)
    if std:
        perturb(model.techniques["early_exit"], std, seed=1)
    return model


def _prompts(seed: int = 1, lengths: tuple[int, ...] = (12, 7, 20)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in lengths]


def _loss(model: Any, objective: Any, ids: torch.Tensor) -> dict[str, Any]:
    module = ObjectiveModule(model, objective)
    try:
        return module(ids)
    finally:
        module.close()


# ---- configuration -------------------------------------------------------------------------------------------------
def test_config_validation_and_round_trip() -> None:
    config = EarlyExitConfig(exit_layer=2, head="shared", adaptation="lora", lora_rank=4)
    assert EarlyExitConfig.from_dict(config.to_dict()) == config
    assert config.resolved_lora_alpha == 8.0
    for bad in (dict(exit_layer=0), dict(exit_layer=2, head="medusa"), dict(exit_layer=2, adaptation="qlora"),
                dict(exit_layer=2, lora_rank=0)):
        with pytest.raises(ConfigError):
            EarlyExitConfig(**bad)
    with pytest.raises(ConfigError):
        economize.apply(make_tiny_model(), EarlyExitConfig(exit_layer=LAYERS))


def test_parameters_follow_head_and_adaptation() -> None:
    names = {}
    for head in HEADS:
        for adaptation in ADAPTATIONS:
            technique = _model(head, adaptation, std=0.0).techniques["early_exit"]
            assert isinstance(technique, EarlyExitTechnique) and technique.layer_span == (1, 3)
            names[head, adaptation] = {n.split(".")[0] + "." + n.split(".")[1] for n, _ in
                                       technique.named_parameters()}
            assert all(p.dtype == torch.float32 for p in technique.parameters())
    assert names["shared", "frozen"] == set()
    assert names["readout", "frozen"] == {"readout.linear", "readout.mlp"}
    assert names["shared", "lora"] == {"path.lora"} and names["shared", "full"] == {"path.blocks"}


# ---- lossless decoding ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("adaptation", ADAPTATIONS)
@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("cycle_length", [1, 3, 6])
def test_greedy_early_exit_equals_plain_greedy(head: str, adaptation: str, cycle_length: int) -> None:
    base = make_tiny_model()
    model = _model(head, adaptation, base=base, cycle_length=cycle_length)
    prompts = _prompts()
    new = [30, 21, 26]
    results = model.decode_batch(prompts, new)
    for prompt, result, n in zip(prompts, results, new):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, n))
    if cycle_length > 1:
        assert sum(len(r.acceptance) for r in results) > 0


def test_untrained_shared_exit_after_the_last_layers_is_accepted_often() -> None:
    """Exiting after layer ``L - 1`` with the model's own head agrees with the model far more often than chance."""
    base = make_tiny_model()
    model = _model("shared", "frozen", exit_layer=LAYERS - 1, std=0.0, base=base, cycle_length=4)
    result = model.decode(_prompts()[2], 40)
    assert torch.equal(result.tokens, reference_greedy(base, _prompts()[2], 40))
    accepted = sum(a for _, a in result.acceptance)
    drafted = sum(k for k, _ in result.acceptance)
    assert accepted / drafted > 0.2


def test_rejection_at_near_zero_temperature_is_greedy() -> None:
    base = make_tiny_model()
    model = _model("readout", "lora", base=base)
    prompts = _prompts()
    config = DecodeConfig(cycle_length=4, verifier="rejection")
    results = model.decode_batch(prompts, 24, decoding=config, selector=TopPSampler(1e-6, seed=0))
    for prompt, result in zip(prompts, results):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 24))


def test_cost_log_has_one_draft_event_per_draft_step() -> None:
    model = _model("readout", "full", exit_layer=2, cycle_length=5)
    result = model.decode(_prompts()[0], 30)
    kinds = {event.kind for event in result.cost.events}
    assert kinds <= {"exact", "draft", "draft_wasted"} and "draft" in kinds
    for event in result.cost.events:
        if event.kind.startswith("draft"):
            assert (event.tokens, event.first_layer, event.last_layer) == (1, 1, 2)
        else:
            assert (event.first_layer, event.last_layer) == (1, LAYERS)
    drafts = sum(1 for e in result.cost.events if e.kind == "draft")
    assert drafts == sum(k for k, _ in result.acceptance)


def test_save_and_load_round_trip(tmp_path) -> None:
    for adaptation in ADAPTATIONS:
        model = _model("readout", adaptation, cycle_length=4)
        economize.save(model, tmp_path / adaptation)
        loaded = economize.load(make_tiny_model(), tmp_path / adaptation)
        assert loaded.drafting_technique.config == model.drafting_technique.config
        for (name, a), (_, b) in zip(model.techniques.state_dict().items(), loaded.techniques.state_dict().items()):
            assert torch.equal(a, b), name
        prompt = _prompts()[1]
        first, second = model.decode(prompt, 20), loaded.decode(prompt, 20)
        assert torch.equal(first.tokens, second.tokens) and first.acceptance == second.acceptance


def test_early_exit_on_a_pruned_model_equals_its_greedy(calib) -> None:
    from economize.techniques.prune import PruneConfig

    base = make_tiny_model()
    model = economize.apply(base, PruneConfig(method="wanda", sparsity="2:4"),
                         EarlyExitConfig(exit_layer=3, adaptation="lora", lora_rank=2),
                         decoding=DecodeConfig(cycle_length=4), calibration=calib)
    assert list(model.techniques) == ["prune", "early_exit"]
    perturb(model.techniques["early_exit"], 0.05, seed=2)
    prompts = _prompts()
    for prompt, result in zip(prompts, model.decode_batch(prompts, 24)):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 24))


def test_draft_quant_quantizes_the_layer_copies_only() -> None:
    from economize.techniques.quant import DraftQuantConfig

    base = make_tiny_model()
    reference = {k: v.clone() for k, v in base.state_dict().items()}
    model = economize.apply(base, EarlyExitConfig(exit_layer=3, adaptation="full"), DraftQuantConfig(group_size=None),
                         decoding=DecodeConfig(cycle_length=4))
    assert all(torch.equal(v, reference[k]) for k, v in base.state_dict().items())
    prompt = _prompts()[0]
    assert torch.equal(model.decode(prompt, 20).tokens, reference_greedy(base, prompt, 20))


# ---- training ------------------------------------------------------------------------------------------------------
def test_default_objective_and_parameter_groups() -> None:
    model = _model("readout", "lora", std=0.0)
    objective = default_objective(model)
    assert isinstance(objective, EarlyExitKL)
    groups = {g.name: g for g in objective.parameter_groups(model)}
    assert set(groups) == {"drafter", "lora"} and groups["lora"].kind == "backbone"
    full = _model("shared", "full", std=0.0)
    assert [g.name for g in EarlyExitKL().parameter_groups(full)] == ["blocks"]
    with pytest.raises(IncompatibleTechniquesError):
        EarlyExitKL().parameter_groups(_model("shared", "frozen", std=0.0))
    with pytest.raises(ValueError):
        EarlyExitKL(horizon=0)


@pytest.mark.parametrize("adaptation", ["lora", "full"])
def test_untrained_adapted_path_rollouts_equal_the_frozen_loss(adaptation: str) -> None:
    """At initialization the adapted path computes the model's own layers, so rollouts with the true tokens give the
    frozen path's loss at every horizon."""
    ids = torch.randint(0, VOCAB, (2, 18), generator=torch.Generator().manual_seed(4))
    frozen = _model("readout", "frozen", std=0.0)
    perturb(frozen.techniques["early_exit"].readout, 0.05, seed=3)
    with torch.enable_grad():
        expected = _loss(frozen, EarlyExitKL(), ids)["loss"].detach()
        adapted = _model("readout", adaptation, std=0.0)
        adapted.techniques["early_exit"].readout.load_state_dict(frozen.techniques["early_exit"].readout.state_dict())
        one = _loss(adapted, EarlyExitKL(horizon=1), ids)
        torch.testing.assert_close(one["loss"].detach(), expected, rtol=1e-4, atol=1e-6)
        three = _loss(adapted, EarlyExitKL(horizon=3), ids)
        assert three["metrics"]["klK"].item() > 0
        distill = _loss(adapted, DistillKL(), ids)
        torch.testing.assert_close(distill["loss"].detach(), expected, rtol=1e-4, atol=1e-6)
        three["loss"].backward()
    trained = [p for p in adapted.techniques["early_exit"].parameters() if p.grad is not None]
    assert trained and any(p.grad.abs().sum() > 0 for p in trained)


def test_zero_init_readout_loss_is_the_kl_of_the_intermediate_head() -> None:
    """A zero-initialized readout adapter is the identity, so the loss is KL(final || norm + head at layer 3)."""
    ids = torch.randint(0, VOCAB, (2, 16), generator=torch.Generator().manual_seed(5))
    model = _model("readout", "frozen", std=0.0)
    base = model.base_model
    with torch.no_grad():
        out = base(ids, output_hidden_states=True)
        final = out.logits[:, :-1].float().log_softmax(-1)
        early = base.lm_head(base.model.norm(out.hidden_states[3]))[:, :-1].float().log_softmax(-1)
    kl = (final.exp() * (final - early)).sum(-1).mean()
    with torch.enable_grad():
        loss = _loss(model, EarlyExitKL(), ids)["loss"]
    torch.testing.assert_close(loss.detach(), kl, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("adaptation", ["frozen", "lora"])
def test_fit_trains_the_exit(adaptation: str) -> None:
    rows = torch.randint(0, VOCAB, (16, 24), generator=torch.Generator().manual_seed(3))
    model = _model("readout", adaptation, std=0.0)
    with torch.enable_grad():
        result = fit(model, rows, steps=30, batch_size=8, lr=3e-3, warmup_steps=3, seed=0, log_every=10)
    assert result.steps == 30
    assert sum(result.losses[-5:]) < 0.95 * sum(result.losses[:5])
    assert not any(p.requires_grad for p in model.base_model.parameters())


@pytest.mark.parametrize("adaptation", ADAPTATIONS)
def test_distill_kl_trains_the_exit_through_the_trainable_drafter_protocol(adaptation: str) -> None:
    rows = torch.randint(0, VOCAB, (8, 20), generator=torch.Generator().manual_seed(6))
    model = _model("readout", adaptation, std=0.0)
    technique = model.techniques["early_exit"]
    before = [p.detach().clone() for p in technique.parameters()]
    with torch.enable_grad():
        result = fit(model, rows, "distill_kl", steps=3, batch_size=8, lr=1e-2, warmup_steps=0, seed=0, log_every=1)
        assert result.steps == 3 and result.losses[-1] < result.losses[0]
        # zero-initialized outputs (readout, LoRA B) pass gradients on once the first step has moved them
        assert all(not torch.equal(a, b) for a, b in zip(before, technique.parameters()))
        technique.zero_grad(set_to_none=True)
        _loss(model, DistillKL(), rows[:2])["loss"].backward()
    for name, parameter in technique.named_parameters():
        assert parameter.grad is not None and bool(parameter.grad.abs().sum() > 0), name
    assert not any(p.requires_grad for p in model.base_model.parameters())


class _RecordingReadout(nn.Module):
    """Stands in for a readout placed on another device: it records where its input lives and returns zeros on CPU."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, device="meta"))
        self.seen: list[torch.device] = []

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.seen.append(hidden.device)
        return torch.zeros(hidden.shape, dtype=torch.float32)


def test_early_exit_moves_the_hidden_state_to_the_readout_device() -> None:
    """With a model split over devices, layer ``r`` may live elsewhere than the readout (placed with the head)."""
    model = economize.apply(make_tiny_model(), EarlyExitConfig(exit_layer=3, head="readout"))
    technique = model.techniques["early_exit"]
    hidden = torch.randn(1, 2, model.adapter.hidden_size, dtype=model.adapter.dtype)
    readout = _RecordingReadout(model.adapter.hidden_size)
    technique.readout = readout
    logits = technique.logits(hidden)
    assert readout.seen == [torch.device("meta")]
    assert logits.shape[:2] == (1, 2)


# ---- confidence log ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("min_confidence", [None, 0.1])
def test_record_confidence_logs_top_prob_per_draft_without_changing_tokens(min_confidence: float | None) -> None:
    base = make_tiny_model()
    model = _model(head="shared", base=base, std=0.0)
    prompts = _prompts()
    plain = DecodeConfig(cycle_length=4, min_confidence=min_confidence)
    logged = DecodeConfig(cycle_length=4, min_confidence=min_confidence, record_confidence=True)
    off = model.decode_batch(prompts, 24, decoding=plain)
    on = model.decode_batch(prompts, 24, decoding=logged)
    for a, b in zip(off, on):
        assert torch.equal(a.tokens, b.tokens) and a.acceptance == b.acceptance
        assert a.confidence == []
        assert len(b.confidence) == len(b.acceptance) > 0
        for (drafted, _), conf in zip(b.acceptance, b.confidence):
            assert len(conf) == drafted and all(0.0 < p <= 1.0 for p in conf)
            if min_confidence is not None:  # drafts after the first stop below the threshold
                assert all(p >= min_confidence for p in conf[1:])


# ---- distribution match log ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("sampled", [False, True])
def test_record_distributions_logs_match_per_draft_without_changing_tokens(sampled: bool) -> None:
    model = _model(head="shared", base=make_tiny_model())
    prompts = _prompts()
    sampling = {"verifier": "rejection"} if sampled else {}

    def run(**extra: Any) -> Any:
        selector = TopPSampler(0.8, seed=0) if sampled else None
        return model.decode_batch(prompts, 24, decoding=DecodeConfig(cycle_length=4, **sampling, **extra),
                                  selector=selector)

    off, on = run(), run(record_distributions=True)
    for a, b in zip(off, on):
        assert torch.equal(a.tokens, b.tokens) and a.acceptance == b.acceptance
        assert a.match == []
        assert len(b.match) == len(b.acceptance) > 0
        for (drafted, accepted), cycle in zip(b.acceptance, b.match):
            assert len(cycle) == drafted
            for top1, kl, tv in cycle:
                assert top1 in (0.0, 1.0) and kl >= -1e-5 and -1e-6 <= tv <= 1.0 + 1e-6
            if not sampled:  # greedy: a draft is accepted exactly when it matches the verifier's top-1
                assert [t for t, _, _ in cycle[:accepted]] == [1.0] * accepted
                if accepted < drafted:
                    assert cycle[accepted][0] == 0.0


def test_distribution_match_of_identical_distributions_is_exact() -> None:
    from economize.decoding.commit import distribution_match

    logits = torch.randn(2, 3, VOCAB, generator=torch.Generator().manual_seed(0))
    same = distribution_match(logits, logits.clone())
    assert torch.equal(same[..., 0], torch.ones(2, 3))
    assert same[..., 1].abs().max() < 1e-5 and same[..., 2].abs().max() < 1e-6
    other = distribution_match(logits, torch.zeros_like(logits))
    assert (other[..., 1] > 0).all() and (other[..., 2] > 0).all()
