"""FREE, DMTD and Hidden Transfer: the dynamic-depth methods that decode through the drafting cycle.

* FREE keeps shallow exits above the softmax-gap threshold; its adaptive threshold fits a Beta mixture, and its
  distillation map is the least-cost monotone assignment of shallow to deep layers.
* DMTD drafts ``tau - 1`` tokens per cycle through the last ``L_d`` layers from the token embedding; its cost is the
  paper's PLT, and its rollout objective follows the cycle.
* Hidden Transfer maps the state after layer ``t_i`` to the state of the token ``i`` positions ahead; greedy
  verification makes it lossless, and its training logits equal its draft steps.
"""

from __future__ import annotations

import itertools

import pytest
import torch

import economize
from economize.decoding.cost import CostKind
from economize.errors import ConfigError
from economize.techniques.dynamic_depth import (
    DmtdConfig,
    DmtdRollout,
    FreeConfig,
    FreeObjective,
    HiddenTransferConfig,
    HiddenTransferDistill,
    beta_moments,
    bmm_threshold,
    free_calibration_pairs,
    kd_layer_map,
    monotone_layer_map,
    percentage_layers_per_token,
    softmax_gap,
)
from economize.techniques.early_exit import EarlyExitKL
from economize.train import Batch, ObjectiveModule, default_objective, run_teacher
from economize.train.objectives import DistillKL
from helpers import VOCAB, make_tiny_model, perturb, reference_greedy

LAYERS = 6


def _prompt(seed: int = 1, length: int = 12) -> torch.Tensor:
    return torch.randint(1, VOCAB, (length,), generator=torch.Generator().manual_seed(seed))


def _ids(seed: int = 2, rows: int = 2, length: int = 24) -> torch.Tensor:
    return torch.randint(0, VOCAB, (rows, length), generator=torch.Generator().manual_seed(seed))


def _backward(model: economize.EfficientModel, objective: object) -> float:
    module = ObjectiveModule(model, objective)
    try:
        with torch.enable_grad():
            loss = module(_ids())["loss"]
            loss.backward()
    finally:
        module.close()
    return float(loss)


def _grad(parameters) -> float:
    return sum(float(p.grad.abs().sum()) for p in parameters if p.grad is not None)


# ---- FREE: decoding ------------------------------------------------------------------------------------------------
def test_free_without_exits_is_greedy() -> None:
    base = make_tiny_model()
    model = economize.apply(base, FreeConfig(exit_layer=2))
    prompt = _prompt()
    result = model.decode(prompt, 20, decoding=model.techniques["free"].decode_config(threshold=1.0))
    assert torch.equal(result.tokens, reference_greedy(base, prompt, 20))


def test_free_exits_reuse_the_shallow_layers() -> None:
    model = economize.apply(make_tiny_model(), FreeConfig(exit_layer=2, max_exits=3))
    free = model.techniques["free"]
    result = model.decode(_prompt(), 24, decoding=free.decode_config(threshold=1e-6))
    summary = free.cost_summary(result)
    assert summary["reused_layer_tokens"] > 0
    assert summary["effective_layer_tokens"] == summary["layer_tokens"] - summary["reused_layer_tokens"]
    assert summary["relative_effective_layer_tokens"] < summary["relative_layer_tokens"]
    # every draft runs 2 of 6 layers and at most 3 run between deep passes
    assert summary["relative_sequential_layers"] < 1.0
    drafts = {(e.first_layer, e.last_layer) for e in result.cost.events if e.kind == CostKind.DRAFT}
    assert drafts == {(1, 2)}


def test_softmax_gap() -> None:
    logits = torch.log(torch.tensor([[0.6, 0.3, 0.1], [0.4, 0.4, 0.2]]))
    torch.testing.assert_close(softmax_gap(logits), torch.tensor([0.3, 0.0]))


# ---- FREE: adaptive threshold --------------------------------------------------------------------------------------
def test_beta_moments_recover_the_parameters() -> None:
    torch.manual_seed(0)
    alpha, beta = beta_moments(torch.distributions.Beta(2.0, 5.0).sample((50_000,)))
    assert alpha == pytest.approx(2.0, rel=0.05) and beta == pytest.approx(5.0, rel=0.05)


def test_bmm_threshold_is_the_posterior_crossing() -> None:
    torch.manual_seed(0)
    agree = torch.distributions.Beta(8.0, 2.0).sample((5000,))
    disagree = torch.distributions.Beta(2.0, 8.0).sample((5000,))
    confidence = torch.cat([agree, disagree])
    labels = torch.cat([torch.ones(5000, dtype=torch.bool), torch.zeros(5000, dtype=torch.bool)])
    middle = bmm_threshold(confidence, labels, posterior=0.5)
    assert middle == pytest.approx(0.5, abs=0.03)  # mirror-image components cross in the middle
    assert bmm_threshold(confidence, labels, posterior=0.4) <= middle <= bmm_threshold(confidence, labels, 0.9)


def test_bmm_threshold_defaults_without_both_classes() -> None:
    confidence = torch.rand(10)
    assert bmm_threshold(confidence, torch.ones(10, dtype=torch.bool), default=0.77) == 0.77
    agree = torch.zeros(10, dtype=torch.bool)
    agree[0] = True
    assert bmm_threshold(confidence, agree, default=0.77) == 0.77


def test_free_calibration_pairs() -> None:
    model = economize.apply(make_tiny_model(), FreeConfig(exit_layer=2))
    confidence, agree = free_calibration_pairs(model, _ids())
    assert confidence.shape == agree.shape == (48,)
    assert agree.dtype == torch.bool and bool(((confidence >= 0) & (confidence <= 1)).all())


# ---- FREE: training ------------------------------------------------------------------------------------------------
def test_monotone_layer_map_is_the_least_cost_monotone_assignment() -> None:
    costs = torch.rand(3, 5, generator=torch.Generator().manual_seed(0))
    best = min(itertools.combinations_with_replacement(range(5), 3),
               key=lambda m: sum(float(costs[i, j]) for i, j in enumerate(m)))
    assert monotone_layer_map(costs) == list(best)
    with pytest.raises(ConfigError):
        monotone_layer_map(torch.full((2, 3), float("inf")))


def test_kd_layer_maps() -> None:
    assert kd_layer_map("last", 2, 6) == [5, 6]
    assert kd_layer_map("uniform", 3, 6) == [2, 4, 6]
    assert kd_layer_map("uniform", 2, 6) == [3, 6]
    diagonal = torch.ones(2, 6)
    diagonal[0, 0] = diagonal[1, 1] = 0  # j = i is the shallow layer itself and must not be chosen
    diagonal[0, 3] = diagonal[1, 2] = 0.5
    assert kd_layer_map("dynamic", 2, 6, diagonal) == [3, 3]
    with pytest.raises(ConfigError):
        kd_layer_map("dynamic", 2, 6)
    with pytest.raises(ConfigError):
        kd_layer_map("nearest", 2, 6)


@pytest.mark.parametrize("kwargs", [{"threshold": 0.0}, {"threshold": 1.5}, {"adaptation": "lora"},
                                    {"max_exits": 0}, {"posterior": 0.0}])
def test_free_config_errors(kwargs: dict) -> None:
    with pytest.raises(ConfigError):
        FreeConfig(exit_layer=2, **kwargs)


@pytest.mark.parametrize("mapping", ["dynamic", "last", "uniform", "none"])
def test_free_objective_trains_the_model(mapping: str) -> None:
    model = economize.apply(make_tiny_model(), FreeConfig(exit_layer=2))
    assert isinstance(default_objective(model), FreeObjective)
    loss = _backward(model, FreeObjective(kd_mapping=mapping, parameters="all"))
    assert loss == loss and loss > 0
    assert _grad(model.adapter.layer(1).parameters()) > 0


def test_free_readout_trains_with_early_exit_kl() -> None:
    model = economize.apply(make_tiny_model(), FreeConfig(exit_layer=2, head="readout"))
    free = model.techniques["free"]
    _backward(model, EarlyExitKL(technique="free"))
    assert _grad(free.parameters()) > 0


# ---- DMTD ----------------------------------------------------------------------------------------------------------
def test_dmtd_resolves_its_layers() -> None:
    model = economize.apply(make_tiny_model(), DmtdConfig(decoding_layers=2, adaptation="frozen"))
    dmtd = model.techniques["dmtd"]
    assert (dmtd.config.target_layer, dmtd.config.source_layer) == (4, 4)
    assert list(dmtd.parameters()) == []
    assert dmtd.plt == pytest.approx(percentage_layers_per_token(6, 2, 3)) == pytest.approx(10 / 18)


@pytest.mark.parametrize("kwargs", [{"exit_layer": 5}, {"target_layer": 4}, {"extra_source_layers": (2,)},
                                    {"decoding_layers": 0}, {"cycle_length": 0}])
def test_dmtd_config_errors(kwargs: dict) -> None:
    with pytest.raises(ConfigError):
        DmtdConfig(**{"decoding_layers": 2, **kwargs})


def test_dmtd_needs_fewer_decoding_layers_than_the_model() -> None:
    with pytest.raises(ConfigError):
        economize.apply(make_tiny_model(), DmtdConfig(decoding_layers=LAYERS))


@pytest.mark.parametrize("tau", [2, 3])
def test_dmtd_cost_is_the_plt(tau: int) -> None:
    model = economize.apply(make_tiny_model(), DmtdConfig(decoding_layers=2, cycle_length=tau, adaptation="frozen"))
    dmtd = model.techniques["dmtd"]
    result = model.decode(_prompt(), 6 * tau + 1, decoding=dmtd.decode_config())
    summary = dmtd.cost_summary(result)
    assert summary["relative_sequential_layers"] == pytest.approx(dmtd.plt)
    assert summary["relative_effective_layer_tokens"] == pytest.approx(1.0)  # every token runs every layer once
    drafts = {(e.first_layer, e.last_layer) for e in result.cost.events if e.kind == CostKind.DRAFT}
    assert drafts == {(5, 6)}


def test_dmtd_rollout_follows_the_cycle() -> None:
    model = economize.apply(make_tiny_model(), DmtdConfig(decoding_layers=2, cycle_length=4, adaptation="lora"))
    objective = default_objective(model)
    assert isinstance(objective, DmtdRollout)
    _backward(model, objective)
    assert (objective.horizon, objective.stride) == (3, 4)
    assert _grad(model.techniques["dmtd"].parameters()) > 0


def test_dmtd_rollout_rejects_a_cycle_of_one() -> None:
    model = economize.apply(make_tiny_model(), DmtdConfig(decoding_layers=2, cycle_length=1, adaptation="lora"))
    with pytest.raises(ConfigError):
        _backward(model, DmtdRollout())


# ---- Hidden Transfer -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("context", ["teacher", "draft"])
@pytest.mark.parametrize("noise", [False, True])
def test_hidden_transfer_is_lossless(context: str, noise: bool) -> None:
    base = make_tiny_model()
    model = economize.apply(base, HiddenTransferConfig(train_context=context))
    technique = model.techniques["hidden_transfer"]
    assert technique.transfer_layers == (4, 4, 5)  # round(0.625 * 6), round(0.75 * 6), round(0.875 * 6)
    if noise:
        perturb(technique, 0.05)
    prompt = _prompt()
    result = model.decode(prompt, 24, decoding=technique.decode_config())
    assert torch.equal(result.tokens, reference_greedy(base, prompt, 24))
    kinds = (CostKind.DRAFT, CostKind.DRAFT_WASTED)
    drafts = {(e.first_layer, e.last_layer) for e in result.cost.events if e.kind in kinds}
    assert drafts <= {(5, 6), (6, 6)} and (5, 6) in drafts


def test_hidden_transfer_draft_capacity() -> None:
    model = economize.apply(make_tiny_model(), HiddenTransferConfig(transfer_layers=(3, 4)))
    drafter = model.techniques["hidden_transfer"].drafter(model.runner)
    assert [drafter.draft_capacity(n) for n in (1, 2, 3, 5)] == [0, 1, 2, 2]


def test_hidden_transfer_train_logits_equal_the_teacher_steps() -> None:
    model = economize.apply(make_tiny_model(), HiddenTransferConfig())
    technique = model.techniques["hidden_transfer"]
    perturb(technique, 0.05)
    ids = _ids()
    batch = Batch(ids)
    starts = torch.tensor([0, 5, 17])
    with torch.no_grad():
        teacher = run_teacher(model, batch, technique.train_request(model))
        logits = technique.train_logits(model, batch, teacher)
        states = technique.step_states(model, batch, teacher, starts, 3)
    assert sorted(logits) == [2, 3, 4] and all(v.shape == (2, 24, VOCAB) for v in logits.values())
    for i, state in enumerate(states, 1):
        torch.testing.assert_close(technique.step_logits(state, i), logits[i + 1][:, starts], rtol=1e-4, atol=1e-4)


def test_hidden_transfer_contexts_agree_on_the_first_step() -> None:
    ids = _ids()
    outputs = {}
    for context in ("teacher", "draft"):
        model = economize.apply(make_tiny_model(), HiddenTransferConfig(train_context=context))
        technique = model.techniques["hidden_transfer"]
        perturb(technique, 0.05)
        with torch.no_grad():
            teacher = run_teacher(model, Batch(ids), technique.train_request(model))
            outputs[context] = technique.step_states(model, Batch(ids), teacher, torch.tensor([3, 9]), 3)
    torch.testing.assert_close(outputs["teacher"][0], outputs["draft"][0])
    assert not torch.allclose(outputs["teacher"][2], outputs["draft"][2])  # later steps see their own pseudo states


@pytest.mark.parametrize("context", ["teacher", "draft"])
@pytest.mark.parametrize("objective", ["default", "distill_kl"])
def test_hidden_transfer_trains_every_map(context: str, objective: str) -> None:
    model = economize.apply(make_tiny_model(), HiddenTransferConfig(train_context=context))
    technique = model.techniques["hidden_transfer"]
    assert isinstance(default_objective(model), HiddenTransferDistill)
    chosen = (HiddenTransferDistill(eval_first=2, eval_stride=4, eval_horizon=4) if objective == "default"
              else DistillKL(technique="hidden_transfer"))
    loss = _backward(model, chosen)
    assert loss == loss and loss > 0
    assert all(float(w.weight.grad.abs().sum()) > 0 for w in technique.transfers)


@pytest.mark.parametrize("kwargs", [{"transfer_layers": (5, 4)}, {"transfer_layers": (6,)}, {"transfer_layers": (0,)},
                                    {"transfer_layers": ()}, {"transfer_fractions": (1.0,)},
                                    {"transfer_fractions": ()}, {"train_context": "oracle"}])
def test_hidden_transfer_config_errors(kwargs: dict) -> None:
    with pytest.raises(ConfigError):
        economize.apply(make_tiny_model(), HiddenTransferConfig(**kwargs))
