"""Training objectives of the dynamic-depth techniques: they train only the parameters the technique adds, ask for
a teacher pass only when they need one, and log their regularizers."""

from __future__ import annotations

import math

import pytest
import torch

import economize
from economize.errors import IncompatibleTechniquesError
from economize.techniques.dynamic_depth import (
    CalmConfig,
    FlexiDepthConfig,
    GateSkipConfig,
    LoRADropConfig,
    MoDConfig,
    SkipDecodeConfig,
)
from economize.train import fit
from economize.train.objective import OBJECTIVES
from helpers import VOCAB, make_tiny_model

CASES = [
    ("gateskip", GateSkipConfig(), dict(final_budget=0.5), {"ce", "mean_gate", "budget"}),
    ("mod", MoDConfig(router_bias=0.0), {}, {"ce", "router_bce", "causal_routed_fraction"}),
    ("flexidepth", FlexiDepthConfig(router_bias=0.0), {}, {"ce", "layer_penalty", "run_fraction"}),
    ("loradrop", LoRADropConfig(), dict(kl_weight=1.0, ce_weight=0.0), {"kl"}),
    ("calm", CalmConfig(confidence="classifier"), {}, {"bce", "consistent"}),
]


def _rows() -> torch.Tensor:
    return torch.randint(0, VOCAB, (16, 24), generator=torch.Generator().manual_seed(0))


def _fit(model: economize.EfficientModel, objective: object, steps: int = 6):
    with torch.enable_grad():
        return fit(model, _rows(), objective, steps=steps, batch_size=4, lr=3e-3, warmup_steps=0, seed=0, log_every=2)


@pytest.mark.parametrize("name, config, kwargs, keys", CASES, ids=[c[0] for c in CASES])
def test_objective_trains_only_the_technique(name: str, config: object, kwargs: dict, keys: set[str]) -> None:
    base = make_tiny_model()
    base_before = {k: v.clone() for k, v in base.state_dict().items()}
    model = economize.apply(base, config)
    technique = model.techniques[config.technique]
    before = {k: v.clone() for k, v in technique.state_dict().items()}
    result = _fit(model, OBJECTIVES.get(name)(technique=config.technique, **kwargs))
    assert len(result.losses) == 6 and all(math.isfinite(x) for x in result.losses)
    assert any(not torch.equal(before[k], v) for k, v in technique.state_dict().items())
    assert all(torch.equal(base_before[k], v) for k, v in base.state_dict().items())
    assert not any(p.requires_grad for p in base.parameters())
    assert result.history and keys <= set(result.history[-1])


@pytest.mark.parametrize("name, config, kwargs, keys", CASES[:4], ids=[c[0] for c in CASES[:4]])
def test_teacher_pass_only_for_the_divergence(name: str, config: object, kwargs: dict, keys: set[str]) -> None:
    model = economize.apply(make_tiny_model(), config)
    plain = OBJECTIVES.get(name)(technique=config.technique)
    distill = OBJECTIVES.get(name)(technique=config.technique, kl_weight=0.5)
    assert plain.teacher_request(model) is None
    assert distill.teacher_request(model).hidden_layers == (model.adapter.num_layers,)


def test_lm_weights_must_not_both_vanish() -> None:
    with pytest.raises(ValueError):
        OBJECTIVES.get("mod")(ce_weight=0.0, kl_weight=0.0)
    with pytest.raises(ValueError):
        OBJECTIVES.get("gateskip")(final_budget=0.0)


def test_calm_classifier_starts_at_chance() -> None:
    """The exit classifier is zero-initialized, so its first BCE is log 2 and the teacher returns every exit layer."""
    model = economize.apply(make_tiny_model(), CalmConfig(confidence="classifier", min_layer=2))
    objective = OBJECTIVES.get("calm")()
    assert objective.teacher_request(model).hidden_layers == tuple(range(2, model.adapter.num_layers + 1))
    result = _fit(model, objective, steps=2)
    assert result.losses[0] == pytest.approx(math.log(2), abs=1e-4)


@pytest.mark.parametrize(
    "name, config",
    [("mod", GateSkipConfig()), ("gateskip", SkipDecodeConfig()), ("calm", CalmConfig(confidence="softmax")),
     ("loradrop", None)],
    ids=["wrong-technique", "parameter-free", "calm-softmax", "missing"],
)
def test_objective_needs_its_technique(name: str, config: object) -> None:
    model = economize.apply(make_tiny_model(), *([config] if config is not None else []))
    technique = config.technique if config is not None and name != "calm" else name
    objective = OBJECTIVES.get(name)(technique=technique) if name != "calm" else OBJECTIVES.get(name)()
    with pytest.raises(IncompatibleTechniquesError):
        objective.parameter_groups(model)


@pytest.mark.parametrize("name, config", [("gateskip", GateSkipConfig()), ("mod", MoDConfig(router_bias=0.0))])
def test_chunked_lm_loss_matches_the_whole_batch_loss(name: str, config: object) -> None:
    """The head and the loss run over token chunks (a 262k vocabulary in float32 does not fit for a whole batch);
    loss and gradients equal those of the unchunked loss, also with padding and a loss mask."""
    from economize.techniques.dynamic_depth import student_pass
    from economize.train.objective import Batch
    from economize.train.teacher import run_teacher

    torch.manual_seed(0)
    model = economize.apply(make_tiny_model(), config)
    ids = _rows()[:3]
    attention = torch.ones_like(ids)
    attention[1, 17:] = 0
    loss_mask = torch.ones_like(ids)
    loss_mask[2, :5] = 0
    batch = Batch(ids, attention, loss_mask)
    params = [p for p in model.techniques[config.technique].parameters() if p.requires_grad]
    results = []
    for chunk in (0, 7, 1):
        objective = OBJECTIVES.get(name)(technique=config.technique, kl_weight=0.5, ce_weight=1.0, head_chunk=chunk)
        teacher = run_teacher(model, batch, objective.teacher_request(model))
        policy = objective.trained(model).make_policy()
        objective.configure(policy)
        with torch.enable_grad():
            if chunk == 0:  # the reference: the unchunked loss on the full logits
                loss, metrics = objective.lm_loss(batch, student_pass(model, batch, policy), teacher)
            else:
                from economize.techniques.dynamic_depth import student_states
                loss, metrics = objective.chunked_lm_loss(model, batch, student_states(model, batch, policy), teacher)
            grads = torch.autograd.grad(loss, params, allow_unused=True)
        results.append((loss.detach(), metrics, grads))
    reference = results[0]
    for loss, metrics, grads in results[1:]:
        torch.testing.assert_close(loss, reference[0], rtol=1e-5, atol=1e-6)
        assert metrics.keys() == reference[1].keys()
        for key in metrics:
            torch.testing.assert_close(metrics[key], reference[1][key], rtol=1e-5, atol=1e-6)
        for got, want in zip(grads, reference[2], strict=True):
            if want is None:
                assert got is None or not got.abs().any()
            else:
                torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-6)
