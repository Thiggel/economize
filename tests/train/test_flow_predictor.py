"""The ``"flow"`` predictors: an MLP (or linear) mean followed by a rectified flow around it.

Without noise, the untrained flow predictor equals the MLP predictor with the same weights. Its flow-matching loss
trains the velocity network, enters the ``jump_rollout`` objective in place of the state loss, and drafting with it
is deterministic. The per-channel residual scale is a buffer, so it is saved and exported with the weights.
"""

from __future__ import annotations

import math

import pytest
import torch

import economize
from economize import DecodeConfig, JumpConfig
from economize.errors import ConfigError
from economize.techniques.jump import (
    PREDICTORS,
    FlowLinearPredictor,
    FlowPredictor,
    JumpRollout,
    LinearPredictor,
    MLPPredictor,
    jump_rollout,
)
from economize.train import Batch
from economize.train.teacher import run_teacher
from helpers import VOCAB, make_tiny_model

D, LENGTH, ROWS = 32, 24, 2


def _config(**kwargs) -> JumpConfig:
    return JumpConfig(**{"target_layer": 2, "source_layer": 4, "predictor": "flow", **kwargs})


def _inputs(seed: int = 0, shape: tuple[int, ...] = (2, 3)) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, D, generator=generator), torch.randn(*shape, D, generator=generator)


@pytest.mark.parametrize("name,mean_class", [("flow", MLPPredictor), ("flow_linear", LinearPredictor)])
def test_untrained_flow_without_noise_is_the_mean_predictor(name: str, mean_class: type) -> None:
    flow = PREDICTORS.get(name)(D, _config(predictor=name)).eval()
    mean = mean_class(D, _config(predictor=name))
    missing, unexpected = mean.load_state_dict(flow.state_dict(), strict=False)
    assert not missing and all(key.startswith("velocity.") or key == "residual_scale" for key in unexpected)
    source, embedding = _inputs()
    with torch.no_grad():
        out = flow(source, embedding)
        assert out.shape == (2, 3, D) and out.dtype == torch.float32
        torch.testing.assert_close(out, mean(source, embedding), atol=1e-3, rtol=1e-3)  # velocity out std 1e-3
        flow.velocity.out.weight.zero_()
        torch.testing.assert_close(flow(source, embedding), mean(source, embedding), atol=0, rtol=0)


def test_noise_follows_the_mode_and_the_flags() -> None:
    source, embedding = _inputs()
    flow = FlowPredictor(D, _config())
    with torch.no_grad():
        assert not torch.equal(flow(source, embedding), flow(source, embedding))  # training: noisy start
        flow.eval()
        torch.testing.assert_close(flow(source, embedding), flow(source, embedding), atol=0, rtol=0)
        noisy = FlowPredictor(D, _config(flow_eval_noise=True)).eval()
        assert not torch.equal(noisy(source, embedding), noisy(source, embedding))
        quiet = FlowPredictor(D, _config(flow_train_noise=False))
        torch.testing.assert_close(quiet(source, embedding), quiet(source, embedding), atol=0, rtol=0)
        still = FlowPredictor(D, _config(flow_sigma=0.0))
        torch.testing.assert_close(still(source, embedding), still(source, embedding), atol=0, rtol=0)


def test_auxiliary_loss_is_finite_and_trains_the_velocity_net() -> None:
    torch.manual_seed(0)
    flow = FlowPredictor(D, _config())
    source, embedding = _inputs()
    target = torch.randn(2, 3, D) + 5.0
    assert not flow.residual_scale.any()
    loss = flow.auxiliary_loss(source, embedding, target)
    assert loss.shape == (2, 3) and torch.isfinite(loss).all()
    assert flow.residual_scale.gt(0).all()  # set by the first update
    loss.mean().backward()
    assert flow.velocity.out.weight.grad.abs().sum() > 0
    assert flow.velocity.gate.weight.grad.abs().sum() > 0
    assert flow.bias.grad.abs().sum() > 0  # the mean's squared error
    assert MLPPredictor(D, _config(predictor="mlp")).auxiliary_loss(source, embedding, target) is None


@pytest.mark.parametrize("divergence_grad", [False, True])
def test_the_divergence_reaches_the_velocity_weights_only_when_asked(divergence_grad: bool) -> None:
    torch.manual_seed(0)
    flow = FlowPredictor(D, _config(flow_divergence_grad=divergence_grad))
    with torch.no_grad():
        flow.residual_scale.fill_(1.0)
        flow.velocity.out.weight.normal_(0, 0.1)
    source, embedding = _inputs()
    source.requires_grad_(True)
    flow(source, embedding).pow(2).sum().backward()
    assert flow.bias.grad.abs().sum() > 0 and source.grad.abs().sum() > 0
    assert (flow.velocity.out.weight.grad is not None) == divergence_grad
    reference = FlowPredictor(D, _config(flow_divergence_grad=True))
    reference.load_state_dict(flow.state_dict())
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        first = flow(source, embedding)
        torch.manual_seed(1)
        torch.testing.assert_close(first, reference(source, embedding), atol=0, rtol=0)


def test_the_residual_scale_leaves_out_masked_tokens() -> None:
    torch.manual_seed(0)
    source, embedding = _inputs()
    target = torch.randn(2, 3, D)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    padded = target.clone()
    padded[~mask] = 1e4  # padding states far from anything the mean predicts
    flow = FlowPredictor(D, _config())
    reference = FlowPredictor(D, _config())
    reference.load_state_dict(flow.state_dict())
    flow.auxiliary_loss(source, embedding, padded, mask)
    reference.auxiliary_loss(source[mask], embedding[mask], target[mask])
    torch.testing.assert_close(flow.residual_scale, reference.residual_scale)
    flow.auxiliary_loss(source, embedding, padded, torch.zeros_like(mask))  # nothing real: the scale stays
    torch.testing.assert_close(flow.residual_scale, reference.residual_scale)


def test_optimizer_steps_reduce_the_flow_loss() -> None:
    """The velocity net learns a conditional residual the mean cannot fit: the sign of the first source channel
    flips a target channel by +-3 around a mean of zero."""
    torch.manual_seed(0)
    flow = FlowPredictor(D, _config(flow_width=2.0))
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-2)
    generator = torch.Generator().manual_seed(1)
    source, embedding = torch.randn(256, D, generator=generator), torch.randn(256, D, generator=generator)
    target = torch.ones(256, D) + torch.randn(256, D, generator=generator) * 0.1
    target[:, 0] += 3 * source[:, 0].sign()

    def flow_loss() -> float:
        flow.eval()  # no scale update, same noise for both measurements
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            mean = flow._head(flow._inputs(source, embedding), embedding)
            value = flow.auxiliary_loss(source, embedding, target) - (mean - target).pow(2).mean(-1) / target.pow(
                2).mean(-1)
        flow.train()
        return value.mean().item()

    before = flow_loss()
    for _ in range(100):
        optimizer.zero_grad()
        flow.auxiliary_loss(source, embedding, target).mean().backward()
        optimizer.step()
    assert flow_loss() < 0.5 * before


def test_config_validates_the_flow_fields() -> None:
    for field, bad in (("flow_steps", 0), ("flow_sigma", -1.0), ("flow_width", 0.0)):
        with pytest.raises(ConfigError, match=field):
            _config(**{field: bad})
    config = _config(flow_steps=2, flow_sigma=0.5)
    assert JumpConfig.from_dict(config.to_dict()) == config
    assert isinstance(PREDICTORS.get("flow_linear")(D, _config(predictor="flow_linear")), FlowLinearPredictor)


# ---- in a model ----------------------------------------------------------------------------------------------------
def _model(**kwargs) -> economize.EfficientModel:
    torch.manual_seed(0)
    return economize.apply(make_tiny_model(), _config(**kwargs))


def _ids(seed: int = 3) -> torch.Tensor:
    return torch.randint(3, VOCAB, (ROWS, LENGTH), generator=torch.Generator().manual_seed(seed))


def test_rollout_passes_the_auxiliary_loss_and_the_objective_uses_it() -> None:
    model = _model()
    model.techniques.train()
    jump = model.techniques["jump"]
    objective = JumpRollout(horizon=3, stride=4)
    batch = Batch(_ids())
    teacher = run_teacher(model, batch, objective.teacher_request(model))

    seen = []

    def step(k, positions, h_m_hat, h_r_hat, auxiliary=None):
        seen.append(auxiliary)
        return h_m_hat

    starts = torch.tensor([1, 6, 11])
    jump_rollout(model, jump, batch.input_ids, dict(teacher.hidden), teacher.cache, starts, 3, step, auxiliary=True)
    assert len(seen) == 3 and all(a is not None and a.shape == (ROWS, 3) for a in seen)
    hidden = {layer: h for layer, h in teacher.hidden.items() if layer != jump.target_layer}
    with pytest.raises(ValueError, match="target layer"):
        jump_rollout(model, jump, batch.input_ids, hidden, teacher.cache, starts, 3, step, auxiliary=True)

    out = objective.loss(model, batch, teacher)
    assert {"aux1", "auxK", "nmse1", "kl1"} <= set(out.metrics) and math.isfinite(out.loss.item())
    out.loss.backward()
    assert jump.predictor.velocity.out.weight.grad.abs().sum() > 0
    jump.zero_grad()
    without = JumpRollout(horizon=3, stride=4, divergence="none", state_weight=1.0).loss(model, batch, teacher)
    without.loss.backward()  # the state term alone (the flow loss) still trains the velocity net
    assert jump.predictor.velocity.out.weight.grad.abs().sum() > 0


def test_fit_and_evaluate_with_the_flow_predictor(tmp_path) -> None:
    from economize.train import fit

    rows = torch.randint(0, VOCAB, (16, 32), generator=torch.Generator().manual_seed(3))
    model = _model(predictor="flow_linear", flow_steps=2)
    objective = JumpRollout(horizon=4, stride=4, eval_horizon=4, eval_stride=6, eval_first=6)
    result = fit(model, rows, objective, steps=3, batch_size=4, lr=2e-3, warmup_steps=1, seed=0,
                 output_dir=tmp_path)
    assert all(math.isfinite(loss) for loss in result.losses)
    assert model.techniques["jump"].predictor.residual_scale.gt(0).all()


def test_decoding_is_deterministic_and_lossless_and_save_load_keeps_the_scale(prompt, tmp_path) -> None:
    model = _model()
    jump = model.techniques["jump"]
    with torch.no_grad():
        jump.predictor.residual_scale.copy_(torch.rand_like(jump.predictor.residual_scale) + 0.5)
        for parameter in jump.predictor.velocity.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    model.eval()
    exact = model.decode(prompt, 12, decoding=DecodeConfig(cycle_length=1)).tokens
    first = model.decode(prompt, 12, decoding=DecodeConfig(cycle_length=4)).tokens
    second = model.decode(prompt, 12, decoding=DecodeConfig(cycle_length=4)).tokens
    assert first.tolist() == second.tolist() == exact.tolist()

    model.save_pretrained(tmp_path)
    loaded = economize.load(make_tiny_model(), tmp_path)
    torch.testing.assert_close(loaded.techniques["jump"].predictor.residual_scale, jump.predictor.residual_scale)


def test_export_round_trip_keeps_the_scale(tmp_path) -> None:
    from economize.backends.vllm import export_drafter
    from economize.techniques.jump.export import ExportedJumpDrafter

    model = _model()
    jump = model.techniques["jump"]
    with torch.no_grad():
        jump.predictor.residual_scale.copy_(torch.rand_like(jump.predictor.residual_scale) + 0.5)
    model.eval()
    drafter = ExportedJumpDrafter.from_pretrained(export_drafter(model, tmp_path / "drafter"))
    assert not drafter.training
    torch.testing.assert_close(drafter.predictor.residual_scale, jump.predictor.residual_scale, atol=0, rtol=0)
    d = jump.adapter.hidden_size
    states, embedding = [torch.randn(5, d)], torch.randn(5, d)
    with torch.no_grad():
        torch.testing.assert_close(drafter.predict(states, embedding), jump.predictor(states[0], embedding),
                                   atol=0, rtol=0)
