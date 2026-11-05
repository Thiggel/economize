"""The ``step_distill`` objective: decode-faithful draft steps of EAGLE-3, Medusa, an MTP module and early exit,
checked against direct computations with the Hugging Face model and against step-by-step decoding."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

import economize
from economize.errors import EconomizeError, IncompatibleTechniquesError
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.early_exit import EarlyExitConfig
from economize.techniques.lookup import LookupConfig
from economize.techniques.medusa import MedusaConfig
from economize.techniques.vendor_mtp import VendorMTPConfig
from economize.train import (
    OBJECTIVES,
    Batch,
    ObjectiveModule,
    StepDistill,
    StepDrafter,
    TeacherRequest,
    build_objective,
    evaluate,
    run_teacher,
)
from economize.train.objectives import STEP_CURVE_METRICS
from helpers import VOCAB, family_skip_reason, make_family_model, make_tiny_model, perturb

LENGTH = 24


def _batch(rows: int = 2, length: int = LENGTH, seed: int = 3) -> Batch:
    ids = torch.randint(3, VOCAB, (rows, length), generator=torch.Generator().manual_seed(seed))
    return Batch(ids)


def _model(config, *, seed: int = 0, family: str | None = None):
    torch.manual_seed(seed)
    model = economize.apply(make_tiny_model() if family is None else make_family_model(family), config)
    technique = next(iter(model.techniques.values()))
    perturb(technique, std=0.05, seed=seed)
    return model, technique


def _teacher(model, objective: StepDistill, batch: Batch):
    return run_teacher(model, batch, objective.teacher_request(model))


def _hf_logits(model, batch: Batch) -> torch.Tensor:
    with torch.no_grad():
        return model.base_model(batch.input_ids).logits.float()


def _kl(reference: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    """KL(reference || student) per position from log-probabilities."""
    return (reference.exp() * (reference - student)).sum(-1)


# ---- registry and configuration ---------------------------------------------------------------------------------
def test_registered_and_exported() -> None:
    assert OBJECTIVES.get("step_distill") is StepDistill
    objective = build_objective("step_distill", steps=3, stride=2)
    assert isinstance(objective, StepDistill) and objective.steps == 3 and objective.stride == 2
    assert STEP_CURVE_METRICS[0] == "kl" and "run" in STEP_CURVE_METRICS


@pytest.mark.parametrize("bad", [dict(steps=0), dict(stride=0), dict(eval_horizon=0), dict(step_decay=0.0),
                                 dict(ce_weight=-1.0), dict(head_chunk=-1), dict(divergence="nope")])
def test_rejects_bad_config(bad: dict) -> None:
    with pytest.raises((ValueError, EconomizeError)):
        StepDistill(**bad)


@pytest.mark.parametrize("config", [Eagle3Config(), MedusaConfig(num_heads=3), VendorMTPConfig(),
                                    EarlyExitConfig(exit_layer=2)])
def test_techniques_implement_the_protocol(config) -> None:
    model, technique = _model(config)
    assert isinstance(technique, StepDrafter)
    groups = StepDistill().parameter_groups(model)
    trained = {id(p) for g in groups for p in g.params}
    assert trained and all(p.requires_grad for g in groups for p in g.params)
    assert {id(p) for p in technique.parameters() if p.requires_grad} <= trained


def test_needs_a_step_drafter() -> None:
    model = economize.apply(make_tiny_model(), LookupConfig())
    with pytest.raises(IncompatibleTechniquesError):
        StepDistill().teacher_request(model)
    model, _ = _model(MedusaConfig(num_heads=2))
    with pytest.raises(IncompatibleTechniquesError):
        StepDistill(technique="eagle3").teacher_request(model)


def test_mtp_rollout_with_several_modules_reaches_their_count() -> None:
    """Several chained modules roll out in parallel up to their count; recycled steps beyond it need one module."""
    model, _ = _model(VendorMTPConfig(num_modules=2))
    batch = _batch()
    objective = StepDistill(steps=3)
    with pytest.raises(IncompatibleTechniquesError):
        objective.loss(model, batch, _teacher(model, objective, batch))
    for steps in (1, 2):
        objective = StepDistill(steps=steps)
        assert torch.isfinite(objective.loss(model, batch, _teacher(model, objective, batch)).loss)


# ---- the loss against direct computations ---------------------------------------------------------------------
def test_eagle3_loss_matches_manual() -> None:
    model, technique = _model(Eagle3Config())
    objective = StepDistill(steps=3, stride=1, step_decay=0.5, ce_weight=0.3)
    batch = _batch()
    teacher = _teacher(model, objective, batch)
    out = objective.loss(model, batch, teacher)
    ids = batch.input_ids
    reference = _hf_logits(model, batch).log_softmax(-1)
    starts = torch.arange(0, LENGTH - 3 - 1)
    states = technique.rollout(teacher.hidden, 3)
    terms = []
    for k in range(1, 4):
        positions = starts + k
        student = technique.head_logits(states[k - 1][:, starts]).float().log_softmax(-1)
        kl = _kl(reference[:, positions], student).mean()
        nll = -student.gather(-1, ids[:, positions + 1, None])[..., 0].mean()
        terms.append(kl + 0.3 * nll)
        torch.testing.assert_close(out.metrics[f"kl{k}"], terms[-1].detach(), rtol=1e-5, atol=1e-6)
    expected = (terms[0] + 0.5 * terms[1] + 0.25 * terms[2]) / 1.75
    torch.testing.assert_close(out.loss, expected, rtol=1e-5, atol=1e-6)


def test_medusa_steps_capped_and_padded() -> None:
    model, technique = _model(MedusaConfig(num_heads=2))
    objective = StepDistill(steps=5, eval_horizon=4, eval_first=2, eval_stride=3)
    batch = _batch()
    teacher = _teacher(model, objective, batch)
    out = objective.loss(model, batch, teacher)
    assert set(out.metrics) == {"kl1", "kl2"}
    reference = _hf_logits(model, batch).log_softmax(-1)
    final = teacher.hidden[model.adapter.num_layers]
    starts = torch.arange(0, LENGTH - 2 - 1)
    for k in (1, 2):
        student = technique.logits_of(final[:, starts], k).float().log_softmax(-1)
        expected = _kl(reference[:, starts + k], student).mean()
        torch.testing.assert_close(out.metrics[f"kl{k}"], expected, rtol=1e-5, atol=1e-6)
    curve = objective.evaluate(model, batch, teacher)
    assert set(curve) == set(STEP_CURVE_METRICS)
    assert curve["agree"][2:].eq(0).all() and curve["run"][2:].eq(0).all()
    assert curve["kl"][2:].isnan().all() and curve["kl"][:2].isfinite().all()
    assert curve["outside"][:2].eq(0).all()


def test_head_chunk_matches_unchunked_with_masks() -> None:
    model, technique = _model(Eagle3Config(draft_vocab=40))
    technique.set_draft_vocab(torch.randperm(VOCAB, generator=torch.Generator().manual_seed(2))[:40])
    attention = torch.ones(2, LENGTH, dtype=torch.long)
    attention[1, -5:] = 0
    loss_mask = torch.ones_like(attention)
    loss_mask[:, :4] = 0
    batch = Batch(_batch().input_ids, attention, loss_mask)
    whole = StepDistill(steps=3, ce_weight=0.5)
    teacher = _teacher(model, whole, batch)
    params = [p for g in whole.parameter_groups(model) for p in g.params]
    values, grads = [], []
    for chunk in (0, 7):
        objective = StepDistill(steps=3, ce_weight=0.5, head_chunk=chunk)
        loss = objective.loss(model, batch, teacher).loss
        values.append(loss.detach())
        grads.append(torch.autograd.grad(loss, params, allow_unused=True))
    torch.testing.assert_close(values[0], values[1], rtol=1e-5, atol=1e-6)
    for a, b in zip(*grads):
        if a is not None:
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
    assert any(g is not None and g.abs().sum() > 0 for g in grads[0])


def test_restricted_vocabulary_mask_and_outside() -> None:
    model, technique = _model(Eagle3Config(draft_vocab=40))
    vocab = torch.randperm(VOCAB, generator=torch.Generator().manual_seed(2))[:40].sort().values
    technique.set_draft_vocab(vocab)
    objective = StepDistill(steps=1, eval_horizon=1, eval_first=0, eval_stride=1)
    batch = _batch()
    teacher = _teacher(model, objective, batch)
    full = _hf_logits(model, batch).log_softmax(-1)
    starts = torch.arange(0, LENGTH - 1 - 1)
    positions = starts + 1
    restricted = full[:, positions][..., vocab].log_softmax(-1)
    student = technique.head_logits(technique.rollout(teacher.hidden, 1)[0][:, starts]).float().log_softmax(-1)
    inside = torch.isin(full.argmax(-1), vocab)[:, positions].float()
    kl = _kl(restricted, student)
    expected = (kl * inside).sum() / inside.sum()
    torch.testing.assert_close(objective.loss(model, batch, teacher).loss, expected, rtol=1e-5, atol=1e-6)
    curve = objective.evaluate(model, batch, teacher)
    starts = torch.arange(0, LENGTH - 1 - 2)
    positions = starts + 1
    mass = full[:, positions][..., vocab].exp().sum(-1)
    torch.testing.assert_close(curve["outside"][0], (1 - mass).mean(), rtol=1e-5, atol=1e-6)
    top = vocab[technique.head_logits(technique.rollout(teacher.hidden, 1)[0][:, starts]).argmax(-1)]
    agree = (top == full[:, positions].argmax(-1)).float().mean()
    torch.testing.assert_close(curve["agree"][0], agree)


# ---- decode faithfulness --------------------------------------------------------------------------------------
@pytest.mark.parametrize("family", [None, "gemma4_text"])
def test_mtp_steps_match_sequential_decoding(family: str | None) -> None:
    """Step ``k`` after an exact pass that ended at ``s``: the module's cache holds the entries of positions
    ``< s`` from the model's states, step 1 writes entry ``s`` from the model's state at ``s``, step ``k`` writes
    entry ``s + k - 1`` from step ``k - 1``'s state; one token at a time on a fresh cache per start."""
    if family is not None and family_skip_reason(family):
        pytest.skip(family_skip_reason(family))
    model, technique = _model(VendorMTPConfig(), family=family)
    batch = _batch(rows=1)
    ids = batch.input_ids
    objective = StepDistill(steps=4)
    teacher = _teacher(model, objective, batch)
    starts = torch.tensor([0, 3, 9, 15])
    with torch.no_grad():
        states = technique.step_states(model, batch, teacher, starts, 4)
        target = technique.target_state(teacher.hidden[model.adapter.num_layers])
        runner, module = model.runner, technique.layers[0]
        # the batched step states and the sequential reference reduce in another float32 order (1e-5 on some CPUs)
        for i, s in enumerate(starts.tolist()):
            cache = runner.new_cache(LENGTH, 1, kind="positional")
            prefix = torch.arange(s + 1)[None]
            ctx = runner.context(cache, prefix, trusted=True)
            raw, normed = module(technique.adapter, target[:, : s + 1], technique.embed(ids[:, 1 : s + 2], ctx), ctx)
            torch.testing.assert_close(states[0][:, i], normed[:, -1], rtol=1e-4, atol=1e-4)
            state = normed[:, -1:] if technique.layout.recycle == "normed" else raw[:, -1:]
            for k in range(2, 5):
                ctx = runner.context(cache, torch.tensor([[s + k - 1]]), trusted=True)
                raw, normed = module(technique.adapter, state, technique.embed(ids[:, s + k : s + k + 1], ctx), ctx)
                torch.testing.assert_close(states[k - 1][:, i], normed[:, 0], rtol=1e-4, atol=1e-4)
                state = normed if technique.layout.recycle == "normed" else raw


def test_early_exit_frozen_equals_unchanged_full_copy() -> None:
    frozen = economize.apply(make_tiny_model(), EarlyExitConfig(exit_layer=2))
    copied = economize.apply(make_tiny_model(), EarlyExitConfig(exit_layer=2, adaptation="full"))
    a, b = frozen.techniques["early_exit"], copied.techniques["early_exit"]
    perturb(a.readout, std=0.05)  # the layer copies stay equal to the model's layers
    b.readout.load_state_dict(a.readout.state_dict())
    batch = _batch()
    objective = StepDistill(steps=4, eval_horizon=4, eval_first=1, eval_stride=2)
    out_a = objective.loss(frozen, batch, _teacher(frozen, objective, batch))
    out_b = objective.loss(copied, batch, _teacher(copied, objective, batch))
    torch.testing.assert_close(out_a.loss, out_b.loss, rtol=1e-4, atol=1e-5)
    curve_a = objective.evaluate(frozen, batch, _teacher(frozen, objective, batch))
    curve_b = objective.evaluate(copied, batch, _teacher(copied, objective, batch))
    for name in STEP_CURVE_METRICS:
        torch.testing.assert_close(curve_a[name], curve_b[name], rtol=1e-4, atol=1e-5)


def test_eagle3_curve_matches_decoding_agreement() -> None:
    """The ``run`` of the curve is the fraction of starts whose first ``k`` drafts all equal the teacher's greedy
    tokens, computed from the teacher-forced rollout."""
    model, technique = _model(Eagle3Config(head="shared"))
    batch = _batch()
    objective = StepDistill(eval_horizon=3, eval_first=1, eval_stride=4)
    teacher = _teacher(model, objective, batch)
    curve = objective.evaluate(model, batch, teacher)
    greedy = _hf_logits(model, batch).argmax(-1)
    starts = torch.arange(1, LENGTH - 3 - 2, 4)
    states = technique.rollout(teacher.hidden, 3)
    alive = torch.ones(2, len(starts))
    for k in range(1, 4):
        top = technique.head_logits(states[k - 1][:, starts]).argmax(-1)
        alive = alive * (top == greedy[:, starts + k]).float()
        torch.testing.assert_close(curve["run"][k - 1], alive.mean())


# ---- training -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("config", [Eagle3Config(draft_vocab=40), MedusaConfig(num_heads=3), VendorMTPConfig(),
                                    EarlyExitConfig(exit_layer=2), EarlyExitConfig(exit_layer=2, adaptation="full")])
def test_gradients_reach_every_group_and_fit_runs(config) -> None:
    model, technique = _model(config)
    if isinstance(config, Eagle3Config):
        technique.set_draft_vocab(torch.arange(40))
    objective = StepDistill(steps=3, stride=2, head_chunk=8, eval_horizon=4, eval_first=2, eval_stride=4)
    batch = _batch()
    groups = objective.parameter_groups(model)
    loss = objective.loss(model, batch, _teacher(model, objective, batch)).loss
    loss.backward()
    for group in groups:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in group.params), group.name
    base = [p for p in model.base_model.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert not base
    curve = evaluate(model, objective, _batch(rows=4, seed=9), batch_size=2)
    assert set(curve) == set(STEP_CURVE_METRICS) and len(curve["run"]) == 4
    assert all(math.isfinite(v) for v in curve["kl"][: technique.step_limit() or 4])


def test_teacher_request_includes_last_layer() -> None:
    model, _ = _model(EarlyExitConfig(exit_layer=2, adaptation="lora"))
    request = StepDistill().teacher_request(model)
    assert isinstance(request, TeacherRequest)
    assert model.adapter.num_layers in request.hidden_layers and 2 in request.hidden_layers and request.cache


@pytest.mark.parametrize("config", [Eagle3Config(head="shared"), MedusaConfig(num_heads=2), VendorMTPConfig(),
                                    EarlyExitConfig(exit_layer=2, adaptation="full")])
def test_gemma4_text_loss_and_curve(config) -> None:
    """Gemma-4 (the E1 backbone): sliding and full layers, per-layer token inputs, shared KV."""
    reason = family_skip_reason("gemma4_text")
    if reason:
        pytest.skip(reason)
    model, technique = _model(config, family="gemma4_text")
    objective = StepDistill(steps=3, head_chunk=8, eval_horizon=3, eval_first=2, eval_stride=4)
    batch = _batch()
    loss = objective.loss(model, batch, _teacher(model, objective, batch)).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in technique.parameters())
    curve = evaluate(model, objective, batch, batch_size=2)
    assert all(math.isfinite(v) for v in curve["kl"][: technique.step_limit() or 3])


# ---- short sequences ----
def _loss(model: Any, objective: Any, ids: torch.Tensor) -> torch.Tensor:
    module = ObjectiveModule(model, objective)
    try:
        return module(ids, None, None)["loss"]
    finally:
        module.close()


def test_step_distill_short_sequences_fail_clearly() -> None:
    torch.manual_seed(0)
    model = economize.apply(make_tiny_model(), EarlyExitConfig(exit_layer=2))
    ids = torch.randint(0, VOCAB, (2, 8), generator=torch.Generator().manual_seed(0))
    for seed in range(8):
        torch.manual_seed(seed)
        assert torch.isfinite(_loss(model, StepDistill(steps=4, stride=8), ids))
    with pytest.raises(ValueError, match="too short"):
        _loss(model, StepDistill(steps=7, stride=8), ids)
    with pytest.raises(ValueError, match="too short"):
        evaluate(model, StepDistill(eval_horizon=4, eval_first=6), ids)
