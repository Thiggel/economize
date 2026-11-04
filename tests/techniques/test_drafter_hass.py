"""HASS: the ``hass`` objective on the EAGLE-3 drafter (harmonized objective distillation and context alignment)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

import economize
from economize import DecodeConfig
from economize.errors import IncompatibleTechniquesError
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.early_exit import EarlyExitConfig
from economize.techniques.hass import HassDistill, top_k_cross_entropy
from economize.train import ObjectiveModule, StepDistill, fit
from economize.train.objective import OBJECTIVES
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, perturb, reference_greedy

DRAFT = 96


def _model(config: Eagle3Config | None = None, std: float = 0.05, base: torch.nn.Module | None = None) -> Any:
    config = config or Eagle3Config()
    model = economize.apply(base if base is not None else make_tiny_model(), config)
    technique = model.techniques["eagle3"]
    if config.draft_vocab is not None:
        technique.set_draft_vocab(torch.randperm(VOCAB, generator=torch.Generator().manual_seed(3))[:DRAFT])
    if std:
        perturb(technique, std, seed=1)
    return model


def _loss(model: Any, objective: Any, ids: torch.Tensor) -> dict[str, Any]:
    module = ObjectiveModule(model, objective)
    try:
        return module(ids)
    finally:
        module.close()


def _ids(seed: int = 4, rows: int = 2, length: int = 14) -> torch.Tensor:
    return torch.randint(0, VOCAB, (rows, length), generator=torch.Generator().manual_seed(seed))


def test_registration_and_validation() -> None:
    assert OBJECTIVES.get("hass") is HassDistill
    objective = HassDistill()
    assert (objective.steps, objective.top_k, objective.top_k_weight, objective.step_decay) == (3, 10, 1.0, 1.0)
    assert objective.detach_steps
    for bad in (dict(top_k=0), dict(top_k_weight=-1.0), dict(steps=0)):
        with pytest.raises(ValueError):
            HassDistill(**bad)
    with pytest.raises(IncompatibleTechniquesError):
        HassDistill().parameter_groups(economize.apply(make_tiny_model(), EarlyExitConfig(exit_layer=2)))


def test_top_k_cross_entropy() -> None:
    generator = torch.Generator().manual_seed(0)
    teacher = torch.randn(3, 5, 20, generator=generator).log_softmax(-1)
    student = torch.randn(3, 5, 20, generator=generator).log_softmax(-1)
    top = teacher.topk(4, -1).indices
    expected = -(teacher.exp().gather(-1, top) * student.gather(-1, top)).sum(-1)
    torch.testing.assert_close(top_k_cross_entropy(teacher, student, 4), expected)
    full = -(teacher.exp() * student).sum(-1)
    torch.testing.assert_close(top_k_cross_entropy(teacher, student, 50), full)


def test_without_its_two_terms_hass_is_step_distill() -> None:
    model = _model(Eagle3Config(train_steps=3))
    ids = _ids()
    with torch.enable_grad():
        hass = _loss(model, HassDistill(top_k_weight=0.0, detach_steps=False), ids)
        plain = _loss(model, StepDistill(steps=3, step_decay=1.0), ids)
    torch.testing.assert_close(hass["loss"].detach(), plain["loss"].detach(), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("draft_vocab", [None, DRAFT])
def test_the_loss_adds_the_top_k_term_of_every_step(draft_vocab: int | None) -> None:
    """Loss = mean over steps of [KL + w * top-K CE] against the (restricted, renormalized) teacher, on the
    detached rollout, at every position with three steps to go (and teacher argmax inside the draft vocabulary)."""
    model = _model(Eagle3Config(draft_vocab=draft_vocab))
    technique = model.techniques["eagle3"]
    ids = _ids()
    length = ids.shape[1]
    with torch.enable_grad():
        out = _loss(model, HassDistill(top_k=5, top_k_weight=0.5), ids)
    with torch.no_grad():
        states = model.base_model(ids, output_hidden_states=True).hidden_states
        hidden = {layer: states[layer] for layer in (0, *technique.aux_layers)}
        teacher = model.base_model(ids).logits.float().log_softmax(-1)
        starts = torch.arange(length - 4)
        terms = []
        for k, step in enumerate(technique.rollout(hidden, 3), start=1):
            student = technique.head_logits(step[:, starts]).float().log_softmax(-1)
            reference = teacher[:, starts + k]
            mask = torch.ones(reference.shape[:2], dtype=torch.bool)
            if draft_vocab is not None:
                mask = technique.t2d[reference.argmax(-1)]
                reference = reference[..., technique.draft_ids].log_softmax(-1)
            kl = (reference.exp() * (reference - student)).sum(-1)
            value = kl + 0.5 * top_k_cross_entropy(reference, student, 5)
            terms.append(value[mask].mean())
    torch.testing.assert_close(out["loss"].detach(), torch.stack(terms).mean(), rtol=1e-5, atol=1e-6)
    assert {"kl1", "kl2", "kl3"} <= set(out["metrics"])


def test_context_alignment_changes_the_gradients_only() -> None:
    """``detach_steps`` gives the same rollout; the last step's loss no longer reaches the earlier steps through
    their outputs, so its gradient differs from the training-time test's."""
    model = _model()
    technique = model.techniques["eagle3"]
    ids = _ids()
    states = model.base_model(ids, output_hidden_states=True).hidden_states
    hidden = {layer: states[layer].detach() for layer in (0, *technique.aux_layers)}
    grads = []
    with torch.enable_grad():
        outputs = []
        for detach in (False, True):
            technique.zero_grad(set_to_none=True)
            rollout = technique.rollout(hidden, 3, detach_steps=detach)
            rollout[-1].pow(2).sum().backward()
            outputs.append([r.detach() for r in rollout])
            grads.append(technique.fc.weight.grad.clone())
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.allclose(grads[0], grads[1])


@pytest.fixture(scope="module")
def trained() -> Any:
    generator = torch.Generator().manual_seed(1)
    prompts = [torch.randint(1, VOCAB, (PROMPT_LEN,), generator=generator) for _ in range(4)]
    base = make_tiny_model()
    with torch.no_grad():
        rows = torch.stack([torch.cat([p, reference_greedy(base, p, 28)]) for p in prompts])
    model = economize.apply(make_tiny_model(), Eagle3Config(draft_vocab=DRAFT), decoding=DecodeConfig(cycle_length=4))
    model.techniques["eagle3"].set_draft_vocab(torch.unique(rows)[:DRAFT] if torch.unique(rows).numel() >= DRAFT
                                               else torch.randperm(VOCAB, generator=generator)[:DRAFT].sort().values)
    with torch.enable_grad():
        result = fit(model, rows, HassDistill(), steps=40, batch_size=4, lr=1e-2, warmup_steps=0, seed=0,
                     log_every=1)
    return model, base, prompts, result


def test_training_decreases_the_loss(trained: Any) -> None:
    *_, result = trained
    assert result.steps == 40 and result.losses[-1] < result.losses[0]


def test_a_hass_drafter_is_lossless_and_accepted(trained: Any) -> None:
    model, base, prompts, _ = trained
    results = model.decode_batch(prompts, 24)
    for prompt, result in zip(prompts, results):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 24))
    assert sum(a for r in results for _, a in r.acceptance) > 0


def test_checkpoint_round_trip_through_read_checkpoint_and_attach(trained: Any, tmp_path: Path) -> None:
    """HASS checkpoints are EAGLE-3 checkpoints; bench31.py --ckpts loads them as read_checkpoint + attach."""
    model, base, prompts, _ = trained
    economize.save(model, tmp_path / "final")
    checkpoint = economize.read_checkpoint(tmp_path / "final")
    loaded = economize.attach(make_tiny_model(), economize.Checkpoint(list(checkpoint.configs), checkpoint.states))
    for (name, a), (_, b) in zip(model.techniques["eagle3"].state_dict().items(),
                                 loaded.techniques["eagle3"].state_dict().items()):
        assert torch.equal(a, b), name
    config = DecodeConfig(cycle_length=4)
    first, second = model.decode(prompts[1], 20, decoding=config), loaded.decode(prompts[1], 20, decoding=config)
    assert torch.equal(first.tokens, second.tokens) and first.acceptance == second.acceptance
    assert torch.equal(second.tokens, reference_greedy(base, prompts[1], 20))
