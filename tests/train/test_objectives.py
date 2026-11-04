"""The objective registry and the built-in objectives other than ``jump_rollout`` (``ce``, ``layerskip`` and
``distill_kl``), checked against direct computations with the Hugging Face model, plus the teacher helpers."""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch

import economize
from economize.errors import IncompatibleTechniquesError
from economize.techniques.early_exit import LayerSkip
from economize.techniques.early_exit.layerskip import exit_weights, layer_dropout_rates
from economize.techniques.jump import JumpRollout
from economize.train import (
    DEFAULT_OBJECTIVES,
    OBJECTIVES,
    Batch,
    CrossEntropy,
    DistillKL,
    Objective,
    ObjectiveModule,
    ObjectiveOutput,
    ParamGroup,
    TeacherRequest,
    build_objective,
    default_objective,
    evaluate,
    fit,
    larger_teacher,
    register_objective,
    run_teacher,
    teacher_pass,
)
from economize.train.objective import register_default_objective
from economize.train.objectives import STATE_LOSSES
from economize.train.objectives.base import base_parameter_groups
from economize.train.objectives.divergences import DIVERGENCES
from economize.train.objectives.weights import target_weights
from economize.train.teacher import make_teacher_head
from helpers import VOCAB, make_identity_jump, make_tiny_model

LAYERS = 6


@pytest.fixture
def ids() -> torch.Tensor:
    return torch.randint(0, VOCAB, (3, 20), generator=torch.Generator().manual_seed(1))


@pytest.fixture
def masks(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Right padding of the last row and a loss mask that leaves out each row's first five predictions."""
    attention = torch.ones_like(ids)
    attention[2, -6:] = 0
    loss_mask = torch.ones_like(ids)
    loss_mask[:, :5] = 0
    return attention, loss_mask


def _module_loss(model: Any, objective: Objective, batch: Batch) -> dict[str, Any]:
    module = ObjectiveModule(model, objective)
    try:
        return module(batch.input_ids, batch.attention_mask, batch.loss_mask)
    finally:
        module.close()


def _manual_nll(logits: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    """Mean NLL of token ``t + 1`` under ``logits[:, t]`` over the counted positions."""
    nll = torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")
    if weights is None:
        return nll.mean()
    return (nll * weights).sum() / weights.sum()


def _counted(attention: torch.Tensor, loss_mask: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Whether the prediction of token ``t + k`` at position ``t`` counts."""
    length = attention.shape[1]
    return (attention[:, k:] * loss_mask[:, k - 1 : length - 1]).float()


# ---- registry ---------------------------------------------------------------------------------------------------
def test_builtin_objectives_are_registered() -> None:
    assert {"jump_rollout", "distill_kl", "layerskip", "ce"} <= set(OBJECTIVES.names())
    assert OBJECTIVES.get("ce") is CrossEntropy and CrossEntropy.name == "ce"
    assert DEFAULT_OBJECTIVES["jump"] == "jump_rollout"


def test_default_objective_follows_the_attached_technique(toy_heads: Any) -> None:
    objective = default_objective(make_identity_jump(make_tiny_model()))
    assert isinstance(objective, JumpRollout) and objective.technique == "jump"
    with pytest.raises(ValueError, match="no default objective"):
        default_objective(economize.apply(make_tiny_model(), toy_heads()))


def test_registering_an_objective_with_a_default(toy_heads: Any) -> None:
    @dataclass
    class HeadNorm(Objective):
        technique: str = "toy_heads"

        def parameter_groups(self, model: Any) -> list[ParamGroup]:
            return [ParamGroup("heads", list(model.techniques[self.technique].parameters()))]

        def loss(self, model: Any, batch: Batch, teacher: Any) -> ObjectiveOutput:
            return ObjectiveOutput(sum(p.square().sum() for p in self.parameter_groups(model)[0].params))

    register_objective("head_norm", default_for=("toy_heads",))(HeadNorm)
    try:
        assert HeadNorm.name == "head_norm" and OBJECTIVES.get("head_norm") is HeadNorm
        assert isinstance(default_objective(economize.apply(make_tiny_model(), toy_heads())), HeadNorm)
    finally:
        OBJECTIVES.unregister("head_norm")
        DEFAULT_OBJECTIVES.pop("toy_heads")


def test_build_objective_and_hyperparameters() -> None:
    objective = build_objective("jump_rollout", horizon=4, stride=2)
    assert isinstance(objective, JumpRollout) and (objective.horizon, objective.stride) == (4, 2)
    assert build_objective(objective) is objective
    changed = build_objective(objective, horizon=6)
    assert changed.horizon == 6 and objective.horizon == 4
    names = set(LayerSkip().hyperparameters())
    assert "exit_scale" in names and not any(name.startswith("_") for name in names)
    assert LayerSkip().with_options(rotation=2).rotation == 2
    for bad in ({"divergence": "nope"}, {"state_loss": "l1"}):
        with pytest.raises((KeyError, ValueError)):
            build_objective("jump_rollout", **bad)
    with pytest.raises(ValueError):
        LayerSkip(curriculum="random")  # type: ignore[arg-type]


def test_target_weights() -> None:
    batch = Batch(torch.zeros(1, 4, dtype=torch.long), torch.tensor([[1, 1, 1, 0]]), torch.tensor([[0, 1, 1, 1]]))
    assert target_weights(batch, slice(1, None)).tolist() == [[False, True, False]]
    assert target_weights(batch, torch.tensor([1, 2])).tolist() == [[False, True]]
    assert target_weights(Batch(batch.input_ids), slice(1, None)) is None
    for bad in (slice(0, None), slice(1, None, 2)):
        with pytest.raises(ValueError):
            target_weights(batch, bad)


# ---- ce ---------------------------------------------------------------------------------------------------------
def test_ce_is_the_next_token_nll(ids: torch.Tensor, masks: tuple[torch.Tensor, torch.Tensor]) -> None:
    base = make_tiny_model()
    model = economize.apply(base)
    attention, loss_mask = masks
    with torch.no_grad():
        logits = base(ids, attention_mask=attention).logits
    module = ObjectiveModule(model, CrossEntropy(parameters="all"))
    try:
        assert module.trains_base and [g.name for g in module.groups] == ["base"]
        out = module(ids, attention, loss_mask)
        out["loss"].backward()
    finally:
        module.close()
    expected = _manual_nll(logits, ids, _counted(attention, loss_mask))
    torch.testing.assert_close(out["loss"], expected)
    torch.testing.assert_close(out["metrics"]["nll"], expected)
    assert all(p.grad is not None for p in base.parameters())
    assert not any(p.requires_grad for p in base.parameters())  # restored by close()


def test_base_parameter_selection(toy_heads: Any) -> None:
    plain = economize.apply(make_tiny_model())
    with pytest.raises(ValueError, match="nothing to train"):
        base_parameter_groups(plain, "trainable")
    with pytest.raises(ValueError, match="unknown"):
        base_parameter_groups(plain, "some")  # type: ignore[arg-type]
    (group,) = base_parameter_groups(plain, "all")
    assert (group.name, group.kind) == ("base", "backbone")
    assert len(group.params) == len(list(plain.base_model.parameters()))
    heads = economize.apply(make_tiny_model(), toy_heads())
    (group,) = base_parameter_groups(heads, "none")
    assert (group.name, group.kind, len(group.params)) == ("techniques", "head", 4)
    base, added = base_parameter_groups(heads, "all")
    assert (base.name, added.name) == ("base", "techniques")
    heads.base_model.model.norm.weight.requires_grad_(True)
    (base, _) = base_parameter_groups(heads, "trainable")
    assert base.params == [heads.base_model.model.norm.weight]


# ---- layerskip --------------------------------------------------------------------------------------------------
def test_layerskip_schedules() -> None:
    rates = layer_dropout_rates(LAYERS, 0.2)
    assert rates[0] == 0.0 and rates[-1] == pytest.approx(0.2)
    assert all(a < b for a, b in itertools.pairwise(rates))
    assert layer_dropout_rates(1, 0.2) == [0.0]
    # e(l) = e_scale * sum_{i < l} i before the last block, L - 1 + e_scale * sum_{i < L - 1} i at the last block
    assert exit_weights(4, 0.1) == pytest.approx([0.0, 0.1, 0.3, 3.3])


def test_layerskip_rotational_curriculum() -> None:
    objective = LayerSkip(rotation=2)
    objective.begin_step(0, 10)
    assert objective._enabled(LAYERS) == (1, 3, 5, 6)
    objective.begin_step(1, 10)
    assert objective._enabled(LAYERS) == (2, 4, 6)
    assert LayerSkip(curriculum="none", exit_layers=(2,))._enabled(LAYERS) == (2, 6)


def test_layerskip_evaluation_reads_every_exit(ids: torch.Tensor) -> None:
    base = make_tiny_model()
    model = economize.apply(base)
    metrics = evaluate(model, LayerSkip(exit_layers=(2, 4)), ids, batch_size=3)
    assert set(metrics) == {"loss", "nll2", "nll4", "nll6"}
    assert all(not layer._forward_hooks for layer in base.model.layers)  # installed for the call only
    with torch.no_grad():
        out = base(ids, output_hidden_states=True)
        for layer in (2, 4):
            expected = _manual_nll(model.adapter.logits(out.hidden_states[layer]), ids, None)
            assert metrics[f"nll{layer}"] == pytest.approx(expected.item(), rel=1e-5)
        assert metrics["nll6"] == pytest.approx(_manual_nll(out.logits, ids, None).item(), rel=1e-5)
    weights = exit_weights(LAYERS, 0.1)
    expected_loss = sum(weights[l - 1] * metrics[f"nll{l}"] for l in (2, 4, 6)) / sum(weights[l - 1] for l in (2, 4, 6))
    assert metrics["loss"] == pytest.approx(expected_loss, rel=1e-5)


def test_layerskip_training_pass(ids: torch.Tensor) -> None:
    base = make_tiny_model()
    model = economize.apply(base)
    objective = LayerSkip(curriculum="none", dropout_max=0.0, exit_layers=(3,), parameters="all")
    module = ObjectiveModule(model, objective)
    try:
        assert module.trains_base and base.training
        assert len(base.model.layers[2]._forward_hooks) == 1
        out = module(ids)
        assert set(out["metrics"]) == {"nll3", "nll6"}
        out["loss"].backward()
        assert base.model.layers[0].mlp.up_proj.weight.grad is not None
    finally:
        module.close()
    assert all(not layer._forward_hooks for layer in base.model.layers)
    assert not base.training
    with pytest.raises(RuntimeError, match="before setup"):
        objective.loss(model, Batch(ids), None)


def test_layerskip_dropout_skips_whole_blocks(ids: torch.Tensor) -> None:
    """With every block but the first dropped, the output is the first block's state through the head."""
    base = make_tiny_model()
    model = economize.apply(base)
    objective = LayerSkip(curriculum="none", dropout_max=1e9, exit_layers=(), parameters="all")
    module = ObjectiveModule(model, objective)
    try:
        dropped = module(ids)["metrics"]["nll6"]
    finally:
        module.close()
    with torch.no_grad():
        first = base(ids, output_hidden_states=True).hidden_states[1]
        expected = _manual_nll(model.adapter.logits(first), ids, None)
    torch.testing.assert_close(dropped, expected)


# ---- distill_kl -------------------------------------------------------------------------------------------------
def _heads_model(toy_heads: Any, heads: int = 2) -> Any:
    return economize.apply(make_tiny_model(), toy_heads(heads=heads))


def _kl(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student) per position from logits."""
    p = teacher.float().log_softmax(-1)
    return (p.exp() * (p - student.float().log_softmax(-1))).sum(-1)


def test_distill_kl_of_untrained_heads(toy_heads: Any, ids: torch.Tensor,
                                       masks: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Zero-initialized heads draft token ``t + k`` with the model's distribution of token ``t + 1``."""
    model = _heads_model(toy_heads)
    attention, loss_mask = masks
    with torch.no_grad():
        logits = model.base_model(ids, attention_mask=attention).logits
    out = _module_loss(model, DistillKL(step_decay=0.5), Batch(ids, attention, loss_mask))
    assert out["metrics"]["kl1"].item() == pytest.approx(0.0, abs=1e-6)
    weights = _counted(attention, loss_mask, 2)
    kl2 = (_kl(logits[:, 1:-1], logits[:, :-2]) * weights).sum() / weights.sum()
    torch.testing.assert_close(out["metrics"]["kl2"], kl2)
    torch.testing.assert_close(out["loss"], (out["metrics"]["kl1"] + 0.5 * kl2) / 1.5)
    out["loss"].backward()
    assert all(p.grad is not None for p in model.techniques["toy_heads"].parameters())


def test_distill_kl_options(toy_heads: Any, ids: torch.Tensor) -> None:
    model = _heads_model(toy_heads, heads=1)
    with torch.no_grad():
        logits = model.base_model(ids).logits
    out = _module_loss(model, DistillKL(divergence="rkl", ce_weight=0.3), Batch(ids))
    nll = _manual_nll(logits, ids, None)
    torch.testing.assert_close(out["loss"], out["metrics"]["rkl1"] + 0.3 * nll)
    with pytest.raises(KeyError):
        DistillKL(divergence="nope")


def test_distill_kl_needs_one_trainable_drafter(toy_heads: Any) -> None:
    jump = make_identity_jump(make_tiny_model())
    with pytest.raises(IncompatibleTechniquesError):
        DistillKL().parameter_groups(jump)
    with pytest.raises(IncompatibleTechniquesError):
        DistillKL(technique="jump").parameter_groups(jump)
    model = _heads_model(toy_heads)
    (group,) = DistillKL(technique="toy_heads").parameter_groups(model)
    assert group.name == "toy_heads" and len(group.params) == 4
    request = DistillKL().teacher_request(model)
    assert request == TeacherRequest(hidden_layers=(LAYERS,))


# ---- teachers ---------------------------------------------------------------------------------------------------
def test_larger_teacher_replaces_the_head(ids: torch.Tensor) -> None:
    base = make_tiny_model()
    model = economize.apply(base)
    batch = Batch(ids)
    own = run_teacher(model, batch, TeacherRequest(hidden_layers=(2, LAYERS)))
    same = make_teacher_head(model, copy.deepcopy(base))(batch, own)
    torch.testing.assert_close(same.log_probs(slice(None)), own.log_probs(slice(None)))
    other = make_tiny_model(seed=3, num_layers=3)
    replaced = larger_teacher(other, vocab_size=100)(batch, own)
    with torch.no_grad():
        expected = other(ids).logits[..., :100].log_softmax(-1)
    torch.testing.assert_close(replaced.log_probs(slice(None)), expected, rtol=1e-5, atol=1e-5)
    assert replaced.hidden[2] is own.hidden[2]  # rollouts still start from the trained model's states
    assert make_teacher_head(model, None) is None

    def head(batch: Batch, outputs: Any) -> Any:
        return outputs

    assert make_teacher_head(model, head) is head



def test_standalone_larger_teacher_matches_the_replaced_outputs(ids: torch.Tensor) -> None:
    model = economize.apply(make_tiny_model())
    batch = Batch(ids)
    teacher = larger_teacher(make_tiny_model(seed=3, num_layers=3), vocab_size=100)
    replaced = teacher(batch, run_teacher(model, batch, TeacherRequest(hidden_layers=(LAYERS,))))
    alone = teacher.standalone(batch, TeacherRequest(logits=True, own_pass=False))
    assert dict(alone.hidden) == {} and alone.cache is None
    torch.testing.assert_close(alone.log_probs(slice(None)), replaced.log_probs(slice(None)))
    assert alone.logits is not None and alone.logits.shape[-1] == 100


def test_teacher_pass_skips_the_own_pass_only_when_the_request_allows_it(
    ids: torch.Tensor, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = economize.apply(make_tiny_model())
    calls = []
    forward = model.runner.forward
    monkeypatch.setattr(model.runner, "forward", lambda *a, **k: calls.append(1) or forward(*a, **k))
    batch, request = Batch(ids), TeacherRequest(hidden_layers=(LAYERS,), own_pass=False)
    head = larger_teacher(make_tiny_model(seed=3, num_layers=3))
    teacher_pass(model, batch, request, head)
    assert calls == []  # the larger teacher runs alone
    teacher_pass(model, batch, replace(request, own_pass=True), head)
    teacher_pass(model, batch, request, lambda batch, outputs: outputs)  # a plain head needs the own pass
    teacher_pass(model, batch, request)  # without a head the own pass is the teacher
    assert calls == [1, 1, 1]
    assert request.merge(TeacherRequest()).own_pass and not request.merge(request).own_pass


def test_logit_distill_reads_only_the_larger_teachers_pass(ids: torch.Tensor) -> None:
    from economize.train.objectives import LogitDistill

    model = economize.apply(make_tiny_model())
    batch, objective = Batch(ids), LogitDistill(parameters="all")
    request = objective.teacher_request(model)
    assert not request.own_pass
    head = larger_teacher(make_tiny_model(seed=3, num_layers=3))
    with torch.no_grad():
        alone = objective.loss(model, batch, teacher_pass(model, batch, request, head)).loss
        both = objective.loss(model, batch, head(batch, run_teacher(model, batch, request))).loss
    torch.testing.assert_close(alone, both)

def _adapted_jump_model(adaptation: str = "frozen", **kwargs: Any) -> Any:
    torch.manual_seed(0)
    config = economize.JumpConfig(target_layer=2, source_layer=4, adaptation=adaptation, lora_rank=2, **kwargs)
    return economize.apply(make_tiny_model(), config)


def _eval_objective(**kwargs: Any) -> JumpRollout:
    return JumpRollout(**{"horizon": 4, "stride": 4, "eval_horizon": 4, "eval_stride": 6, "eval_first": 6, **kwargs})


def _rows(count: int = 16, length: int = 24) -> torch.Tensor:
    return torch.randint(0, VOCAB, (count, length), generator=torch.Generator().manual_seed(3))


# ---- registries -----------------------------------------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore::UserWarning", "ignore::FutureWarning")
def test_default_objectives_keep_their_dict_interface() -> None:
    assert DEFAULT_OBJECTIVES["jump"] == "jump_rollout"
    DEFAULT_OBJECTIVES["toy_a"] = "ce"
    assert DEFAULT_OBJECTIVES.setdefault("toy_a", "distill") == "ce"
    register_default_objective("toy_b", "distill")
    assert DEFAULT_OBJECTIVES.get("toy_b") == "distill" and "toy_b" in DEFAULT_OBJECTIVES.names()
    assert DEFAULT_OBJECTIVES.pop("toy_a") == "ce" and DEFAULT_OBJECTIVES.pop("toy_a", "gone") == "gone"
    del DEFAULT_OBJECTIVES["toy_b"]
    assert "toy_b" not in DEFAULT_OBJECTIVES


@pytest.mark.filterwarnings("ignore::UserWarning", "ignore::FutureWarning")
def test_state_losses_are_registered_and_selectable() -> None:
    assert {"nmse", "cos"} <= set(STATE_LOSSES.names())
    x = torch.randn(3, 8)
    assert torch.allclose(STATE_LOSSES.get("cos")(x, x), torch.zeros(3), atol=1e-6)
    with pytest.raises(KeyError):
        _eval_objective(state_loss="nope")
    model = _adapted_jump_model()
    loss = fit(model, _rows(), _eval_objective(state_loss="cos"), steps=1, batch_size=4, seq_len=24).losses
    assert len(loss) == 1 and torch.isfinite(torch.tensor(loss)).all()


# ---- masked divergences stay finite ----
# ---- divergences on log-probabilities with -inf entries -----------------------------------------------------------
def _masked_log_probs(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite log-probabilities ``[3, 8]`` and the same logits with tokens 5.. masked to ``-inf``."""
    logits = torch.randn(3, 8, generator=torch.Generator().manual_seed(seed))
    masked = logits.clone()
    masked[:, 5:] = float("-inf")
    return torch.log_softmax(logits, -1), torch.log_softmax(masked, -1)


@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")
@pytest.mark.parametrize("name", ["rkl", "js"])
def test_student_masked_divergences_are_finite(name: str) -> None:
    """A student that masks tokens to ``-inf`` (a draft vocabulary, a top-k head) gives a finite divergence and
    finite gradients under the divergences that weight by the student (``0 * -inf`` was NaN)."""
    teacher, student = _masked_log_probs(0)
    student.requires_grad_(True)
    value = DIVERGENCES.get(name)(teacher, student)
    assert torch.isfinite(value).all()
    value.sum().backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")
@pytest.mark.parametrize("name", ["kl", "js"])
def test_teacher_masked_divergences_are_finite(name: str) -> None:
    """The same for a teacher with masked tokens under the divergences that weight by the teacher."""
    _, teacher = _masked_log_probs(1)
    student = _masked_log_probs(2)[0].requires_grad_(True)
    value = DIVERGENCES.get(name)(teacher, student)
    assert torch.isfinite(value).all()
    value.sum().backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")
@pytest.mark.parametrize("name", ["kl", "rkl", "js"])
def test_divergences_keep_their_values_on_finite_inputs(name: str) -> None:
    teacher, student = _masked_log_probs(3)[0], _masked_log_probs(4)[0]
    mix = torch.logaddexp(teacher, student) - math.log(2)
    reference = {
        "kl": (teacher.exp() * (teacher - student)).sum(-1),
        "rkl": (student.exp() * (student - teacher)).sum(-1),
        "js": 0.5 * ((teacher.exp() * (teacher - mix)).sum(-1) + (student.exp() * (student - mix)).sum(-1)),
    }[name]
    torch.testing.assert_close(DIVERGENCES.get(name)(teacher, student), reference, rtol=0, atol=0)
