"""``fit``: training, logging, checkpoints and resume, gradient accumulation, evaluation and teachers."""

from __future__ import annotations

import json
import logging
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM

import economize
from economize.errors import ConfigError, NonFiniteLossError
from economize.techniques.early_exit import LayerSkip
from economize.techniques.jump import JumpRollout
from economize.train import (
    Batch,
    CrossEntropy,
    DistillKL,
    TensorSource,
    TrainConfig,
    evaluate,
    fit,
    latest_checkpoint,
    load_trained,
)
from economize.train.checkpoint import BASE_FILE, STATE_FILE
from economize.train.loop import LOG_FILE
from economize.train.objective import ObjectiveOutput, TeacherOutputs
from economize.train.utils import autocast_dtype as _dtype
from helpers import VOCAB, make_tiny_model

pytestmark = pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")

ROWS, LENGTH = 32, 32


@pytest.fixture
def rows() -> torch.Tensor:
    return torch.randint(0, VOCAB, (ROWS, LENGTH), generator=torch.Generator().manual_seed(3))


def _jump_model(seed: int = 0, adaptation: str = "frozen") -> Any:
    torch.manual_seed(seed)
    config = economize.JumpConfig(target_layer=2, source_layer=4, exit_layer=5, adaptation=adaptation, lora_rank=2)
    return economize.apply(make_tiny_model(), config)


def _objective(**kwargs: Any) -> JumpRollout:
    return JumpRollout(**{"horizon": 4, "stride": 4, "eval_horizon": 4, "eval_stride": 6, "eval_first": 6, **kwargs})


def _state(model: Any) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in model.techniques.state_dict().items()}


class Interrupt(Exception):
    pass


class _Rows(Dataset[torch.Tensor]):
    def __init__(self, rows: torch.Tensor) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.rows[index]


# ---- training -----------------------------------------------------------------------------------------------------
def test_fit_trains_a_jump_drafter(rows: torch.Tensor) -> None:
    model = _jump_model()
    before = _state(model)
    result = fit(model, rows, _objective(), steps=40, batch_size=8, lr=3e-3, warmup_steps=5, seed=0, log_every=10)
    assert result.steps == 40 and len(result.losses) == 40 and result.output_dir is None
    assert sum(result.losses[-5:]) < 0.92 * sum(result.losses[:5])
    assert [r["step"] for r in result.history] == [0, 10, 20, 30, 39]
    assert {"loss", "lr", "kl1", "klK", "nmse1", "nmseK", "time"} <= set(result.history[0])
    assert result.history[0]["lr"] == pytest.approx(3e-3 / 5)
    after = _state(model)
    assert any(not torch.equal(before[k], after[k]) for k in before)
    assert not any(p.requires_grad for p in model.base_model.parameters())
    assert all(p.requires_grad for p in model.technique_parameters())


def test_fit_uses_the_default_objective_and_checks_resume(rows: torch.Tensor) -> None:
    result = fit(_jump_model(), rows, steps=1, batch_size=2, log_every=1)
    assert "kl1" in result.history[0]
    with pytest.raises(ValueError, match="output_dir"):
        fit(_jump_model(), rows, _objective(), steps=1, resume=True)
    with pytest.raises(ValueError):
        _dtype("float7")
    assert _dtype("bf16") is torch.bfloat16 and _dtype("torch.float16") is torch.float16


def test_fit_writes_logs_and_final_weights(rows: torch.Tensor, tmp_path: Path) -> None:
    model = _jump_model(adaptation="lora")
    result = fit(model, rows, _objective(), steps=4, batch_size=4, log_every=2, output_dir=tmp_path, seed=1)
    assert result.final_path == tmp_path / "final"
    logged = [json.loads(line) for line in (tmp_path / LOG_FILE).read_text().splitlines()]
    assert [r["step"] for r in logged] == [0, 2, 3]
    meta = json.loads((tmp_path / "train_config.json").read_text())
    assert meta["objective"] == "jump_rollout" and meta["hyperparameters"]["horizon"] == 4
    assert meta["config"]["steps"] == 4 and meta["optimizer"]["lr"] == JumpRollout.defaults.lr
    assert not (tmp_path / "final" / BASE_FILE).exists()  # a drafter run leaves the base model alone
    loaded = economize.load(make_tiny_model(), tmp_path / "final")
    for name, value in model.techniques.state_dict().items():
        torch.testing.assert_close(loaded.techniques.state_dict()[name], value)
    stored = json.loads((tmp_path / "final" / "economize_config.json").read_text())
    assert stored["train"]["step"] == 4 and stored["train"]["objective"] == "jump_rollout"


def test_checkpoints_are_pruned(rows: torch.Tensor, tmp_path: Path) -> None:
    fit(_jump_model(), rows, _objective(), steps=5, batch_size=2, save_every=1, keep_checkpoints=2,
        output_dir=tmp_path)
    assert sorted(p.name for p in tmp_path.glob("checkpoint-*")) == ["checkpoint-3", "checkpoint-4"]
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint-4"
    assert (tmp_path / "checkpoint-4" / STATE_FILE).is_file()
    assert latest_checkpoint(tmp_path / "missing") is None


@pytest.mark.parametrize("source", ["tensor", "token_file", "shuffling_loader"])
def test_resume_continues_the_same_run(rows: torch.Tensor, tmp_path: Path, token_file: Path, source: str) -> None:
    """A run interrupted after step 4 and resumed from its checkpoint at step 3 ends where an uninterrupted run
    ends: same losses, same weights. A shuffling ``DataLoader`` is re-iterated on resume and must reshuffle into the
    same order."""

    def data() -> Any:
        if source == "shuffling_loader":
            return DataLoader(_Rows(rows), batch_size=4, shuffle=True)
        return rows if source == "tensor" else token_file

    options = dict(steps=6, batch_size=4, seq_len=LENGTH, lr=2e-3, warmup_steps=2, save_every=3, log_every=1, seed=0)
    full = fit(_jump_model(), data(), _objective(), output_dir=tmp_path / "full", **options)
    expected = _state(_jump_model())  # every run below starts from these weights

    def stop(record: dict[str, Any]) -> None:
        if record.get("step") == 4:
            raise Interrupt

    with pytest.raises(Interrupt):
        fit(_jump_model(), data(), _objective(), output_dir=tmp_path / "cut", callbacks=[stop], **options)
    assert latest_checkpoint(tmp_path / "cut") == tmp_path / "cut" / "checkpoint-3"
    model = _jump_model()
    assert all(torch.equal(v, expected[k]) for k, v in _state(model).items())
    resumed = fit(model, data(), _objective(), output_dir=tmp_path / "cut", resume=True, **options)
    assert resumed.steps == 6 and resumed.losses == full.losses
    assert [r["step"] for r in resumed.history] == list(range(6))
    reference = economize.load(make_tiny_model(), tmp_path / "full" / "final")
    for name, value in model.techniques.state_dict().items():
        torch.testing.assert_close(value, reference.techniques.state_dict()[name], rtol=0, atol=0)


# ---- objectives that train the base model ---------------------------------------------------------------------------
def test_ce_trains_the_base_model_and_saves_it(rows: torch.Tensor, tmp_path: Path) -> None:
    base = make_tiny_model()
    model = economize.apply(base)
    before = base.model.layers[0].mlp.up_proj.weight.clone()
    result = fit(model, rows, CrossEntropy(parameters="all"), steps=3, batch_size=4, lr=1e-3, warmup_steps=0,
                 output_dir=tmp_path)
    assert not torch.equal(before, base.model.layers[0].mlp.up_proj.weight)
    final = result.final_path
    assert final is not None and (final / BASE_FILE).is_file() and (final / "base" / "config.json").is_file()
    fresh = economize.apply(make_tiny_model())
    # The fingerprint leaves out the trained parameters (all of them here), so neither the fresh base nor
    # final/base raises a mismatch.
    with warnings.catch_warnings():
        warnings.simplefilter("error", economize.errors.TeacherMismatchWarning)
        load_trained(fresh, final)
        economize.load(AutoModelForCausalLM.from_pretrained(final / "base"), final)
    for (name, a), b in zip(base.named_parameters(), fresh.base_model.parameters()):
        torch.testing.assert_close(a, b, msg=name)
    assert not any(p.requires_grad for p in base.parameters())
    assert "trained_base_parameters" in json.loads((final / "economize_config.json").read_text())


def test_load_trained_checks_the_teacher(rows: torch.Tensor, tmp_path: Path) -> None:
    """Restoring a run onto its own base is silent; onto a base with other frozen weights it warns."""
    fit(_jump_model(), rows, _objective(), steps=1, batch_size=4, output_dir=tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", economize.errors.TeacherMismatchWarning)
        load_trained(_jump_model(), tmp_path / "final")
    other = _jump_model()
    with torch.no_grad():
        other.base_model.model.layers[0].mlp.up_proj.weight.add_(1.0)
    with pytest.warns(economize.errors.TeacherMismatchWarning, match="restored from"):
        load_trained(other, tmp_path / "final")


def test_gradient_accumulation_equals_a_larger_batch(rows: torch.Tensor) -> None:
    weights = []
    for batch_size, accum in ((8, 1), (4, 2), (2, 4)):
        base = make_tiny_model()
        model = economize.apply(base)
        fit(model, TensorSource(rows, batch_size, shuffle=False), CrossEntropy(parameters="all"), steps=2,
            grad_accum=accum, lr=1e-3, warmup_steps=0, grad_clip=0)
        weights.append([p.detach().clone() for p in base.parameters()])
    for other in weights[1:]:
        for a, b in zip(weights[0], other):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=2e-5)


def test_layerskip_fit(rows: torch.Tensor) -> None:
    base = make_tiny_model()
    model = economize.apply(base)
    result = fit(model, rows, LayerSkip(parameters="all", rotation=2), steps=4, batch_size=4, log_every=1,
                 eval_data=rows[:4], warmup_steps=0)
    assert set(result.history[0]) >= {"nll1", "nll3", "nll5", "nll6"} and "nll2" not in result.history[0]
    assert set(result.history[1]) >= {"nll2", "nll4", "nll6"} and "nll1" not in result.history[1]
    assert result.eval is not None and {f"nll{l}" for l in range(1, 7)} <= set(result.eval)
    assert all(not layer._forward_hooks for layer in base.model.layers) and not base.training


# ---- evaluation, teachers, distill_kl -------------------------------------------------------------------------------
def test_fit_evaluates_the_curve(rows: torch.Tensor) -> None:
    result = fit(_jump_model(), rows, _objective(), steps=4, batch_size=4, eval_data=rows[:4], eval_every=2,
                 eval_batch_size=2)
    evals = [r for r in result.history if "eval" in r]
    assert [r["step"] for r in evals] == [1, 3]
    assert result.eval == evals[-1]["eval"]
    assert {"kl", "excess_nll", "agree", "nmse", "cos", "run"} <= set(result.eval)
    assert all(isinstance(v, list) and len(v) == 4 for v in result.eval.values())


def _kl(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    p = teacher.float().log_softmax(-1)
    return (p.exp() * (p - student.float().log_softmax(-1))).sum(-1)


def test_distill_kl_from_a_larger_teacher(toy_heads: Any, rows: torch.Tensor) -> None:
    base = make_tiny_model()
    model = economize.apply(base, toy_heads(heads=1))
    teacher = make_tiny_model(seed=5, num_layers=3)
    data = TensorSource(rows, 4, shuffle=False)
    result = fit(model, data, DistillKL(), steps=1, log_every=1, teacher_model=teacher)
    with torch.no_grad():
        # head 1 at position t drafts token t + 1; the last position has no target
        expected = _kl(teacher(rows[:4]).logits, base(rows[:4]).logits)[:, :-1].mean()
    assert result.history[0]["kl1"] == pytest.approx(expected.item(), rel=1e-5)
    trained = fit(model, rows, "distill_kl", steps=20, batch_size=8, lr=1e-2, warmup_steps=0, teacher_model=teacher)
    assert trained.losses[-1] < trained.losses[0]


def test_fit_with_bfloat16_autocast(rows: torch.Tensor) -> None:
    result = fit(_jump_model(), rows, _objective(), steps=2, batch_size=2, autocast="bf16", eval_data=rows[:2])
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.losses)
    assert result.eval is not None


@dataclass
class _NanAt(JumpRollout):
    """The jump objective, whose loss is ``NaN`` at step ``at``."""

    at: int = 2
    step: int = -1

    def begin_step(self, step: int, total_steps: int) -> None:
        self.step = step

    def loss(self, model: Any, batch: Batch, teacher: TeacherOutputs | None) -> ObjectiveOutput:
        out = super().loss(model, batch, teacher)
        return ObjectiveOutput(out.loss * math.nan, out.metrics) if self.step == self.at else out


def _nan_objective() -> _NanAt:
    return _NanAt(horizon=4, stride=4, eval_horizon=4, eval_stride=6, eval_first=6)


def test_a_nonfinite_loss_stops_the_run_before_the_step_is_applied(rows: torch.Tensor) -> None:
    model = _jump_model()
    with pytest.raises(NonFiniteLossError, match="step 2: the loss is nan"):
        fit(model, rows, _nan_objective(), steps=5, batch_size=2, lr=1e-3, seed=0)
    two_steps = _jump_model()
    fit(two_steps, rows, _objective(), steps=2, batch_size=2, lr=1e-3, seed=0)
    for name, value in _state(two_steps).items():
        torch.testing.assert_close(_state(model)[name], value, msg=name)


def test_a_skipped_nonfinite_step_leaves_the_weights_finite(rows: torch.Tensor,
                                                             caplog: pytest.LogCaptureFixture) -> None:
    model = _jump_model()
    with caplog.at_level(logging.WARNING, "economize.train"):
        result = fit(model, rows, _nan_objective(), steps=5, batch_size=2, lr=1e-3, seed=0, nonfinite="skip")
    assert "step 2: the loss is nan, the step is skipped" in caplog.text
    assert [math.isfinite(loss) for loss in result.losses] == [True, True, False, True, True]
    assert all(torch.isfinite(value).all() for value in _state(model).values())
    warned = _jump_model()
    fit(warned, rows, _nan_objective(), steps=5, batch_size=2, lr=1e-3, seed=0, nonfinite="warn")
    assert not all(torch.isfinite(value).all() for value in _state(warned).values())
    unknown: dict[str, Any] = {"nonfinite": "ignore"}
    with pytest.raises(ValueError, match="nonfinite"):
        TrainConfig(**unknown)


def _half_masked(rows: torch.Tensor) -> Batch:
    """``rows`` whose odd rows have every prediction masked out."""
    loss_mask = torch.ones_like(rows)
    loss_mask[1::2] = 0
    return Batch(rows, loss_mask=loss_mask)


def test_batches_without_a_counted_prediction_are_drawn_past(rows: torch.Tensor) -> None:
    result = fit(_jump_model(), _half_masked(rows), _objective(), steps=6, batch_size=1, lr=1e-3, seed=0,
                 log_every=1)
    assert all(loss > 0 for loss in result.losses)
    assert result.history[-1]["skipped_batches"] > 0
    nothing = Batch(rows, loss_mask=torch.zeros_like(rows))
    with pytest.raises(ConfigError, match="no position that counts"):
        fit(_jump_model(), nothing, _objective(), steps=1, batch_size=1)


def test_evaluation_leaves_out_batches_without_a_counted_prediction(rows: torch.Tensor) -> None:
    model, objective = _jump_model(), _objective()
    counted = Batch(rows[:4])
    empty = Batch(rows[4:8], loss_mask=torch.zeros_like(rows[4:8]))
    assert not empty.has_targets() and counted.has_targets()
    assert evaluate(model, objective, [counted, empty]) == evaluate(model, objective, [counted])
    with pytest.raises(ConfigError, match="none of the 1 evaluation batches"):
        evaluate(model, objective, [empty])


def test_float16_autocast_scales_the_loss_and_checkpoints_the_scale(rows: torch.Tensor, tmp_path: Path) -> None:
    result = fit(_jump_model(), rows, _objective(), steps=2, batch_size=2, autocast="fp16", log_every=1,
                 save_every=1, output_dir=tmp_path)
    assert all(math.isfinite(loss) for loss in result.losses)
    assert all(record["loss_scale"] > 1 for record in result.history if "loss" in record)
    state = torch.load(tmp_path / "checkpoint-1" / STATE_FILE, weights_only=True)
    assert state["scaler"]["scale"] > 1
    resumed = fit(_jump_model(), rows, _objective(), steps=2, batch_size=2, autocast="fp16", log_every=1,
                  save_every=1, output_dir=tmp_path, resume=tmp_path / "checkpoint-1")
    assert resumed.steps == 2


def test_train_config_overrides() -> None:
    config = TrainConfig(steps=10, lr=1e-3)
    assert config.to_dict()["lr"] == 1e-3
    with pytest.raises(ValueError):
        TrainConfig(grad_accum=0)
