"""Learning-rate schedules and the optimizer settings a run derives from an objective."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import economize
from economize.errors import UnknownNameError
from economize.techniques.jump import JumpRollout
from economize.train import SCHEDULES, OptimDefaults, ParamGroup, TrainConfig, build_optimizer, fit, lr_factor
from economize.train.optim import build_scheduler
from helpers import VOCAB, make_tiny_model


def test_cosine_is_the_tskip_schedule() -> None:
    steps = 200
    for step in range(steps):
        tskip = min(1, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        assert lr_factor(step, steps, 100, "cosine", 0.1) == pytest.approx(tskip, rel=1e-12)
        continued = min(1, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * step / steps))  # tskip --init_from
        assert lr_factor(step, steps, 50, "cosine", 0.0) == pytest.approx(continued, rel=1e-12, abs=1e-15)


def test_linear_and_constant() -> None:
    assert lr_factor(0, 10, 0, "linear", 0.0) == 1.0
    assert lr_factor(5, 10, 0, "linear", 0.0) == pytest.approx(0.5)
    assert lr_factor(5, 10, 0, "linear", 0.2) == pytest.approx(0.6)
    assert lr_factor(1, 10, 4, "constant", 0.0) == pytest.approx(0.5)
    assert lr_factor(9, 10, 4, "constant", 0.0) == 1.0


def test_trapezoid_is_duo_attentions_schedule() -> None:
    """DuoAttention's ``LambdaLR`` (``duo_attn/train.py``), up to its ramp-up counting step ``s`` as ``s + 1``."""
    steps = 2000
    fifth = steps // 5
    for step in range(steps):
        reference = min(1, max((step + 1) / fifth, 0.1), max((steps - step) / fifth, 0.1))
        ours = lr_factor(step, steps, 0, "trapezoid", 0.1)
        assert ours == pytest.approx(reference, abs=1 / fifth + 1e-12)
        if fifth <= step:
            assert ours == pytest.approx(min(1, max((steps - step) / fifth, 0.1)), rel=1e-12)


def test_schedules_are_registered_by_name() -> None:
    assert {"cosine", "linear", "constant"} <= set(SCHEDULES.names())
    SCHEDULES.register("halving", lambda progress, floor: 0.5)
    try:
        assert lr_factor(3, 10, 0, "halving", 0.0) == 0.5
        assert TrainConfig(schedule="halving").schedule == "halving"
    finally:
        SCHEDULES.unregister("halving")
    with pytest.raises(UnknownNameError):
        TrainConfig(schedule="halving")


def test_run_settings_override_the_objective_defaults() -> None:
    defaults = OptimDefaults(lr=1e-3, warmup_steps=100, grad_clip=1.0)
    settings = TrainConfig(lr=5e-4, warmup_steps=10, grad_clip=0).optimizer_settings(defaults)
    assert (settings.lr, settings.warmup_steps, settings.grad_clip) == (5e-4, 10, None)
    assert settings.betas == defaults.betas and settings.schedule == "cosine"
    with pytest.raises(ValueError):
        TrainConfig(steps=0)


def test_build_optimizer_rates_per_group_kind() -> None:
    head, lora, block, fixed = (nn.Parameter(torch.zeros(2)) for _ in range(4))
    groups = [
        ParamGroup("head", [head]),
        ParamGroup("lora", [lora], "backbone", lr=2e-4, weight_decay=0.0),
        ParamGroup("blocks", [block], "backbone"),
        ParamGroup("fixed", [fixed], "head", lr=1.0),  # a head group's own rate is ignored
    ]
    optimizer = build_optimizer(groups, OptimDefaults(lr=1e-3, weight_decay=0.01))
    rates = {g["name"]: (g["lr"], g["initial_lr"], g["weight_decay"]) for g in optimizer.param_groups}
    assert rates == {"head": (1e-3, 1e-3, 0.01), "lora": (2e-4, 2e-4, 0.0), "blocks": (1e-3, 1e-3, 0.01),
                     "fixed": (1e-3, 1e-3, 0.01)}
    optimizer = build_optimizer(groups, OptimDefaults(lr=1e-3, lr_backbone=3e-5))
    assert [g["lr"] for g in optimizer.param_groups] == [1e-3, 3e-5, 3e-5, 1e-3]
    assert optimizer.defaults["betas"] == (0.9, 0.95)


# ---- the scheduler factory ----
def _adapted_jump_model(adaptation: str = "frozen", **kwargs: Any) -> Any:
    torch.manual_seed(0)
    config = economize.JumpConfig(target_layer=2, source_layer=4, adaptation=adaptation, lora_rank=2, **kwargs)
    return economize.apply(make_tiny_model(), config)


def _eval_objective(**kwargs: Any) -> JumpRollout:
    return JumpRollout(**{"horizon": 4, "stride": 4, "eval_horizon": 4, "eval_stride": 6, "eval_first": 6, **kwargs})


def _rows(count: int = 16, length: int = 24) -> torch.Tensor:
    return torch.randint(0, VOCAB, (count, length), generator=torch.Generator().manual_seed(3))


# ---- schedules and the optimizer factory --------------------------------------------------------------------------
@pytest.fixture
def step_schedule() -> Any:
    SCHEDULES.register("halving", lambda progress, floor: max(floor, 0.5 ** int(progress * 4)))
    yield "halving"
    SCHEDULES.unregister("halving")


@pytest.mark.filterwarnings("ignore::UserWarning", "ignore::FutureWarning")
def test_build_scheduler_follows_lr_factor_and_resumes() -> None:
    param = torch.nn.Parameter(torch.zeros(2))
    settings = OptimDefaults(lr=1.0, warmup_steps=3, schedule="cosine", min_lr_ratio=0.1)
    optimizer = torch.optim.SGD([param], lr=1.0)
    scheduler = build_scheduler(optimizer, settings, 10)
    for step in range(10):
        assert optimizer.param_groups[0]["lr"] == pytest.approx(lr_factor(step, 10, 3, "cosine", 0.1))
        optimizer.step()
        scheduler.step()
    resumed = torch.optim.SGD([param], lr=1.0)
    build_scheduler(resumed, settings, 10, start_step=6)
    assert resumed.param_groups[0]["lr"] == pytest.approx(lr_factor(6, 10, 3, "cosine", 0.1))


@pytest.mark.filterwarnings("ignore::UserWarning", "ignore::FutureWarning")
def test_custom_schedule_in_fit_and_trainer_arguments(step_schedule: str, tmp_path: Path) -> None:
    from economize.train import EffTrainingArguments, economize_training_args

    result = fit(_adapted_jump_model(), _rows(), _eval_objective(), steps=4, batch_size=4, seq_len=24, warmup_steps=0,
                 schedule=step_schedule, min_lr_ratio=0.0, lr=1e-3, log_every=1)
    assert [r["lr"] for r in result.history] == pytest.approx([1e-3, 5e-4, 2.5e-4, 1.25e-4])
    objective = _eval_objective()
    objective.defaults = replace(objective.defaults, schedule=step_schedule)
    pytest.importorskip("accelerate")  # transformers.TrainingArguments needs it
    args = economize_training_args(objective, str(tmp_path), report_to=[], use_cpu=True)  # used to raise KeyError
    assert isinstance(args, EffTrainingArguments)
    assert args.economize_schedule == step_schedule and args.min_lr_ratio == objective.defaults.min_lr_ratio
    with pytest.raises(KeyError):
        economize_training_args(objective, str(tmp_path), report_to=[], use_cpu=True, economize_schedule="nope")


@pytest.mark.filterwarnings("ignore::UserWarning", "ignore::FutureWarning")
def test_trainer_uses_the_shared_factory(tmp_path: Path) -> None:
    pytest.importorskip("accelerate")  # transformers.TrainingArguments needs it
    from economize.train import EffTrainer, economize_training_args

    objective = _eval_objective()
    args = economize_training_args(objective, str(tmp_path), report_to=[], use_cpu=True, max_steps=8,
                                economize_schedule="linear", min_lr_ratio=0.2, warmup_steps=2, lr_backbone=1e-4)
    model = _adapted_jump_model("lora", exit_layer=5)
    trainer = EffTrainer(model, objective, args, train_dataset=[{"input_ids": row} for row in _rows()])
    trainer.create_optimizer_and_scheduler(8)
    groups = {group["name"]: group for group in trainer.optimizer.param_groups}
    # peak rates (the schedule has already scaled "lr" to the first warmup step)
    assert groups["lora"]["initial_lr"] == 1e-4 and groups["drafter"]["initial_lr"] == args.learning_rate
    assert trainer.lr_scheduler.lr_lambdas[0](5) == pytest.approx(lr_factor(5, 8, 2, "linear", 0.2))
    settings = trainer.optimizer_settings(8)
    assert settings.grad_clip == args.max_grad_norm and settings.warmup_steps == 2
    trainer.close()
