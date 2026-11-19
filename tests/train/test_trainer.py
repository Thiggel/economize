"""``EffTrainer``: economize objectives inside ``transformers.Trainer`` (logging, checkpoints, resume, PEFT)."""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

from helpers import VOCAB, make_tiny_model

pytest.importorskip("accelerate")
pytest.importorskip("transformers")

from transformers import TrainerCallback

import economize
from economize.errors import ConfigError, NonFiniteLossError
from economize.techniques.jump import JumpRollout
from economize.train import CrossEntropy, EffCollator, EffTrainer, Example, economize_training_args, fit
from economize.train.checkpoint import BASE_FILE, load_trained
from economize.train.evaluate import MetricMeans, evaluate, mean_metrics
from economize.train.schedule import lr_factor

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning", "ignore::FutureWarning")


def _dataset(rows: int = 32, length: int = 24, seed: int = 3) -> list[dict[str, torch.Tensor]]:
    ids = torch.randint(0, VOCAB, (rows, length), generator=torch.Generator().manual_seed(seed))
    return [{"input_ids": row} for row in ids]


def _jump_model() -> Any:
    torch.manual_seed(0)
    return economize.apply(make_tiny_model(), economize.JumpConfig(target_layer=2, source_layer=4, exit_layer=5))


def _objective() -> JumpRollout:
    return JumpRollout(horizon=4, stride=4)


def _args(objective: Any, output_dir: Path, **overrides: Any) -> Any:
    settings = {"max_steps": 6, "per_device_train_batch_size": 4, "logging_steps": 2, "save_steps": 3,
                "report_to": [], "use_cpu": True, "seed": 0, "warmup_steps": 1, "disable_tqdm": True}
    settings.update(overrides)
    return economize_training_args(objective, str(output_dir), **settings)


def _state(model: Any) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.techniques.state_dict().items()}


# ---- collator and arguments ---------------------------------------------------------------------------------------
def test_collator_pads_and_turns_labels_into_a_loss_mask() -> None:
    collator = EffCollator(pad_token_id=9, max_length=4)
    out = collator([
        {"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, 3, 4, 5]},
        {"input_ids": [6, 7, 0], "attention_mask": [1, 1, 0]},
        Example(torch.tensor([8, 8, 8]), torch.tensor([1, 0, 0])),
    ])
    assert out["input_ids"].tolist() == [[1, 2, 3, 4], [6, 7, 9, 9], [8, 8, 8, 9]]
    assert out["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]]
    assert out["loss_mask"].tolist() == [[0, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]]
    assert set(EffCollator()([{"input_ids": [1, 2]}, {"input_ids": [3, 4]}])) == {"input_ids"}
    with pytest.raises(ValueError, match="right padding"):
        collator([{"input_ids": [0, 1, 2], "attention_mask": [0, 1, 1]}])


def test_training_arguments_follow_the_objective(tmp_path: Path) -> None:
    objective = _objective()
    args = economize_training_args(objective, str(tmp_path), report_to=[])
    assert args.learning_rate == objective.defaults.lr and args.warmup_steps == objective.defaults.warmup_steps
    assert args.max_grad_norm == objective.defaults.grad_clip
    assert args.economize_schedule == objective.defaults.schedule
    assert args.min_lr_ratio == objective.defaults.min_lr_ratio
    assert (args.adam_beta1, args.adam_beta2) == objective.defaults.betas
    args = economize_training_args("ce", str(tmp_path), learning_rate=1e-5, report_to=[])
    assert args.learning_rate == 1e-5
    args = economize_training_args("ce", str(tmp_path), lr_scheduler_type="linear", report_to=[])
    assert args.economize_schedule is None and getattr(args.lr_scheduler_type, "value", None) == "linear"


def test_trainer_follows_the_learning_rates_of_fit(tmp_path: Path) -> None:
    objective = _objective()
    assert objective.defaults.min_lr_ratio > 0  # the floor that the transformers cosine schedule used to drop
    steps, warmup = 6, 2
    result = fit(_jump_model(), torch.stack([row["input_ids"] for row in _dataset()]), _objective(), steps=steps,
                 batch_size=4, seq_len=24, warmup_steps=warmup, log_every=1)
    trainer = EffTrainer(_jump_model(), objective, _args(objective, tmp_path, max_steps=steps, warmup_steps=warmup),
                         train_dataset=_dataset())
    trainer.create_optimizer_and_scheduler(steps)
    lrs = []
    for _ in range(steps):
        lrs.append(trainer.lr_scheduler.get_last_lr()[0])
        trainer.optimizer.step()
        trainer.lr_scheduler.step()
    assert lrs == pytest.approx([record["lr"] for record in result.history if "lr" in record])
    assert lrs[-1] > 0


# ---- training, logging, checkpoints, resume -------------------------------------------------------------------------
def _train_jump(output_dir: Path, resume_from: Path | None = None) -> tuple[Any, EffTrainer]:
    model = _jump_model()
    trainer = EffTrainer(model, _objective(), _args(_objective(), output_dir, learning_rate=3e-3),
                         train_dataset=_dataset())
    trainer.train(resume_from_checkpoint=None if resume_from is None else str(resume_from))
    trainer.close()
    return model, trainer


def test_trainer_trains_logs_and_saves(tmp_path: Path) -> None:
    before = _state(_jump_model())
    model, trainer = _train_jump(tmp_path)
    logged = [r for r in trainer.state.log_history if "loss" in r]
    assert len(logged) == 3 and all({"kl1", "klK", "nmse1"} <= set(r) for r in logged)
    after = _state(model)
    assert any(not torch.equal(before[k], after[k]) for k in before)
    assert not any(p.requires_grad for p in model.base_model.parameters())
    for step in (3, 6):
        checkpoint = tmp_path / f"checkpoint-{step}"
        assert (checkpoint / "economize_config.json").is_file() and not (checkpoint / BASE_FILE).exists()
    loaded = economize.load(make_tiny_model(), tmp_path / "checkpoint-6")
    for name, value in after.items():
        torch.testing.assert_close(loaded.techniques.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.skipif(tuple(int(v) for v in torch.__version__.split(".")[:2]) < (2, 6),
                    reason="transformers refuses torch.load of optimizer states before torch 2.6")
def test_trainer_resumes(tmp_path: Path) -> None:
    """Resuming from step 3 replays steps 4 to 6 exactly: data order, rollout starts, optimizer and schedule."""
    model, _ = _train_jump(tmp_path)
    resumed, _ = _train_jump(tmp_path, resume_from=tmp_path / "checkpoint-3")
    reference = _state(model)
    for name, value in _state(resumed).items():
        torch.testing.assert_close(value, reference[name], rtol=1e-6, atol=1e-7)


def test_trainer_evaluates_with_the_objective(tmp_path: Path) -> None:
    trainer = EffTrainer(_jump_model(), _objective(), _args(_objective(), tmp_path, max_steps=2, save_strategy="no",
                                                            eval_strategy="steps", eval_steps=2),
                         train_dataset=_dataset(), eval_dataset=_dataset(8, seed=4))
    trainer.train()
    metrics = trainer.evaluate()
    trainer.close()
    assert {"eval_loss", "eval_kl1", "eval_nmse1"} <= set(metrics)
    assert any("eval_kl1" in r for r in trainer.state.log_history[:-1])


def test_trainer_with_an_economize_schedule_and_backbone_rate(tmp_path: Path) -> None:
    base = make_tiny_model()
    model = economize.apply(base)
    before = [p.clone() for p in base.parameters()]
    objective = CrossEntropy(parameters="all")
    trainer = EffTrainer(model, objective, _args(objective, tmp_path, max_steps=3, save_steps=3, warmup_steps=2),
                         train_dataset=_dataset(), lr_backbone=2e-4, schedule="cosine", min_lr_ratio=0.1)
    trainer.train()
    trainer.close()
    assert [g["initial_lr"] for g in trainer.optimizer.param_groups] == [2e-4]
    assert trainer.lr_scheduler.lr_lambdas[0](1) == pytest.approx(lr_factor(1, 3, 2, "cosine", 0.1))
    assert any(not torch.equal(a, b) for a, b in zip(before, base.parameters()))
    checkpoint = tmp_path / "checkpoint-3"
    assert (checkpoint / BASE_FILE).is_file() and (checkpoint / "base" / "config.json").is_file()
    fresh = economize.apply(make_tiny_model())
    load_trained(fresh, checkpoint)
    for (name, a), b in zip(base.named_parameters(), fresh.base_model.parameters()):
        torch.testing.assert_close(a, b, msg=name)


def test_trainer_defaults_to_the_drafter_objective(tmp_path: Path) -> None:
    trainer = EffTrainer(_jump_model(), args=_args("jump_rollout", tmp_path, max_steps=1, save_strategy="no"),
                         train_dataset=_dataset(8))
    assert trainer.objective.name == "jump_rollout"
    trainer.train()
    trainer.close()
    assert math.isfinite(trainer.state.log_history[-1]["train_loss"])


# ---- PEFT -------------------------------------------------------------------------------------------------------
class _NanAt(JumpRollout):
    """The jump rollout with a ``NaN`` loss at optimizer step ``at``."""

    def __init__(self, at: int) -> None:
        super().__init__(horizon=4, stride=4)
        self.at, self.step = at, -1

    def begin_step(self, step: int, total_steps: int) -> None:
        super().begin_step(step, total_steps)
        self.step = step

    def loss(self, model: Any, batch: Any, teacher: Any) -> Any:
        out = super().loss(model, batch, teacher)
        return type(out)(out.loss * math.nan, out.metrics) if self.step == self.at else out


def _nan_trainer(tmp_path: Path, **kwargs: Any) -> EffTrainer:
    objective = _NanAt(at=2)
    args = _args(objective, tmp_path, max_steps=4, save_strategy="no", logging_steps=1)
    return EffTrainer(_jump_model(), objective, args, train_dataset=_dataset(), **kwargs)


def test_trainer_stops_on_a_nonfinite_loss_before_the_step_is_applied(tmp_path: Path) -> None:
    trainer = _nan_trainer(tmp_path)
    with pytest.raises(NonFiniteLossError, match="step 2: the loss is nan"):
        trainer.train()
    trainer.close()
    assert trainer.state.global_step == 2
    assert all(torch.isfinite(value).all() for value in _state(trainer.economize_model).values())


def test_trainer_skips_or_applies_a_nonfinite_step_on_request(tmp_path: Path,
                                                              caplog: pytest.LogCaptureFixture) -> None:
    skipped = _nan_trainer(tmp_path / "skip", nonfinite="skip")
    with caplog.at_level(logging.WARNING, "economize.train"):
        skipped.train()
    skipped.close()
    assert "step 2: the loss is nan, the step is skipped" in caplog.text
    assert skipped.state.global_step == 4
    assert all(torch.isfinite(value).all() for value in _state(skipped.economize_model).values())
    objective = _NanAt(at=2)
    args = _args(objective, tmp_path / "warn", max_steps=4, save_strategy="no", nonfinite="warn")
    warned = EffTrainer(_jump_model(), objective, args, train_dataset=_dataset())
    warned.train()
    warned.close()
    assert not all(torch.isfinite(value).all() for value in _state(warned.economize_model).values())
    with pytest.raises(ConfigError, match="nonfinite"):
        _args(objective, tmp_path, nonfinite="ignore")
    with pytest.raises(ConfigError, match="nonfinite"):
        EffTrainer(_jump_model(), objective, args, train_dataset=_dataset(), nonfinite="ignore")  # type: ignore[arg-type]


def test_trainer_counts_batches_without_a_counted_prediction(tmp_path: Path,
                                                            caplog: pytest.LogCaptureFixture) -> None:
    rows = _dataset()
    for row in rows[:4]:  # the first optimizer step sees only masked rows
        row["loss_mask"] = torch.zeros_like(row["input_ids"])
    objective = _objective()
    trainer = EffTrainer(_jump_model(), objective,
                         _args(objective, tmp_path, max_steps=2, save_strategy="no", logging_steps=1,
                               train_sampling_strategy="sequential"),
                         train_dataset=rows)
    before = _state(trainer.economize_model)
    first_step: dict[str, torch.Tensor] = {}
    trainer.add_callback(_SnapshotAfterFirstStep(trainer, first_step))
    with caplog.at_level(logging.WARNING, "economize.train"):
        trainer.train()
    trainer.close()
    assert "every prediction masked" in caplog.text
    assert trainer.skipped_batches == 1
    assert any(entry.get("skipped_batches") == 1 for entry in trainer.state.log_history)
    for name, value in before.items():
        torch.testing.assert_close(first_step[name], value, msg=name)


class _SnapshotAfterFirstStep(TrainerCallback):
    def __init__(self, trainer: EffTrainer, into: dict[str, torch.Tensor]) -> None:
        self.trainer, self.into = trainer, into

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.global_step == 1:
            self.into.update(_state(self.trainer.economize_model))


def test_trainer_gives_up_on_data_without_a_counted_prediction(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    from economize.train import trainer as trainer_module

    monkeypatch.setattr(trainer_module, "MAX_EMPTY_DRAWS", 3)
    rows = [{**row, "loss_mask": torch.zeros_like(row["input_ids"])} for row in _dataset()]
    objective = _objective()
    trainer = EffTrainer(_jump_model(), objective, _args(objective, tmp_path, max_steps=4, save_strategy="no"),
                         train_dataset=rows)
    with pytest.raises(ConfigError, match="3 consecutive training batches"):
        trainer.train()
    trainer.close()


def test_trainer_trains_peft_adapters(tmp_path: Path) -> None:
    peft = pytest.importorskip("peft")
    hf = make_tiny_model()
    policy = peft.get_peft_model(hf, peft.LoraConfig(r=2, target_modules=["q_proj", "v_proj"], init_lora_weights=False))
    lora = {n: p.detach().clone() for n, p in policy.named_parameters() if p.requires_grad}
    frozen = {n: p.detach().clone() for n, p in policy.named_parameters() if not p.requires_grad}
    assert lora
    model = economize.apply(policy)
    objective = CrossEntropy(parameters="trainable")
    trainer = EffTrainer(model, objective, _args(objective, tmp_path, max_steps=3, save_steps=3,
                                                 learning_rate=1e-2), train_dataset=_dataset())
    trainer.train()
    trainer.close()
    params = dict(policy.named_parameters())
    assert all(not torch.equal(params[n], value) for n, value in lora.items())
    assert all(torch.equal(params[n], value) for n, value in frozen.items())
    assert all(params[n].requires_grad for n in lora)

    fresh = peft.get_peft_model(make_tiny_model(), peft.LoraConfig(r=2, target_modules=["q_proj", "v_proj"]))
    # The fingerprint leaves out the trained adapters, so restoring onto fresh adapters raises no mismatch.
    with warnings.catch_warnings():
        warnings.simplefilter("error", economize.errors.TeacherMismatchWarning)
        load_trained(economize.apply(fresh), tmp_path / "checkpoint-3")
    fresh_params = dict(fresh.named_parameters())
    for name in lora:
        torch.testing.assert_close(fresh_params[name], params[name], msg=name)


# ---- parameter groups, metric reduction and optimizer steps ----
def _adapted_jump_model(adaptation: str = "frozen", **kwargs: Any) -> Any:
    torch.manual_seed(0)
    config = economize.JumpConfig(target_layer=2, source_layer=4, adaptation=adaptation, lora_rank=2, **kwargs)
    return economize.apply(make_tiny_model(), config)


def _eval_objective(**kwargs: Any) -> JumpRollout:
    return JumpRollout(**{"horizon": 4, "stride": 4, "eval_horizon": 4, "eval_stride": 6, "eval_first": 6, **kwargs})


def _rows(count: int = 16, length: int = 24) -> torch.Tensor:
    return torch.randint(0, VOCAB, (count, length), generator=torch.Generator().manual_seed(3))


# ---- parameter roles ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("adaptation, roles", [
    ("frozen", {"drafter"}), ("lora", {"drafter", "lora"}), ("full", {"drafter", "blocks"}),
])
def test_jump_parameter_groups_by_role(adaptation: str, roles: set[str]) -> None:
    model = _adapted_jump_model(adaptation, exit_layer=5)
    technique = model.techniques["jump"]
    groups = technique.parameter_groups()
    assert set(groups) == roles
    ids = [id(p) for params in groups.values() for p in params]
    assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in technique.parameters()}
    objective = _eval_objective()
    by_name = {group.name: group for group in objective.parameter_groups(model)}
    assert set(by_name) == roles and by_name["drafter"].kind == "head"
    for role in roles - {"drafter"}:
        assert by_name[role].kind == "backbone"
        assert by_name[role].lr == objective.group_lr[role]
        assert by_name[role].weight_decay == objective.group_weight_decay[role]


def test_metric_means_reduce_over_processes() -> None:
    """Two processes hold different sums; a sum over both gives the mean over all their additions."""
    first, second = MetricMeans(), MetricMeans()
    first.add({"kl": 1.0, "curve": torch.tensor([1.0, 2.0])})
    second.add({"kl": 3.0, "curve": torch.tensor([3.0, 4.0])})
    second.add({"kl": 5.0, "curve": torch.tensor([5.0, 6.0])})
    packed: list[torch.Tensor] = []

    def fake_reduce(value: torch.Tensor) -> torch.Tensor:
        packed.append(value)
        return value + other  # the other process's packed tensor

    other = torch.cat([second.sums["curve"], second.sums["kl"].reshape(1), torch.tensor([2.0, 2.0])])
    first.all_reduce(fake_reduce, "cpu")
    assert len(packed) == 1  # one collective for all metrics
    assert first.means() == {"curve": pytest.approx([3.0, 4.0]), "kl": pytest.approx(3.0)}


def test_evaluate_shards_batches_over_processes() -> None:
    model = _adapted_jump_model()
    objective = _eval_objective()
    rows = _rows(8)
    single = evaluate(model, objective, rows, batch_size=2)
    # each "process" evaluates its half; summing their packed sums and counts reproduces the single-process means
    halves = []
    for rank in range(2):
        packed: dict[str, torch.Tensor] = {}

        def reduce(value: torch.Tensor, packed: dict[str, torch.Tensor] = packed) -> torch.Tensor:
            packed["value"] = value.clone()
            return value

        evaluate(model, objective, rows, batch_size=2, rank=rank, world=2, all_reduce=reduce)
        halves.append(packed["value"])
    total = halves[0] + halves[1]
    names = sorted(single)
    offset, means = 0, {}
    for name in names:
        size = len(single[name]) if isinstance(single[name], list) else 1
        means[name] = total[offset : offset + size]
        offset += size
    counts = total[offset:]
    for i, name in enumerate(names):
        expected = torch.as_tensor(single[name], dtype=torch.float64).reshape(-1)
        torch.testing.assert_close(means[name] / counts[i], expected)
    with pytest.raises(ValueError):
        evaluate(model, objective, rows, batch_size=2, rank=0, world=2)


# ---- EffTrainer ---------------------------------------------------------------------------------------------------
def test_trainer_begins_each_optimizer_step_once(tmp_path: Path) -> None:
    from economize.train import EffTrainer, economize_training_args

    objective = _eval_objective()
    calls: list[tuple[int, int]] = []
    objective.begin_step = lambda step, total: calls.append((step, total))  # type: ignore[method-assign]
    args = economize_training_args(objective, str(tmp_path), report_to=[], use_cpu=True, max_steps=3,
                                per_device_train_batch_size=2, gradient_accumulation_steps=4, save_strategy="no",
                                disable_tqdm=True)
    dataset = [{"input_ids": row} for row in _rows(32)]
    trainer = EffTrainer(_adapted_jump_model(), objective, args, train_dataset=dataset)
    trainer.train()
    trainer.close()
    assert calls == [(0, 3), (1, 3), (2, 3)]


def test_trainer_reports_the_objective_protocol(tmp_path: Path) -> None:
    from economize.train import EffTrainer, economize_training_args

    objective = _eval_objective()
    args = economize_training_args(objective, str(tmp_path), report_to=[], use_cpu=True, max_steps=1,
                                per_device_eval_batch_size=4, save_strategy="no", disable_tqdm=True)
    data = [{"input_ids": row} for row in _rows(8)]
    trainer = EffTrainer(_adapted_jump_model(), objective, args, train_dataset=data, eval_with_objective=True)
    metrics = trainer.evaluate(eval_dataset=data)
    trainer.close()
    reference = evaluate(trainer.economize_model, objective, _rows(8), batch_size=4)
    assert metrics["eval_kl1"] == pytest.approx(reference["kl"][0], rel=1e-4)
    assert metrics["eval_agreeK"] == pytest.approx(reference["agree"][-1], rel=1e-4)
    assert trainer.last_eval is not None and len(trainer.last_eval["kl"]) == objective.eval_horizon


def test_mean_metrics_averages_each_metric_over_the_rows_that_report_it() -> None:
    assert mean_metrics([{"a": 1.0, "b": 4.0}, {"a": 3.0}]) == {"a": 2.0, "b": 4.0}
    assert mean_metrics([]) == {}
