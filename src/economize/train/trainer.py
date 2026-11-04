"""``EffTrainer``: economize objectives inside ``transformers.Trainer``.

    import economize
    from economize.train import EffTrainer, economize_training_args

    model = economize.apply(hf_model, economize.JumpConfig(target_layer=16, source_layers=(17,), exit_layer=17))
    args = economize_training_args("jump_rollout", output_dir="runs/jump", max_steps=2000,
                                per_device_train_batch_size=8, bf16=True)
    trainer = EffTrainer(model, "jump_rollout", args, train_dataset=dataset, processing_class=tokenizer)
    trainer.train()

The ``Trainer`` sees an :class:`~economize.train.ObjectiveModule`: its ``forward`` runs the teacher pass and returns
the objective's loss, so everything else of the ``Trainer`` (gradient accumulation, mixed precision,
``accelerate``/DeepSpeed launch, callbacks, logging integrations, checkpoint rotation) works unchanged. economize
overrides the optimizer (one AdamW group per objective parameter group, so backbone groups get their own rate), the
checkpoint format (economize's, plus the trained base parameters) and logging (the objective's metrics next to the
loss).

PEFT: wrap the Hugging Face model first, then attach techniques, and train the adapters with an objective whose
parameter groups include them (``ce`` or ``layerskip`` with ``parameters="trainable"``)::

    peft_model = peft.get_peft_model(hf_model, peft.LoraConfig(target_modules=["q_proj", "v_proj"]))
    model = economize.apply(peft_model, economize.LayerSkipConfig(...))
    EffTrainer(model, economize.train.OBJECTIVES.get("layerskip")(parameters="trainable"), args, ...).train()
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from ..errors import ConfigError
from ..validation import check_registered
from .checkpoint import load_trained, save_trained
from .data import Example, collate
from .data.masks import loss_mask_from_labels
from .evaluate import MetricMeans, evaluate_batch
from .module import ObjectiveModule
from .objective import Batch, Objective, OptimDefaults, build_objective, default_objective
from .optim import build_optimizer, build_scheduler
from .schedule import SCHEDULES
from .teacher import TeacherHead, make_teacher_head
from .utils import jsonable

if TYPE_CHECKING:
    from torch import nn

    from ..core.model import EfficientModel

__all__ = ["EffCollator", "EffTrainer", "EffTrainingArguments", "economize_training_args"]

#: economize schedules that ``transformers`` implements under its own name (they decay to zero there, ignoring
#: ``min_lr_ratio``); :func:`economize_training_args` uses economize's implementation for every other schedule.
_HF_SCHEDULES = {"cosine": "cosine", "linear": "linear", "constant": "constant_with_warmup"}


@dataclass
class EffTrainingArguments(TrainingArguments):
    """``TrainingArguments`` with the optimizer settings of economize that ``transformers`` has no field for.

    Attributes:
        economize_schedule: A name in :data:`~economize.train.SCHEDULES`. When set, :class:`EffTrainer` builds the
            schedule with :func:`~economize.train.build_scheduler` (warmup from ``warmup_steps``, floor
            ``min_lr_ratio``) and ignores ``lr_scheduler_type``.
        min_lr_ratio: Floor of the economize schedule as a fraction of the peak (``None``: the objective's).
        lr_backbone: Learning rate of every ``"backbone"`` parameter group (``None``: each group's own rate).
        offload_optimizer: Keep the AdamW moments in CPU memory (:class:`~economize.train.OffloadedAdamW`).
    """

    economize_schedule: str | None = field(
        default=None, metadata={"help": "economize schedule (economize.train.SCHEDULES)"}
    )
    min_lr_ratio: float | None = field(default=None, metadata={"help": "floor of the economize schedule"})
    lr_backbone: float | None = field(default=None, metadata={"help": "learning rate of backbone parameter groups"})
    offload_optimizer: bool = field(default=False, metadata={"help": "keep the AdamW moments in CPU memory"})

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.economize_schedule is not None:
            # fail early on unknown names
            check_registered("learning-rate schedule", self.economize_schedule, SCHEDULES)


class EffCollator:
    """Right-pads examples into the tensors :class:`~economize.train.ObjectiveModule` takes.

    Each feature is a mapping with ``input_ids`` and optionally ``loss_mask`` (1 where the next-token prediction
    counts) or ``labels`` (Hugging Face convention, ``-100`` = ignored; turned into a ``loss_mask``), or an
    :class:`~economize.train.Example`. The output has ``input_ids``, plus ``attention_mask`` when rows were padded and
    ``loss_mask`` when some example masks positions.

    Args:
        pad_token_id: Id written into padding positions (they are masked, so any id works).
        max_length: Truncate longer examples to this many tokens.
    """

    def __init__(self, pad_token_id: int = 0, max_length: int | None = None) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, features: Sequence[Mapping[str, Any] | Example]) -> dict[str, torch.Tensor]:
        examples = [self._example(feature) for feature in features]
        batch = collate(examples, self.pad_token_id)
        out = {"input_ids": batch.input_ids}
        if batch.attention_mask is not None:
            out["attention_mask"] = batch.attention_mask
        if batch.loss_mask is not None:
            out["loss_mask"] = batch.loss_mask
        return out

    def _example(self, feature: Mapping[str, Any] | Example) -> Example:
        if isinstance(feature, Example):
            ids, loss = feature.input_ids, feature.loss_mask
        else:
            ids = torch.as_tensor(feature["input_ids"], dtype=torch.long)
            attention = feature.get("attention_mask")
            if attention is not None:
                attention = torch.as_tensor(attention, dtype=torch.bool)
                if not bool(attention[: int(attention.sum())].all()):
                    raise ConfigError("EffCollator supports right padding only")
                ids = ids[: int(attention.sum())]
            loss = feature.get("loss_mask")
            loss = None if loss is None else torch.as_tensor(loss, dtype=torch.long)[: len(ids)]
            labels = feature.get("labels")
            if loss is None and labels is not None:
                loss = loss_mask_from_labels(torch.as_tensor(labels)[: len(ids)])
        if self.max_length is not None:
            ids = ids[: self.max_length]
            loss = None if loss is None else loss[: self.max_length]
        return Example(ids, loss)


def economize_training_args(objective: str | Objective, output_dir: str, **overrides: Any) -> EffTrainingArguments:
    """:class:`EffTrainingArguments` with the optimizer and schedule settings the objective recommends.

    Learning rate, backbone rate, weight decay, AdamW betas, warmup steps and gradient clipping come from the
    objective's :class:`~economize.train.OptimDefaults`. A schedule that ``transformers`` implements (``cosine``,
    ``linear``, ``constant``) is set as its ``lr_scheduler_type`` (cosine and linear decay to zero there; set
    ``economize_schedule=`` or pass ``schedule=`` to :class:`EffTrainer` for economize's schedules with a floor). Any
    other registered schedule is set as ``economize_schedule`` with the objective's ``min_lr_ratio``. ``overrides``
    are further fields and win over the defaults.
    """
    defaults = (objective if isinstance(objective, Objective) else build_objective(objective)).defaults
    settings: dict[str, Any] = {
        "learning_rate": defaults.lr,
        "lr_backbone": defaults.lr_backbone,
        "weight_decay": defaults.weight_decay,
        "adam_beta1": defaults.betas[0],
        "adam_beta2": defaults.betas[1],
        "warmup_steps": defaults.warmup_steps,
        "max_grad_norm": defaults.grad_clip or 0.0,
    }
    if defaults.schedule in _HF_SCHEDULES:
        settings["lr_scheduler_type"] = _HF_SCHEDULES[defaults.schedule]
    else:
        settings.update(economize_schedule=defaults.schedule, min_lr_ratio=defaults.min_lr_ratio)
    settings.update(overrides)
    return EffTrainingArguments(output_dir=output_dir, **settings)


class _BeginStep(TrainerCallback):
    """Calls :meth:`Objective.begin_step` once per optimizer step, as :func:`~economize.train.fit` does."""

    def __init__(self, objective: Objective) -> None:
        self.objective = objective

    def on_step_begin(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        self.objective.begin_step(state.global_step, max(state.max_steps, 1))


class EffTrainer(Trainer):
    """A ``transformers.Trainer`` that trains economize techniques with a registered objective.

    Args:
        model: The model with the techniques to train attached (``economize.apply``; the base model may be a PEFT
            model).
        objective: An objective or a registered name (``None``: the default objective of the attached drafter).
        args: Training arguments (``None``: :func:`economize_training_args` of the objective, in
            ``./economize_trainer``). ``learning_rate`` is the rate of ``"head"`` groups. With
            :class:`EffTrainingArguments` its economize fields apply (the keyword arguments below win over them).
        lr_backbone: Learning rate of every ``"backbone"`` group (``None``: ``args.lr_backbone``, else each
            group's own rate, else ``learning_rate``).
        teacher_model: A larger model with the same tokenizer to distill from, or a ready teacher head (see
            :func:`~economize.train.larger_teacher`).
        schedule: A name in :data:`~economize.train.SCHEDULES` used with ``min_lr_ratio`` and ``args.warmup_steps``
            instead of ``args.lr_scheduler_type`` (``None``: ``args.economize_schedule``, else the ``transformers``
            schedule of ``args``).
        min_lr_ratio: Floor of the economize schedule (``None``: ``args.min_lr_ratio``, else the objective's).
        offload_optimizer: Keep the AdamW moments in CPU memory (``None``: ``args.offload_optimizer``).
        eval_with_objective: Also report the objective's evaluation protocol (:meth:`Objective.evaluate`, e.g.
            the teacher-forced curve of a drafter) on the evaluation batches. Scalar metrics are logged as
            ``eval_<name>``, per-step curves as ``eval_<name>1`` and ``eval_<name>K`` (first and last step), and the
            full averaged result is kept in :attr:`last_eval`. Where a name coincides with a training metric
            (``kl1``, ``nmse1``, ...), the protocol's value is reported. The batches must be long enough for the
            protocol (e.g. ``eval_first + eval_horizon + 2`` tokens for ``step_distill``).
        data_collator: Defaults to :class:`EffCollator` with the tokenizer's pad id.
        **trainer_kwargs: Further ``Trainer`` arguments (``train_dataset``, ``eval_dataset``,
            ``processing_class``, ``callbacks``, ...).

    Training metrics of the objective (e.g. ``kl1``, ``nmse1``) are averaged over the steps since the last log and
    logged next to the loss (averaged over processes); evaluation metrics appear with the ``eval_`` prefix.
    :meth:`Objective.begin_step` runs once per optimizer step (a callback at ``on_step_begin``), as in
    :func:`~economize.train.fit`. Checkpoints are written in economize's format and ``resume_from_checkpoint`` restores
    techniques, trained base parameters, optimizer and scheduler. Several GPUs need a distributed launch
    (``accelerate launch`` / ``torchrun``): ``DataParallel`` replicas would not share the techniques' state, so a
    single process always uses one device.
    """

    #: Set by ``Trainer.__init__`` (``None`` until :meth:`create_optimizer` / :meth:`create_scheduler` build them).
    optimizer: torch.optim.Optimizer | None
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR | None
    #: The objective being trained (built from the constructor's ``objective``).
    objective: Objective

    def __init__(
        self,
        model: EfficientModel,
        objective: str | Objective | None = None,
        args: TrainingArguments | None = None,
        *,
        lr_backbone: float | None = None,
        teacher_model: nn.Module | TeacherHead | None = None,
        schedule: str | None = None,
        min_lr_ratio: float | None = None,
        offload_optimizer: bool | None = None,
        eval_with_objective: bool = False,
        data_collator: Any = None,
        **trainer_kwargs: Any,
    ) -> None:
        built = default_objective(model) if objective is None else build_objective(objective)
        if args is None:
            args = economize_training_args(built, "economize_trainer")
        if data_collator is None:
            tokenizer = trainer_kwargs.get("processing_class")
            pad = getattr(tokenizer, "pad_token_id", None)
            if pad is None:
                pad = getattr(tokenizer, "eos_token_id", None)
            data_collator = EffCollator(0 if pad is None else int(pad))
        if schedule is not None:
            check_registered("learning-rate schedule", schedule, SCHEDULES)  # fail early on unknown names
        self.economize_model = model
        self.objective = built
        self.lr_backbone = lr_backbone if lr_backbone is not None else getattr(args, "lr_backbone", None)
        self.schedule = schedule if schedule is not None else getattr(args, "economize_schedule", None)
        if min_lr_ratio is None:
            min_lr_ratio = getattr(args, "min_lr_ratio", None)
        self.min_lr_ratio = built.defaults.min_lr_ratio if min_lr_ratio is None else min_lr_ratio
        if offload_optimizer is None:
            offload_optimizer = bool(getattr(args, "offload_optimizer", False))
        self.offload_optimizer = offload_optimizer
        self.eval_with_objective = eval_with_objective
        #: Averaged :meth:`Objective.evaluate` metrics of the last evaluation (``eval_with_objective``).
        self.last_eval: dict[str, float | list[float]] | None = None
        self.teacher_head = make_teacher_head(model, teacher_model)
        self.objective_module = ObjectiveModule(model, built, self.teacher_head)
        self._metrics: dict[str, MetricMeans] = {"train": MetricMeans(), "eval": MetricMeans()}
        self._protocol = MetricMeans()
        callbacks = [*(trainer_kwargs.pop("callbacks", None) or []), _BeginStep(built)]
        super().__init__(model=self.objective_module, args=args, data_collator=data_collator, callbacks=callbacks,
                         **trainer_kwargs)
        self.model_accepts_loss_kwargs = False  # the objective's loss is a mean: Trainer divides by accumulation
        # One device per process (class docstring). transformers.TrainingArguments.n_gpu is a read-only property over
        # the private field transformers.TrainingArguments._n_gpu, which the Trainer itself assigns the same way.
        if hasattr(self.args, "_n_gpu"):
            self.args._n_gpu = min(self.args._n_gpu, 1)
        elif self.args.n_gpu > 1:
            raise ConfigError("EffTrainer needs one device per process (launch with accelerate or torchrun)")
        model.refresh()  # the Trainer may have moved the model

    # ---- optimizer and schedule ----------------------------------------------------------------------------------
    def optimizer_settings(self, num_training_steps: int | None = None) -> OptimDefaults:
        """The objective's :class:`~economize.train.OptimDefaults` with the settings of ``args`` and of this trainer
        applied (the counterpart of :meth:`TrainConfig.optimizer_settings <economize.train.TrainConfig>`)."""
        args = self.args
        changes: dict[str, Any] = {
            "lr": args.learning_rate,
            "lr_backbone": self.lr_backbone,
            "weight_decay": args.weight_decay,
            "betas": (args.adam_beta1, args.adam_beta2),
            "min_lr_ratio": self.min_lr_ratio,
            "grad_clip": args.max_grad_norm or None,
        }
        if self.schedule is not None:
            changes["schedule"] = self.schedule
        if num_training_steps is not None:
            changes["warmup_steps"] = args.get_warmup_steps(num_training_steps)
        return replace(self.objective.defaults, **changes)

    def create_optimizer(self, model: nn.Module | None = None) -> torch.optim.Optimizer:
        """AdamW with one group per objective parameter group (see :func:`~economize.train.build_optimizer`)."""
        if self.optimizer is None:
            self.optimizer = build_optimizer(self.objective_module.groups, self.optimizer_settings(),
                                             offload=self.offload_optimizer)
        return self.optimizer

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer | None = None
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """economize's schedule (:func:`~economize.train.build_scheduler`) when one is set, else the ``transformers``
        schedule of ``args``."""
        if self.schedule is None or self.lr_scheduler is not None:
            return super().create_scheduler(num_training_steps, optimizer)
        optimizer = optimizer or self.optimizer
        if optimizer is None:
            raise ConfigError("create_scheduler needs an optimizer: call create_optimizer first")
        self.lr_scheduler = build_scheduler(optimizer, self.optimizer_settings(num_training_steps), num_training_steps)
        return self.lr_scheduler

    # ---- loss and evaluation -------------------------------------------------------------------------------------
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        outputs = model(**inputs)
        self._record("train", outputs["metrics"])
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, None, None]:
        """The training loss of the objective on one evaluation batch, with its metrics logged under the ``eval_``
        prefix, and with ``eval_with_objective`` the metrics of :meth:`Objective.evaluate` on the batch.

        The loss is what ``model(**inputs)`` returns, i.e. the objective's training loss (:meth:`Objective.loss`
        through :class:`~economize.train.ObjectiveModule`) without gradients, including any random choices the
        objective makes during training (e.g. randomly drawn rollout offsets).
        The fixed
        evaluation protocol that :func:`~economize.train.fit` and :func:`~economize.train.evaluate` report (e.g.
        teacher-forced accuracy at every horizon step) is :meth:`Objective.evaluate`, which ``eval_with_objective``
        adds. No logits or labels are returned, so ``compute_metrics`` receives none.
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            outputs = model(**inputs)
        self._record("eval", outputs["metrics"])
        if self.eval_with_objective:
            batch = Batch(inputs["input_ids"], inputs.get("attention_mask"), inputs.get("loss_mask"))
            if batch.loss_mask is None and inputs.get("labels") is not None:
                batch = Batch(batch.input_ids, batch.attention_mask, loss_mask_from_labels(inputs["labels"]))
            module = self.objective_module
            was_training = module.training
            module.train(False)
            try:
                self._protocol.add(evaluate_batch(self.economize_model, self.objective, batch, request=module.request,
                                                  teacher_head=self.teacher_head, autocast=self._autocast_dtype()))
            finally:
                module.train(was_training)
        return outputs["loss"].detach(), None, None

    def _autocast_dtype(self) -> torch.dtype | None:
        return torch.bfloat16 if self.args.bf16 else torch.float16 if self.args.fp16 else None

    def _record(self, split: str, metrics: Mapping[str, Any]) -> None:
        self._metrics[split].add({k: v for k, v in metrics.items() if torch.as_tensor(v).dim() == 0})

    def _drain(self, split: str) -> dict[str, float]:
        totals, self._metrics[split] = self._metrics[split], MetricMeans()
        if self.args.world_size > 1:  # every process logs (the loss is gathered there as well)
            totals.all_reduce(self._sum_over_processes, self.args.device)
        return {name: float(value) for name, value in totals.means().items()}  # type: ignore[arg-type]

    def _drain_protocol(self) -> dict[str, float]:
        """The :meth:`Objective.evaluate` means of the evaluation that just ended, as loggable scalars."""
        if not self._protocol.counts:
            return {}
        totals, self._protocol = self._protocol, MetricMeans()
        if self.args.world_size > 1:
            totals.all_reduce(self._sum_over_processes, self.args.device)
        self.last_eval = totals.means()
        flat: dict[str, float] = {}
        for name, value in self.last_eval.items():
            if isinstance(value, list):
                flat.update({f"{name}1": value[0], f"{name}K": value[-1]})
            else:
                flat[name] = value
        return flat

    def _sum_over_processes(self, value: torch.Tensor) -> torch.Tensor:
        return self.accelerator.reduce(value, "sum")

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if "eval_loss" in logs:
            metrics = {**self._drain("eval"), **self._drain_protocol()}
            logs.update({f"eval_{k}": v for k, v in metrics.items() if f"eval_{k}" not in logs})
        elif "loss" in logs:
            logs.update({k: v for k, v in self._drain("train").items() if k not in logs})
        super().log(logs, start_time)

    # ---- checkpoints ---------------------------------------------------------------------------------------------
    def _save(self, output_dir: str | None = None, state_dict: dict[str, Any] | None = None) -> None:
        """Write the techniques in economize's format, the trained base parameters (and ``base/`` via the base
        model's ``save_pretrained``), the tokenizer and the training arguments."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if output_dir is None:
            raise ConfigError("EffTrainer cannot save: neither output_dir nor args.output_dir is set")
        meta = {
            "objective": self.objective.name,
            "step": self.state.global_step,
            "hyperparameters": jsonable(self.objective.hyperparameters()),
        }
        save_trained(
            self.economize_model, self.objective_module.params, output_dir, meta=meta, save_base=True,
            owned=self.objective_module.owned,
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model: nn.Module | None = None) -> None:
        load_trained(self.economize_model, resume_from_checkpoint, owned=self.objective_module.owned)

    def _load_best_model(self) -> None:
        if self.state.best_model_checkpoint is not None:  # Trainer calls this only once a best checkpoint exists
            load_trained(self.economize_model, self.state.best_model_checkpoint, owned=self.objective_module.owned)

    def close(self) -> None:
        """Undo the objective's setup and restore the model's ``requires_grad`` flags and modes."""
        self.objective_module.close()
