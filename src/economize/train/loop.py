"""``fit``: a plain training loop for any registered objective, optionally data-parallel on ``accelerate``.

    import economize
    from economize.train import fit, TokenStream

    model = economize.apply(hf_model, economize.JumpConfig(target_layer=16, source_layers=(17,), exit_layer=17))
    result = fit(model, TokenStream("train.bin", seq_len=1024, batch_size=8), steps=2000,
                 output_dir="runs/jump", eval_data=economize.train.fixed_windows("val.bin", 64, 1024))

Each optimizer step accumulates ``grad_accum`` micro-batches: the teacher pass the objective requests, the loss,
``backward``. Then the gradients are clipped, the learning rate of every group follows warmup and the decay
schedule, and AdamW updates the objective's parameter groups. Checkpoints hold the techniques' weights in economize's
format (``economize.load`` reads them), plus the optimizer, the random-number generators, the data position and any
trained base parameters, so that ``resume=True`` continues an interrupted run on the same sequence of batches.
``resume=True`` on a finished run (its final weights exist) loads them and returns the stored result instead of
training again. :class:`~economize.train.checkpoint.RunCheckpointer` lays out the run directory.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from ..validation import check_positive, check_registered
from .checkpoint import RunCheckpointer, latest_checkpoint, load_trained, save_trained
from .data import BatchSource, as_source, shard_source
from .evaluate import MetricMeans, evaluate
from .module import ObjectiveModule
from .objective import Objective, OptimDefaults, build_objective, default_objective
from .optim import build_optimizer, build_scheduler
from .schedule import SCHEDULES
from .teacher import TeacherHead, make_teacher_head
from .utils import autocast_dtype, jsonable, rng_state, seed_everything, set_rng_state

if TYPE_CHECKING:
    from torch import nn

    from ..core.model import EfficientModel

__all__ = ["TrainConfig", "TrainResult", "build_optimizer", "fit", "latest_checkpoint"]

logger = logging.getLogger("economize.train")

LOG_FILE = "train_log.jsonl"


@dataclass
class TrainConfig:
    """Settings of a :func:`fit` run. Optimizer fields left at ``None`` take the objective's
    :class:`~economize.train.OptimDefaults`.

    Attributes:
        steps: Optimizer steps.
        batch_size: Rows per micro-batch (for data that :func:`fit` turns into a source itself).
        seq_len: Tokens per row of a token file passed as ``data``.
        grad_accum: Micro-batches per optimizer step.
        lr: Learning rate of ``"head"`` groups.
        lr_backbone: Learning rate of every ``"backbone"`` group (``None``: each group's own rate).
        weight_decay: Weight decay of groups that set none.
        betas: AdamW betas.
        warmup_steps: Linear warmup steps.
        schedule: Name in :data:`~economize.train.SCHEDULES` (``"cosine"``, ``"linear"``, ``"constant"``).
        min_lr_ratio: Final learning rate as a fraction of the peak.
        grad_clip: Maximum gradient norm (``0``: no clipping).
        autocast: Autocast dtype for the objective's forward (``"bfloat16"``, ``torch.float16``, ...; ``None``:
            none, the model's own dtypes). Teacher passes always run without autocast.
        log_every: Log every this many steps (and the last step).
        eval_every: Evaluate on ``eval_data`` every this many steps (``None``: only after the last step).
        eval_batch_size: Rows per evaluation batch for tensor evaluation data.
        save_every: Write a checkpoint every this many steps (``None``: only the final weights).
        keep_checkpoints: Number of intermediate checkpoints kept (older ones are deleted).
        output_dir: Directory of checkpoints, logs and the final weights (``None``: nothing is written).
        resume: ``True`` continues from the latest checkpoint in ``output_dir`` (if any), a path continues from
            that checkpoint. ``True`` on a finished run (``output_dir/final`` holds the weights after ``steps``
            or more steps) loads the final weights and returns the stored result without training.
        seed: Seeds ``torch``, ``numpy`` and ``random`` at the start, and the data order of data that :func:`fit`
            turns into a source (``None``: no seeding, the current random state is used, and data seed 0).
        accelerate: Run on ``accelerate`` (``accelerate launch`` for several processes): data-parallel gradient
            averaging, one log and checkpoint writer. Each process reads its own batches through
            :func:`~economize.train.data.shard_source`: random sources draw with ``seed + rank``, a
            :class:`~economize.train.LoaderSource` gives every process disjoint items, and a user-defined source needs
            a ``shard(rank, world)`` method (otherwise every process reads the same batches, with a warning).
        offload_optimizer: Keep the AdamW moments in CPU memory (:class:`~economize.train.OffloadedAdamW`), for
            trained parameters that leave no room for them on the device.
        find_unused_parameters: Passed to ``DistributedDataParallel`` with several processes. ``True`` (the safe
            default) allows objectives that leave some trained parameters without gradient in a step (e.g.
            ``layerskip`` with layer dropout); ``False`` saves a graph traversal per step when every trained
            parameter gets a gradient.

    Logged training records hold the loss and metrics averaged over the step's micro-batches and, with several
    processes, over the processes; evaluation splits the batches over the processes and averages over all of them.
    """

    steps: int = 1000
    batch_size: int = 8
    seq_len: int = 1024
    grad_accum: int = 1
    lr: float | None = None
    lr_backbone: float | None = None
    weight_decay: float | None = None
    betas: tuple[float, float] | None = None
    warmup_steps: int | None = None
    schedule: str | None = None
    min_lr_ratio: float | None = None
    grad_clip: float | None = None
    autocast: str | torch.dtype | None = None
    log_every: int = 50
    eval_every: int | None = None
    eval_batch_size: int = 8
    save_every: int | None = None
    keep_checkpoints: int = 2
    output_dir: str | Path | None = None
    resume: bool | str | Path = False
    seed: int | None = None
    accelerate: bool = False
    offload_optimizer: bool = False
    find_unused_parameters: bool = True

    def __post_init__(self) -> None:
        check_positive("steps", self.steps)
        check_positive("grad_accum", self.grad_accum)
        check_positive("batch_size", self.batch_size)
        if self.schedule is not None:
            check_registered("schedule", self.schedule, SCHEDULES)

    def optimizer_settings(self, defaults: OptimDefaults) -> OptimDefaults:
        """The objective's defaults with this run's settings applied."""
        changes = {f.name: getattr(self, f.name) for f in fields(OptimDefaults) if getattr(self, f.name) is not None}
        if changes.get("grad_clip") == 0:
            changes["grad_clip"] = None
        return replace(defaults, **changes)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["autocast"] = None if self.autocast is None else str(self.autocast)
        out["output_dir"] = None if self.output_dir is None else str(self.output_dir)
        out["resume"] = str(self.resume) if isinstance(self.resume, Path) else self.resume
        return out


@dataclass
class TrainResult:
    """What :func:`fit` returns.

    Attributes:
        steps: Optimizer steps completed (including those of a resumed run).
        losses: Loss of every step of this call.
        history: The logged records (training and evaluation).
        eval: Metrics of the last evaluation (``None`` without ``eval_data``).
        output_dir: The run directory.
        final_path: Directory of the final weights (``economize.load`` reads it).
    """

    steps: int
    losses: list[float] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    eval: dict[str, Any] | None = None
    output_dir: Path | None = None
    final_path: Path | None = None


def fit(
    model: EfficientModel,
    data: Any,
    objective: str | Objective | None = None,
    *,
    config: TrainConfig | None = None,
    eval_data: Any = None,
    teacher_model: nn.Module | TeacherHead | None = None,
    callbacks: Iterable[Callable[[dict[str, Any]], None]] = (),
    **overrides: Any,
) -> TrainResult:
    """Train ``model`` with ``objective`` on ``data``.

    Args:
        model: The model with the techniques to train attached (``economize.apply``).
        data: A :class:`~economize.train.BatchSource`, a token file path, a ``[N, T]`` tensor, a list of
            :class:`~economize.train.Example` s or an iterable of batches (see :func:`~economize.train.as_source`).
        objective: An objective or a registered name (``None``: the default objective of the attached drafter,
            e.g. ``eagle3_ttt`` for ``eagle3``).
        config: Run settings; keyword ``overrides`` replace single fields (``fit(..., steps=200, lr=1e-3)``).
        eval_data: Held-out rows ``[N, T]`` or batches, evaluated with the objective's ``evaluate``.
        teacher_model: A larger model with the same tokenizer to distill from (see
            :func:`~economize.train.larger_teacher`), or a ready teacher head.
        callbacks: Called with every logged record.

    Returns:
        The losses, the logged records, the last evaluation and the output paths.
    """
    config = replace(config or TrainConfig(), **overrides)
    objective = default_objective(model) if objective is None else build_objective(objective)
    run = _Run(model, data, objective, config, eval_data, teacher_model, list(callbacks))
    try:
        return run.train()
    finally:
        run.close()


class _Run:
    """State of one :func:`fit` call."""

    def __init__(
        self,
        model: EfficientModel,
        data: Any,
        objective: Objective,
        config: TrainConfig,
        eval_data: Any,
        teacher_model: Any,
        callbacks: list[Callable[[dict[str, Any]], None]],
    ) -> None:
        self.model, self.objective, self.config, self.callbacks = model, objective, config, callbacks
        self.accelerator = None
        rank, self.world = 0, 1
        if config.accelerate:
            from accelerate import Accelerator, DistributedDataParallelKwargs

            ddp = DistributedDataParallelKwargs(find_unused_parameters=config.find_unused_parameters)
            self.accelerator = Accelerator(kwargs_handlers=[ddp])
            rank, self.world = self.accelerator.process_index, self.accelerator.num_processes
            if model.adapter.device != self.accelerator.device:
                model.to(self.accelerator.device)
                model.refresh()
        self.rank = rank
        self.main = rank == 0
        if config.seed is not None:
            seed_everything(config.seed + rank)
        data_seed = 0 if config.seed is None else config.seed
        source = as_source(data, config.batch_size, seq_len=config.seq_len, seed=data_seed)
        self.source: BatchSource = shard_source(source, rank, self.world)  # random sources draw with seed + rank
        self.eval_data = eval_data
        self.teacher_head: TeacherHead | None = make_teacher_head(model, teacher_model)
        self.module = ObjectiveModule(model, objective, self.teacher_head)
        self.module.train()
        self.settings = config.optimizer_settings(objective.defaults)
        self.optimizer = build_optimizer(self.module.groups, self.settings, offload=config.offload_optimizer)
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None  # built in train(), after any resume
        self.forward: Any = self.module
        if self.accelerator is not None and self.accelerator.num_processes > 1:
            self.forward = self.accelerator.prepare_model(self.module)
        self.autocast = autocast_dtype(config.autocast)
        self.output_dir = None if config.output_dir is None else Path(config.output_dir)
        self.checkpoints = RunCheckpointer(self.output_dir, rank=rank, world=self.world,
                                           keep=config.keep_checkpoints, barrier=self._barrier)
        self.step = 0
        self.history: list[dict[str, Any]] = []
        self.losses: list[float] = []
        self.last_eval: dict[str, Any] | None = None
        self.elapsed = 0.0

    # ---- the loop ------------------------------------------------------------------------------------------------
    def train(self) -> TrainResult:
        config = self.config
        if config.resume is True and self.checkpoints.finished(config.steps):
            return self._finished()
        self._maybe_resume()
        if self.output_dir is not None and self.main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            meta = {"objective": self.objective.name, "hyperparameters": jsonable(self.objective.hyperparameters()),
                    "optimizer": jsonable(asdict(self.settings)), "config": config.to_dict()}
            (self.output_dir / "train_config.json").write_text(json.dumps(meta, indent=2) + "\n")
        self.scheduler = build_scheduler(self.optimizer, self.settings, config.steps, start_step=self.step)
        start = time.time() - self.elapsed
        params = self.module.params
        while self.step < config.steps:
            step = self.step
            self.objective.begin_step(step, config.steps)
            self.optimizer.zero_grad(set_to_none=True)
            logged = step % config.log_every == 0 or step + 1 == config.steps
            loss_sum: torch.Tensor | float = 0.0
            metrics = MetricMeans()
            for micro in range(config.grad_accum):
                batch = self.source.next_batch()
                sync = micro == config.grad_accum - 1 or self.forward is self.module
                with nullcontext() if sync else self.forward.no_sync():
                    with torch.autocast(self.model.adapter.device.type, self.autocast,
                                        enabled=self.autocast is not None):
                        out = self.forward(batch.input_ids, batch.attention_mask, batch.loss_mask)
                    loss = out["loss"] / config.grad_accum
                    if self.accelerator is not None:
                        self.accelerator.backward(loss)
                    else:
                        loss.backward()
                loss_sum = loss_sum + loss.detach()
                if logged:
                    metrics.add(out["metrics"])
            if self.settings.grad_clip is not None:
                if self.accelerator is not None:
                    self.accelerator.clip_grad_norm_(params, self.settings.grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(params, self.settings.grad_clip)
            self.optimizer.step()
            lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()
            self.step += 1
            total = self._mean_over_processes(torch.as_tensor(loss_sum, dtype=torch.float32)).item()  # one sync
            self.losses.append(total)
            if not math.isfinite(total):
                logger.warning("step %d: loss is %s", step, total)
            if logged:
                if self.world > 1:
                    metrics.all_reduce(self._sum_over_processes, self.model.adapter.device)
                self._log({"step": step, "loss": total, "lr": lr, **metrics.means(), "time": time.time() - start})
            last = self.step == config.steps
            if self.eval_data is not None and config.eval_every and self.step % config.eval_every == 0 and not last:
                self._evaluate()
            if config.save_every and self.step % config.save_every == 0 and not last:
                self.elapsed = time.time() - start
                self._save_checkpoint()
        self.elapsed = time.time() - start
        final = self._save_final()
        if self.eval_data is not None:
            self._evaluate()
        self.checkpoints.save_result({"steps": self.step, "losses": self.losses, "history": self.history,
                                      "eval": self.last_eval})
        return TrainResult(self.step, self.losses, self.history, self.last_eval, self.output_dir, final)

    def _finished(self) -> TrainResult:
        """The result of a finished run, with its final weights loaded into the model."""
        final = self.checkpoints.final_dir
        assert final is not None
        load_trained(self.model, final, owned=self.module.owned)
        stored = self.checkpoints.load_result()
        steps = int(stored.get("steps", json.loads((final / "economize_config.json").read_text())["train"]["step"]))
        logger.info("%s is a finished run of %d steps: loaded its final weights, nothing to train", self.output_dir,
                    steps)
        return TrainResult(steps, list(stored.get("losses", [])), list(stored.get("history", [])),
                           stored.get("eval"), self.output_dir, final)

    def close(self) -> None:
        self.module.close()

    # ---- logging and evaluation ----------------------------------------------------------------------------------
    def _sum_over_processes(self, value: torch.Tensor) -> torch.Tensor:
        if self.world == 1 or self.accelerator is None:
            return value
        return self.accelerator.reduce(value.to(self.accelerator.device), "sum")

    def _mean_over_processes(self, value: torch.Tensor) -> torch.Tensor:
        return self._sum_over_processes(value) / self.world

    def _log(self, record: dict[str, Any]) -> None:
        record = {k: (float(v) if isinstance(v, torch.Tensor) else v) for k, v in record.items()}
        self.history.append(record)
        if not self.main:
            return
        logger.info(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in record.items()
                             if not isinstance(v, list)))
        if self.output_dir is not None:
            with open(self.output_dir / LOG_FILE, "a") as handle:
                handle.write(json.dumps(record) + "\n")
        for callback in self.callbacks:
            callback(record)

    def _evaluate(self) -> None:
        # the last step's gradients are dead (every step starts from zero) but would stay resident through the
        # evaluation: for a trainable copy of large layers they and the allocator's cached blocks decide whether
        # the teacher pass fits
        self.optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.module.train(False)
        try:
            self.last_eval = evaluate(self.model, self.objective, self.eval_data,
                                      batch_size=self.config.eval_batch_size, teacher_head=self.teacher_head,
                                      autocast=self.autocast, rank=self.rank, world=self.world,
                                      all_reduce=self._sum_over_processes)
        finally:
            self.module.train(True)
        self._log({"step": self.step - 1, "eval": self.last_eval})

    # ---- checkpoints ---------------------------------------------------------------------------------------------
    def _save_weights(self, path: Path, *, save_base: bool = False) -> None:
        meta = {"objective": self.objective.name, "step": self.step,
                "hyperparameters": jsonable(self.objective.hyperparameters())}
        save_trained(self.model, self.module.params, path, meta=meta, save_base=save_base, owned=self.module.owned)

    def _save_checkpoint(self) -> None:
        """``checkpoint-<step>`` (see :meth:`RunCheckpointer.save`)."""
        self.checkpoints.save(
            self.step,
            save_weights=self._save_weights,
            shared_state=lambda: {
                "step": self.step,
                "optimizer": self.optimizer.state_dict(),
                "rng": rng_state(),
                "data": self.source.state_dict(),
                "history": self.history,
                "losses": self.losses,
                "elapsed": self.elapsed,
                "eval": self.last_eval,
            },
            own_state={"rng": rng_state(), "data": self.source.state_dict()} if self.world > 1 else {},
        )

    def _barrier(self) -> None:
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    def _save_final(self) -> Path | None:
        """The final weights, written by the main process; the others wait until they exist."""
        if self.output_dir is None:
            return None
        path = self.output_dir / "final"
        if self.main:
            self._save_weights(path, save_base=True)
        self._barrier()
        return path

    def _maybe_resume(self) -> None:
        path = self.checkpoints.resume_path(self.config.resume)
        if path is None:
            return
        load_trained(self.model, path, owned=self.module.owned)
        state = self.checkpoints.load_state(path)
        self.optimizer.load_state_dict(state["optimizer"])
        self.source.load_state_dict(state["data"])
        set_rng_state(state["rng"])
        self.step = int(state["step"])
        self.history, self.losses = list(state["history"]), list(state["losses"])
        self._rewrite_log()
        self.elapsed, self.last_eval = float(state.get("elapsed", 0.0)), state.get("eval")
        logger.info("resumed from %s at step %d", path, self.step)

    def _rewrite_log(self) -> None:
        """Replace the log file with the records of the restored history, dropping the lines that the interrupted
        run wrote after its last checkpoint (they are logged again as the steps are repeated)."""
        if self.output_dir is None or not self.main or not (self.output_dir / LOG_FILE).exists():
            return
        text = "".join(json.dumps(record) + "\n" for record in self.history)
        tmp = self.output_dir / (LOG_FILE + ".tmp")
        tmp.write_text(text)
        tmp.replace(self.output_dir / LOG_FILE)
