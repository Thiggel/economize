"""Trained weights on disk, shared by both training loops, and the checkpoint directories of a :func:`fit` run.

A training checkpoint directory holds the techniques in economize's format (``economize_config.json`` and
``economize_model.safetensors``, which ``economize.load`` reads) and, when the objective trained parameters of the base
model (LoRA adapters of a PEFT model, a fine-tuned backbone), those parameters by name in
``base_parameters.safetensors``, and the modules the objective trains outside the model
(:meth:`~economize.train.Objective.modules`, e.g. DuoAttention's gates) in ``objective_state.safetensors``.
:class:`RunCheckpointer` lays out the directory of one
:func:`~economize.train.fit` run::

    output_dir/
        checkpoint-<step>/          the weights, rank_state_<rank>.pt (multi-process runs), trainer_state.pt
        final/                      the final weights (``economize.load`` reads them)
        train_result.pt             the result of the finished run, written last

:class:`~economize.train.EffTrainer` keeps the Hugging Face ``Trainer``'s own checkpoint directories and rotation and
shares only :func:`save_trained` and :func:`load_trained` with :func:`fit`.

Trust boundary: the weights are safetensors, but ``trainer_state.pt``, ``rank_state_<rank>.pt`` and
``train_result.pt`` hold optimizer, random-generator and data-source state as pickles (``torch.load`` with
``weights_only=False``), which can run arbitrary code. Resume only from run directories you wrote yourself;
``economize.load`` of a ``final/`` or ``checkpoint-<step>/`` directory reads the safetensors files alone and is safe.
:func:`save_trained` records the teacher fingerprint without the base parameters the objective trained (listed under
:data:`~economize.core.serialization.TRAINED_ENTRY`), and :func:`load_trained` compares it with the frozen part of the
model it restores the run onto (:func:`economize.core.check_teacher`), so restoring onto a different base model warns.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from ..errors import CheckpointFormatError, ConfigError

if TYPE_CHECKING:
    from ..core.model import EfficientModel

__all__ = [
    "BASE_FILE",
    "OBJECTIVE_FILE",
    "RANK_STATE_FILE",
    "RESULT_FILE",
    "STATE_FILE",
    "RunCheckpointer",
    "latest_checkpoint",
    "load_trained",
    "save_trained",
    "trained_base_state",
]

logger = logging.getLogger("economize.train")

BASE_FILE = "base_parameters.safetensors"
#: States of the modules an objective trains outside the model (:meth:`~economize.train.Objective.modules`).
OBJECTIVE_FILE = "objective_state.safetensors"
#: Shared trainer state of a checkpoint (optimizer, step, history, the random and data state of process 0); its
#: presence marks a complete checkpoint.
STATE_FILE = "trainer_state.pt"
#: Per-process random and data state of a checkpoint of a multi-process run (``{rank}`` = process index): every
#: process continues its own data stream and random numbers after a resume.
RANK_STATE_FILE = "rank_state_{rank}.pt"
#: The result of a finished run (steps, losses, history, last evaluation), written after the final weights.
RESULT_FILE = "train_result.pt"


def trained_base_state(model: EfficientModel, params: Iterable[nn.Parameter]) -> dict[str, torch.Tensor]:
    """CPU copies of the base-model parameters among ``params``, by their names in ``model.base_model``."""
    trained = {id(p) for p in params}
    return {
        name: p.detach().to("cpu", copy=True).contiguous()
        for name, p in model.base_model.named_parameters()
        if id(p) in trained
    }


def save_trained(
    model: EfficientModel,
    params: Iterable[nn.Parameter],
    path: str | Path,
    *,
    meta: Mapping[str, Any] | None = None,
    save_base: bool = False,
    owned: nn.Module | None = None,
) -> Path:
    """Write the techniques of ``model`` and the trained base parameters among ``params`` to ``path``.

    Args:
        model: The trained model.
        params: The parameters the objective trained.
        path: Output directory (created if missing).
        meta: Stored under ``"train"`` in ``economize_config.json``.
        save_base: Also write ``base_model.save_pretrained(path / "base")`` when base parameters were trained (a
            PEFT model writes its adapters only), for use outside economize.
        owned: Modules the objective trains outside the model; their state goes to ``OBJECTIVE_FILE``.
    """
    from safetensors.torch import save_file

    from ..core.serialization import save

    path = Path(path)
    params = list(params)
    save(model, path, extra_meta={"train": dict(meta or {})}, trained=params)
    base = trained_base_state(model, params)
    if base:
        save_file(base, str(path / BASE_FILE))
        if save_base and hasattr(model.base_model, "save_pretrained"):
            cast(Any, model.base_model).save_pretrained(path / "base")  # an HF PreTrainedModel
            _save_tokenizer(getattr(model.base_model, "name_or_path", ""), path / "base")
    state = {} if owned is None else owned.state_dict()
    if state:
        cpu = {k: v.detach().to("cpu", copy=True).contiguous() for k, v in state.items()}
        save_file(cpu, str(path / OBJECTIVE_FILE))
    return path


def _save_tokenizer(source: str, path: Path) -> None:
    """Copy the locally available tokenizer of ``source`` next to the saved base model, so that the folder loads as a
    complete model (e.g. as a draft LM, whose vocabulary is checked against the model's)."""
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(source, local_files_only=True).save_pretrained(path)
    except (OSError, ValueError, ImportError):  # no tokenizer available without downloading: weights only
        pass


def load_trained(model: EfficientModel, path: str | Path, *, owned: nn.Module | None = None) -> None:
    """Load weights written by :func:`save_trained` into the techniques and base parameters of ``model`` (and into
    ``owned``, the modules the objective trains outside the model), then compare ``model``'s frozen base weights with
    the checkpoint's teacher fingerprint.

    Raises:
        ValueError: The checkpoint holds weights of a technique that is not attached, or of a base parameter that
            ``model`` lacks.

    Warns:
        TeacherMismatchWarning: The base weights that were not trained differ from those the run started from.
    """
    from safetensors.torch import load_file

    from ..core.serialization import check_teacher, read_checkpoint

    path = Path(path)
    checkpoint = read_checkpoint(path)
    device = model.adapter.device
    for name, state in checkpoint.states.items():
        if name not in model.techniques:
            raise CheckpointFormatError(f"checkpoint {path} has weights of technique {name!r}, which is not attached")
        model.techniques[name].load_state_dict({k: v.to(device) for k, v in state.items()})
    if (path / BASE_FILE).is_file():
        parameters = dict(model.base_model.named_parameters())
        base = load_file(str(path / BASE_FILE))
        missing = sorted(set(base) - set(parameters))
        if missing:
            raise CheckpointFormatError(f"checkpoint {path} has base parameters the model lacks: {missing[:5]}")
        with torch.no_grad():
            for name, value in base.items():
                parameters[name].copy_(value)
    if owned is not None and (path / OBJECTIVE_FILE).is_file():
        own = owned.state_dict()
        saved = load_file(str(path / OBJECTIVE_FILE))
        owned.load_state_dict({k: v.to(own[k].device) if k in own else v for k, v in saved.items()})
    check_teacher(model, checkpoint.meta, detail=f", restored from {path}")
    model.refresh()


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    """The ``checkpoint-<step>`` directory with the highest step in ``output_dir`` (``None``: there is none)."""
    found = [(step, path) for step, path in _checkpoints(Path(output_dir)) if (path / STATE_FILE).is_file()]
    return max(found)[1] if found else None


class RunCheckpointer:
    """The checkpoints of one :func:`~economize.train.fit` run in ``output_dir`` (see the module docstring).

    Every process of a multi-process run holds one; process 0 writes the weights and the shared state, every process
    its own random and data state.

    Args:
        output_dir: The run directory (``None``: nothing is written or read).
        rank: Index of this process.
        world: Number of processes.
        keep: Number of intermediate checkpoints kept (older ones are deleted).
        barrier: Waits for every process (``None``: a single process, nothing to wait for).
    """

    def __init__(
        self,
        output_dir: str | Path | None,
        *,
        rank: int = 0,
        world: int = 1,
        keep: int = 2,
        barrier: Callable[[], None] | None = None,
    ) -> None:
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.rank, self.world, self.keep = rank, world, keep
        self._barrier = barrier or (lambda: None)

    @property
    def main(self) -> bool:
        """Whether this process writes the weights and the shared state."""
        return self.rank == 0

    @property
    def final_dir(self) -> Path | None:
        return None if self.output_dir is None else self.output_dir / "final"

    # ---- writing -------------------------------------------------------------------------------------------------
    def save(
        self,
        step: int,
        *,
        save_weights: Callable[[Path], None],
        shared_state: Callable[[], Mapping[str, Any]],
        own_state: Mapping[str, Any],
    ) -> Path | None:
        """Write ``checkpoint-<step>``: the weights and the shared state (process 0, the state file last, so that its
        presence marks a complete checkpoint) and, in a multi-process run, every process's ``own_state``; then delete
        the checkpoints beyond the newest ``keep``. Every process must call this.

        Args:
            step: Optimizer steps completed.
            save_weights: Writes the weights into the directory it is given (called on process 0).
            shared_state: The trainer state, e.g. the optimizer, history and process 0's random and data state
                (called on process 0).
            own_state: This process's random and data state.

        Returns:
            The checkpoint directory (``None`` without an ``output_dir``).
        """
        if self.output_dir is None:
            return None
        path = self.output_dir / f"checkpoint-{step}"
        self._barrier()
        if self.main:
            save_weights(path)
        if self.world > 1:  # every process keeps its own data position and random numbers
            self._barrier()
            path.mkdir(parents=True, exist_ok=True)
            save_atomic(dict(own_state), path / RANK_STATE_FILE.format(rank=self.rank))
            self._barrier()
        if self.main:
            save_atomic(dict(shared_state()), path / STATE_FILE)
            self.prune()
        self._barrier()
        return path

    def prune(self) -> None:
        """Delete every ``checkpoint-<step>`` directory but the newest ``keep``."""
        if self.output_dir is None:
            return
        found = sorted(_checkpoints(self.output_dir))
        for _, path in found[: max(len(found) - self.keep, 0)]:
            shutil.rmtree(path, ignore_errors=True)

    def save_result(self, result: Mapping[str, Any]) -> None:
        """Write the result of the finished run (process 0), after the final weights."""
        if self.output_dir is not None and self.main:
            save_atomic(dict(result), self.output_dir / RESULT_FILE)

    # ---- reading -------------------------------------------------------------------------------------------------
    def resume_path(self, resume: bool | str | Path | None) -> Path | None:
        """The checkpoint a run with ``resume`` continues from: the latest in ``output_dir`` for ``True``, the given
        path, or ``None`` (start from scratch).

        Raises:
            ValueError: ``resume=True`` without an ``output_dir``.
        """
        if resume is False or resume is None:
            return None
        if resume is not True:
            return Path(resume)
        if self.output_dir is None:
            raise ConfigError("resume=True needs an output_dir")
        path = latest_checkpoint(self.output_dir)
        if path is None:
            logger.info("no checkpoint in %s, starting from scratch", self.output_dir)
        return path

    def load_state(self, path: str | Path) -> dict[str, Any]:
        """The trainer state of checkpoint ``path``, with this process's own random and data state where the
        checkpoint has one."""
        path = Path(path)
        state = torch.load(path / STATE_FILE, map_location="cpu", weights_only=False)
        own_file = path / RANK_STATE_FILE.format(rank=self.rank)
        if own_file.is_file():
            state.update(torch.load(own_file, map_location="cpu", weights_only=False))
        elif self.world > 1:
            logger.warning("%s has no %s: process %d continues with the data and random state of process 0",
                           path, own_file.name, self.rank)
        return state

    def finished(self, steps: int) -> bool:
        """Whether the final weights of a run of at least ``steps`` steps exist."""
        from ..core.serialization import CONFIG_FILE

        final = self.final_dir
        if final is None or not (final / CONFIG_FILE).is_file():
            return False
        trained = json.loads((final / CONFIG_FILE).read_text()).get("train", {})
        return isinstance(trained.get("step"), int) and trained["step"] >= steps

    def load_result(self) -> dict[str, Any]:
        """The result of the finished run (empty for a run finished by an economize version that wrote none)."""
        if self.output_dir is None or not (self.output_dir / RESULT_FILE).is_file():
            return {}
        return torch.load(self.output_dir / RESULT_FILE, map_location="cpu", weights_only=False)


def save_atomic(obj: Any, path: Path) -> None:
    """``torch.save`` to a temporary file renamed into place, so that an interrupted write leaves no partial file
    (:func:`latest_checkpoint` takes any checkpoint with a state file as complete)."""
    partial = path.with_name(path.name + ".tmp")
    torch.save(obj, partial)
    partial.replace(path)


def _checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    """The ``checkpoint-<step>`` directories in ``output_dir`` with their steps."""
    return [(int(m.group(1)), p) for p in output_dir.glob("checkpoint-*")
            if (m := re.fullmatch(r"checkpoint-(\d+)", p.name))]
