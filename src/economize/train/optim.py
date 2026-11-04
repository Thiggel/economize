"""The optimizer and learning-rate schedule of a training run, shared by :func:`~economize.train.fit` and
:class:`~economize.train.EffTrainer`: AdamW with one parameter group per objective parameter group, and a
``LambdaLR`` over :func:`~economize.train.lr_factor`."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch

from .objective import OptimDefaults, ParamGroup
from .offload import OffloadedAdamW
from .schedule import lr_factor

__all__ = ["build_optimizer", "build_scheduler"]

logger = logging.getLogger("economize.train")


def build_optimizer(
    groups: Sequence[ParamGroup], settings: OptimDefaults, *, offload: bool = False
) -> torch.optim.Optimizer:
    """AdamW over ``groups``: head groups at ``settings.lr``; backbone groups at ``settings.lr_backbone``, else
    their own ``lr``, else ``settings.lr``; each group's ``weight_decay`` unless ``None``. The peak rate of every
    group is stored as ``"initial_lr"``. ``offload`` keeps the optimizer state in CPU memory
    with float32 master weights (:class:`~economize.train.OffloadedAdamW`), which parameters stored in a low
    precision (``copy_dtype="model"``) need."""
    param_groups: list[dict[str, Any]] = []
    for group in groups:
        if group.kind == "backbone":
            lr = settings.lr_backbone if settings.lr_backbone is not None else group.lr
        else:
            lr = None
        lr = settings.lr if lr is None else lr
        decay = settings.weight_decay if group.weight_decay is None else group.weight_decay
        param_groups.append(
            {"params": list(group.params), "lr": lr, "initial_lr": lr, "weight_decay": decay, "name": group.name}
        )
    if offload:
        return OffloadedAdamW(param_groups, betas=settings.betas)
    low = sorted({str(p.dtype) for g in param_groups for p in g["params"] if p.is_floating_point()
                  and p.dtype.itemsize < 4})
    if low:
        logger.warning("training %s parameters without float32 master weights loses small updates to rounding: "
                       "set offload_optimizer=True", "/".join(low))
    return torch.optim.AdamW(param_groups, betas=settings.betas)


def build_scheduler(
    optimizer: torch.optim.Optimizer, settings: OptimDefaults, total_steps: int, *, start_step: int = 0
) -> torch.optim.lr_scheduler.LambdaLR:
    """The learning-rate schedule of ``settings`` (warmup, decay shape, floor) over ``total_steps`` optimizer steps.

    The rate of every group is its ``"initial_lr"`` (see :func:`build_optimizer`) times
    :func:`~economize.train.lr_factor` of the step. Call ``step()`` after every optimizer step.

    Args:
        optimizer: An optimizer from :func:`build_optimizer` (every group has ``"initial_lr"``).
        settings: Warmup steps, schedule name and ``min_lr_ratio``.
        total_steps: Optimizer steps of the run.
        start_step: The step of the next optimizer update (a resumed run's step count).
    """
    warmup, schedule, floor = settings.warmup_steps, settings.schedule, settings.min_lr_ratio
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, total_steps, warmup, schedule, floor), last_epoch=start_step - 1
    )
