"""Learning-rate schedules: linear warmup from zero, then a registered decay shape.

The factor of step ``s`` (0-based) of ``S`` steps is ``min(1, (s + 1) / warmup) * decay(s / S, min_lr_ratio)``,
evaluated before the step's update. ``tests/train/test_schedules.py`` pins the cosine shape with ``warmup = 100``
and ``min_lr_ratio = 0.1`` (a training run from scratch) and with ``warmup = 50`` and ``min_lr_ratio = 0`` (continued
training), and the ``"trapezoid"`` shape of DuoAttention's gate training.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from ..registry import Registry

__all__ = ["SCHEDULES", "lr_factor"]

#: ``decay(progress, min_lr_ratio)`` with ``progress = step / total_steps`` in ``[0, 1)``, by name.
SCHEDULES: Registry[Callable[[float, float], float]] = Registry("learning-rate schedule", "economize.schedules")


@SCHEDULES.register("cosine")
def cosine(progress: float, min_lr_ratio: float) -> float:
    """Half a cosine from 1 to ``min_lr_ratio``."""
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


@SCHEDULES.register("linear")
def linear(progress: float, min_lr_ratio: float) -> float:
    """A straight line from 1 to ``min_lr_ratio``."""
    return min_lr_ratio + (1 - min_lr_ratio) * (1 - progress)


@SCHEDULES.register("constant")
def constant(progress: float, min_lr_ratio: float) -> float:
    """No decay."""
    return 1.0


@SCHEDULES.register("trapezoid")
def trapezoid(progress: float, min_lr_ratio: float) -> float:
    """A linear ramp over the first fifth of training, a plateau at 1 and a linear ramp down over the last fifth,
    never below ``min_lr_ratio``: ``max(min_lr_ratio, min(1, 5 p, 5 (1 - p)))``.

    This is DuoAttention's gate-training schedule (``duo_attn/train.py``: ``LambdaLR`` with
    ``min(1, max((s + 1) / (S // 5), 0.1), max((S - s) / (S // 5), 0.1))`` and floor 0.1). The reference's ramp-up
    counts step ``s`` as ``s + 1``; here it starts at ``5 s / S``, one step later (at most ``1 / (S // 5)`` of the
    peak, 0.25% for ``S = 2000``). Use it with ``warmup_steps = 0``.
    """
    return max(min_lr_ratio, min(1.0, 5 * progress, 5 * (1 - progress)))


def lr_factor(step: int, total_steps: int, warmup_steps: int, schedule: str, min_lr_ratio: float) -> float:
    """Multiplier of the peak learning rate at ``step`` (0-based) of ``total_steps``."""
    warmup = min(1.0, (step + 1) / warmup_steps) if warmup_steps > 0 else 1.0
    return warmup * SCHEDULES.get(schedule)(step / max(total_steps, 1), min_lr_ratio)
