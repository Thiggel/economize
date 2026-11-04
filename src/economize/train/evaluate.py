"""Evaluation of an objective over held-out batches (e.g. the teacher-forced curve of a drafter)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

import torch

from ..errors import ConfigError
from .data import as_batch
from .objective import Batch, Objective, TeacherRequest
from .teacher import TeacherHead, teacher_pass

if TYPE_CHECKING:
    from ..core.model import EfficientModel

__all__ = ["MetricMeans", "evaluate", "evaluate_batch", "mean_metrics", "split_batches"]

#: Sums a tensor over all processes of a data-parallel run (e.g. ``lambda t: accelerator.reduce(t, "sum")``).
AllReduce = Callable[[torch.Tensor], torch.Tensor]


def split_batches(data: torch.Tensor | Batch | Iterable[Any], batch_size: int) -> list[Batch]:
    """Evaluation batches from a ``[N, T]`` tensor (consecutive rows, in order), a :class:`Batch` or an iterable
    of batches."""
    if isinstance(data, Batch):
        data = data.input_ids if data.attention_mask is None and data.loss_mask is None else [data]
    if isinstance(data, torch.Tensor):
        return [Batch(data[i : i + batch_size]) for i in range(0, data.shape[0], batch_size)]
    return [as_batch(item) for item in data]


class MetricMeans:
    """Running means of metric dictionaries: every :meth:`add` counts once for each metric it reports.

    Values may be scalars or vectors (one value per draft step, averaged elementwise). The sums stay tensors until
    :meth:`means`, so adding does not synchronize with the device.
    """

    def __init__(self) -> None:
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}

    def add(self, metrics: Mapping[str, Any]) -> None:
        for name, value in metrics.items():
            value = torch.as_tensor(value).detach().to(torch.float64)
            self.sums[name] = self.sums[name] + value if name in self.sums else value
            self.counts[name] = self.counts.get(name, 0) + 1

    def all_reduce(self, reduce: AllReduce, device: torch.device | str) -> None:
        """Sum the sums and counts over the processes of a data-parallel run (every process must hold the same
        metric names with the same shapes)."""
        names = sorted(self.sums)
        if not names:
            return
        parts = [self.sums[name].to(device).reshape(-1) for name in names]
        counts = torch.tensor([float(self.counts[name]) for name in names], dtype=torch.float64, device=device)
        flat = reduce(torch.cat([*parts, counts]))
        offset = 0
        for name, part in zip(names, parts):
            self.sums[name] = flat[offset : offset + part.numel()].reshape(self.sums[name].shape)
            offset += part.numel()
        for name, count in zip(names, flat[offset:].tolist()):
            self.counts[name] = round(count)

    def means(self) -> dict[str, float | list[float]]:
        """The mean of every metric (floats, or lists for vector metrics)."""
        return {name: _plain(total.cpu() / self.counts[name]) for name, total in self.sums.items()}


@torch.no_grad()
def evaluate_batch(
    model: EfficientModel,
    objective: Objective,
    batch: Batch,
    *,
    request: TeacherRequest | None = None,
    teacher_head: TeacherHead | None = None,
    autocast: torch.dtype | None = None,
) -> Mapping[str, Any]:
    """:meth:`Objective.evaluate` of one batch: the teacher pass ``request`` asks for (default: the objective's;
    without autocast, in the model's own dtypes), then the objective's metrics under ``autocast``."""
    batch = batch.to(model.adapter.device)
    request = objective.teacher_request(model) if request is None else request
    teacher = None
    if request is not None:
        with torch.autocast(model.adapter.device.type, enabled=False):
            teacher = teacher_pass(model, batch, request, teacher_head)
    with torch.autocast(model.adapter.device.type, autocast, enabled=autocast is not None):
        return objective.evaluate(model, batch, teacher)


@torch.no_grad()
def evaluate(
    model: EfficientModel,
    objective: Objective,
    data: torch.Tensor | Batch | Iterable[Any],
    *,
    batch_size: int = 8,
    teacher_head: TeacherHead | None = None,
    autocast: torch.dtype | None = None,
    rank: int = 0,
    world: int = 1,
    all_reduce: AllReduce | None = None,
) -> dict[str, float | list[float]]:
    """The objective's evaluation metrics, averaged over batches (each batch counts equally).

    Vector metrics (one value per draft step) are averaged elementwise and returned as lists. The model's
    techniques are put in eval mode for the call and restored afterwards.

    Args:
        model: The model.
        objective: The objective whose :meth:`~economize.train.Objective.evaluate` is averaged.
        data: Held-out token rows ``[N, T]`` (cut into consecutive batches of ``batch_size``), or batches.
        batch_size: Rows per batch for tensor data.
        teacher_head: Replaces the teacher outputs (a larger target model).
        autocast: Autocast dtype of the objective's computations (``None``: none).
        rank: This process's index in a data-parallel run: it evaluates batches ``rank, rank + world, ...``.
        world: Number of processes. With fewer batches than processes every process evaluates all of them.
        all_reduce: Sums a tensor over the processes (required when ``world > 1``); every process then returns
            the means over all batches.
    """
    batches = split_batches(data, batch_size)
    if not batches:
        raise ConfigError("no evaluation data")
    sharded = world > 1 and len(batches) >= world
    if sharded:
        if all_reduce is None:
            raise ConfigError("evaluate with world > 1 needs all_reduce to combine the processes' metrics")
        batches = batches[rank::world]
    request = objective.teacher_request(model)
    was_training = model.techniques.training
    model.techniques.eval()
    totals = MetricMeans()
    try:
        for batch in batches:
            totals.add(evaluate_batch(model, objective, batch, request=request, teacher_head=teacher_head,
                                      autocast=autocast))
    finally:
        model.techniques.train(was_training)
    if sharded:
        assert all_reduce is not None
        totals.all_reduce(all_reduce, model.adapter.device)
    return totals.means()


def _plain(value: torch.Tensor) -> float | list[float]:
    return value.item() if value.dim() == 0 else value.tolist()


def mean_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Mean of every metric over the scalar metric dictionaries that report it (used for gradient-accumulation
    micro-batches)."""
    totals = MetricMeans()
    for row in rows:
        totals.add(row)
    return {name: float(value) for name, value in totals.means().items()}  # type: ignore[arg-type]
