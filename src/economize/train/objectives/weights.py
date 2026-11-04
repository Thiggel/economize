"""Which positions of a batch a loss looks at: rollout starts, and the predictions that count (padding and prompt
masking)."""

from __future__ import annotations

import torch

from ...errors import ConfigError
from ..objective import Batch

__all__ = ["rollout_starts", "target_weights"]


def rollout_starts(
    length: int, end: int, stride: int, *, first: int | None = None, device: torch.device | str | None = None
) -> torch.Tensor:
    """Rollout starts ``first, first + stride, ...`` below ``end`` for sequences of ``length`` tokens.

    With ``first=None`` (training) the first start is drawn uniformly from ``[0, min(stride, end))``, so that every
    batch gets at least one start; this draws one number from the global generator even for ``stride == 1``.
    Raises a :class:`~economize.errors.ConfigError` naming the sequence length when no start fits (``torch.arange``
    would raise an
    opaque error or return an empty tensor, whose loss is ``NaN``).
    """
    lowest = 0 if first is None else first
    if end <= lowest:
        raise ConfigError(
            f"sequences of {length} tokens are too short for a rollout start at {lowest} (starts must lie below "
            f"{end}); use longer sequences or a shorter horizon"
        )
    if first is None:
        first = int(torch.randint(0, min(stride, end), ()).item())
    return torch.arange(first, end, stride, device=device)


def target_weights(batch: Batch, targets: torch.Tensor | slice) -> torch.Tensor | None:
    """Weights ``[B, ...]`` of the predictions whose target tokens sit at ``targets`` (an index into the sequence).

    A prediction counts when its target is a real token (``attention_mask``) and the position before the target
    is marked in ``loss_mask`` (chat data marks the positions whose next token is an assistant token). Returns
    ``None`` when the batch has neither mask, so that losses reduce with a plain mean.
    """
    weights = None
    if batch.attention_mask is not None:
        weights = batch.attention_mask[:, targets].bool()
    if batch.loss_mask is not None:
        previous = targets - 1 if isinstance(targets, torch.Tensor) else _shift(targets)
        counted = batch.loss_mask[:, previous].bool()
        weights = counted if weights is None else weights & counted
    return weights


def _shift(index: slice) -> slice:
    if index.step not in (None, 1) or index.start is None or index.start < 1:
        raise ConfigError("target slices must be contiguous and start at 1 or later")
    if index.stop == 0:
        raise ConfigError("empty target slice")
    return slice(index.start - 1, -1 if index.stop is None else index.stop - 1)
