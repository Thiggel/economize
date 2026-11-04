"""From importance scores to the kept units: a fixed count per layer (``"uniform"``) or one global threshold over
all layers (``"adaptive"``, FLAP's ``AL-AM``)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from .criteria import Scores
from .stats import LayerShape

__all__ = ["Selection", "keep_count", "select_adaptive", "select_uniform", "top_k"]


@dataclass
class Selection:
    """The kept units of every pruned layer (sorted 0-based indices, int64).

    Attributes:
        groups: Per layer, the kept key/value groups.
        channels: Per layer, the kept FFN channels.
        hidden: The kept residual channels (``None``: all).
    """

    groups: dict[int, Tensor]
    channels: dict[int, Tensor]
    hidden: Tensor | None = None


def keep_count(total: int, ratio: float, multiple: int = 1) -> int:
    """Units kept out of ``total`` when the fraction ``ratio`` is removed: ``total - int(ratio * total)`` (FLAP's and
    LLM-Pruner's rounding), rounded to the nearest multiple of ``multiple`` within ``[multiple, total]`` and at least
    one."""
    kept = total - int(ratio * total)
    if multiple > 1:
        kept = min(total, max(multiple, multiple * round(kept / multiple)))
    return max(1, kept)


def top_k(scores: Tensor, k: int) -> Tensor:
    """The indices of the ``k`` largest scores in increasing order (ties go to the lower index)."""
    order = torch.sort(scores, descending=True, stable=True).indices[:k]
    return torch.sort(order).values


def select_uniform(
    scores: Scores, shapes: Mapping[int, LayerShape], attn_ratio: float, mlp_ratio: float, round_to: int = 1
) -> Selection:
    """Keep the same number of top-scoring groups and channels in every layer.

    Args:
        scores: The criterion's scores.
        shapes: The unit counts of every scored layer.
        attn_ratio: Fraction of the key/value groups removed.
        mlp_ratio: Fraction of the FFN channels removed.
        round_to: Kept FFN channels are a multiple of this.
    """
    selection = Selection({}, {})
    for i, shape in shapes.items():
        selection.groups[i] = top_k(scores.groups[i], keep_count(shape.kv_heads, attn_ratio))
        selection.channels[i] = top_k(scores.channels[i], keep_count(shape.intermediate, mlp_ratio, round_to))
    return selection


def select_adaptive(scores: Scores, shapes: Mapping[int, LayerShape], sparsity: float) -> Selection:
    """FLAP's global threshold (``AL-AM``): sort all standardized scores, weight each unit by the parameters it holds,
    and keep the units above the score at which the removed parameters are closest to ``sparsity`` of the total.

    FLAP's reference weighs a head of a 4096-wide model by ``512 / 3`` against 1 for a channel, its ``4 * 128`` head
    parameters over the 3 channel parameters per hidden unit; a group of ``g`` query heads of width ``hd`` holds
    ``(2g + 2) * hd`` parameters per hidden unit, so it weighs ``(2g + 2) * hd / 3``. A layer whose units would all
    be removed keeps its best one (the reference would leave it empty).
    """
    layers = list(shapes)
    values, weights, owners = [], [], []
    for n, i in enumerate(layers):
        shape = shapes[i]
        group_weight = (2 * shape.group_size + 2) * shape.head_dim / 3
        values += [scores.groups[i], scores.channels[i]]
        weights += [torch.full((shape.kv_heads,), group_weight, dtype=torch.float64),
                    torch.ones(shape.intermediate, dtype=torch.float64)]
        owners += [torch.full((shape.kv_heads,), 2 * n), torch.full((shape.intermediate,), 2 * n + 1)]
    metric, weight = torch.cat(values).double(), torch.cat(weights)
    order = torch.sort(metric, descending=True, stable=True).indices
    cumulative = torch.cumsum(weight[order], 0)
    target = weight.sum() * (1 - sparsity)
    threshold = metric[order][torch.argmin((cumulative - target).abs())]
    keep, owner = metric > threshold, torch.cat(owners)
    selection = Selection({}, {})
    for n, i in enumerate(layers):
        for kind, store in ((2 * n, selection.groups), (2 * n + 1, selection.channels)):
            local = metric[owner == kind]
            kept = torch.nonzero(keep[owner == kind]).flatten()
            store[i] = kept if kept.numel() else local.argmax().reshape(1)
    return selection
