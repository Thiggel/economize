"""Teacher-forced rollouts of self-drafters whose draft steps run model layers from the token embedding.

A rollout of horizon ``K`` from start ``s`` reproduces ``K`` draft steps after an exact pass that ended at ``s``:
step ``k`` embeds the true token at ``s + k`` and runs the draft path at that position, attending to the teacher's
keys and values of positions ``<= s`` and to the draft path's own keys and values of positions ``s+1..s+k-1``. All
starts roll out at once (see :mod:`economize.train.rollout`). Used by the training of early-exit and layer-skipping
drafters whose draft path differs from the exact path (LoRA, trainable copies, skipped layers).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

import torch

if TYPE_CHECKING:
    from ...core.model import EfficientModel
    from ...runtime import KVCache, LayerContext

__all__ = ["self_draft_rollout"]

_R = TypeVar("_R")


def self_draft_rollout(
    model: EfficientModel,
    path: Callable[[torch.Tensor, LayerContext], torch.Tensor],
    input_ids: torch.Tensor,
    cache: KVCache,
    starts: torch.Tensor,
    horizon: int,
    step_fn: Callable[[int, torch.Tensor, torch.Tensor], _R],
) -> list[_R]:
    """Roll a self-drafter out for ``horizon`` steps from every position in ``starts`` with the true tokens.

    Args:
        model: The model the drafter is attached to.
        path: ``path(embedding, ctx)`` runs one draft step's layers on ``[B, S, d]`` token embeddings and returns
            the hidden state its head reads.
        input_ids: ``[B, T]`` tokens; ``starts + horizon`` must be below ``T``.
        cache: The teacher's keys and values (slot ``t`` holds position ``t``).
        starts: Start positions ``[S]`` (``-1``: the rollout sees no prefix).
        horizon: Number of draft steps ``K``.
        step_fn: ``step_fn(k, positions, hidden)`` after every step; its results are returned in step order.
    """
    from ...train.rollout import RolloutCache, RolloutContext

    runner, adapter = model.runner, model.adapter
    rollout_cache = RolloutCache(cache)
    rows = input_ids.shape[0]
    outputs: list[_R] = []
    for k in range(1, horizon + 1):
        positions = starts + k
        grid = positions[None].expand(rows, -1)
        ctx = RolloutContext(adapter, rollout_cache, grid, grid, starts=starts, step=k)
        hidden = path(runner.embed(input_ids[:, positions], ctx), ctx)
        outputs.append(step_fn(k, positions, hidden))
    return outputs
