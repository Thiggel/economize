"""Minitron's depth criterion, registered as the ``depth_prune`` method ``"minitron"``.

Sreenivas et al. ("LLM Pruning and Distillation in Practice: The Minitron Approach", 2024, Section 2.1) remove one
contiguous block of layers: they score every window of the wanted length by the accuracy of the model without it
(Winogrande in the paper) and drop layers 16-31 of Llama-3.1-8B. This criterion scores every window by the
calibration negative log-likelihood instead, the quantity the method's distillation then reduces, so that it needs
no task data. Windows are visited in depth order and share the hidden states of the layers before them, so the
search costs ``O(L^2)`` layer evaluations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ...core.calibration import token_batches
from ...errors import ConfigError
from ..prune.depth import DEPTH_CRITERIA, DepthCriterion, _LayerRunner, _nll

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["MinitronDepth", "contiguous_windows", "minitron_depth_select"]


def contiguous_windows(candidates: Sequence[int], num_blocks: int) -> list[list[int]]:
    """Every run of ``num_blocks`` consecutive layers that lies inside ``candidates``, by first layer."""
    allowed = set(candidates)
    starts = sorted(allowed)
    return [list(range(s, s + num_blocks)) for s in starts if all(s + k in allowed for k in range(num_blocks))]


@torch.no_grad()
def minitron_depth_select(
    adapter: ArchAdapter, batches: Sequence[Tensor], num_blocks: int, candidates: Sequence[int] | None = None
) -> list[int]:
    """The window of ``num_blocks`` consecutive candidate layers whose removal gives the lowest calibration NLL.

    Args:
        adapter: The model.
        batches: Calibration token batches.
        num_blocks: Length of the removed window.
        candidates: Removable 1-indexed layers (default: every layer).

    Returns:
        The 1-indexed layers of the window, sorted (ties go to the earliest window).

    Raises:
        ConfigError: No window of ``num_blocks`` consecutive layers lies inside ``candidates``.
    """
    layers = list(range(1, adapter.num_layers + 1))
    windows = contiguous_windows(layers if candidates is None else candidates, num_blocks)
    if not windows:
        raise ConfigError(f"no {num_blocks} consecutive layers among the candidates {list(candidates or layers)}")
    batches = token_batches(batches)
    runner = _LayerRunner(adapter, batches)
    prefix, done = runner.hidden, 0  # ``prefix``: the states after layers 1..done
    best, best_loss = windows[0], float("inf")
    for window in windows:
        for layer in range(done + 1, window[0]):
            prefix = runner.run(layer, prefix)
        done = window[0] - 1
        states = prefix
        for later in range(window[-1] + 1, adapter.num_layers + 1):
            states = runner.run(later, states)
        loss = sum(_nll(adapter, x, b) for x, b in zip(states, batches))
        if loss < best_loss:
            best, best_loss = window, loss
    return best


@DEPTH_CRITERIA.register("minitron")
class MinitronDepth(DepthCriterion):
    """One contiguous window of layers, the one whose removal costs the least calibration NLL."""

    def select(
        self, adapter: ArchAdapter, batches: Sequence[Tensor], num_blocks: int, candidates: Sequence[int] | None = None
    ) -> list[int]:
        return minitron_depth_select(adapter, batches, num_blocks, candidates)
