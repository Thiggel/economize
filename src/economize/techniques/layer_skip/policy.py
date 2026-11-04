"""The layer policy of a skip set."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from ...runtime.policy import Decision, LayerAction, LayerPolicy

if TYPE_CHECKING:
    from ...runtime import LayerContext

__all__ = ["SkipLayers"]


class SkipLayers(LayerPolicy):
    """Skip a fixed set of layers for every token, writing no K/V there.

    On the draft path the skipped layers' cache slots of the draft positions stay stale; no draft step reads them
    (draft steps skip those layers too) and the next exact pass overwrites them before any exact token attends.
    """

    def __init__(self, layers: Iterable[int]) -> None:
        self.layers = frozenset(int(layer) for layer in layers)

    def decide(self, index: int, hidden: torch.Tensor, ctx: LayerContext) -> Decision:
        return LayerAction.SKIP if index in self.layers else LayerAction.RUN

    def __repr__(self) -> str:
        return f"SkipLayers({sorted(self.layers)})"
