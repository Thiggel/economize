"""The EAGLE-1/2 drafter: the one-layer draft decoder's drafter with the model's LM head (without its final norm)
as fused greedy head."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._draft_layer import DraftLayerDrafter
from .._drafting import GreedyHead, LogitsGreedyHead, adapter_greedy_head

if TYPE_CHECKING:
    from .technique import EagleTechnique

__all__ = ["EagleDrafter"]


class EagleDrafter(DraftLayerDrafter):
    """Drafts with the layer of an :class:`~economize.techniques.eagle.EagleTechnique` (see
    :class:`~economize.techniques._draft_layer.DraftLayerDrafter`). Its cost events belong to the module
    ``"eagle"``."""

    technique: EagleTechnique

    def greedy_head(self, fused: bool = True) -> GreedyHead:
        """The greedy choice from :meth:`~economize.techniques.eagle.EagleTechnique.head_logits` of a step's output,
        through the fused kernel when ``fused`` and the model's head allows it."""
        adapter = self.technique.adapter
        head = adapter_greedy_head(adapter, before=lambda h: h.to(adapter.dtype), final_norm=False) if fused else None
        return head or LogitsGreedyHead(self.technique.draft_logits)
