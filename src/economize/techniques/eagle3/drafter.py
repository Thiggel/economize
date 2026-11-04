"""The EAGLE-3 drafter: the one-layer draft decoder's drafter with a fused greedy head over the own head's draft
vocabulary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._draft_layer import DraftLayerDrafter
from .._drafting import FusedGreedyHead, GreedyHead, LogitsGreedyHead, adapter_greedy_head

if TYPE_CHECKING:
    from .technique import Eagle3Technique

__all__ = ["Eagle3Drafter"]


class Eagle3Drafter(DraftLayerDrafter):
    """Drafts with the layer of an :class:`~economize.techniques.eagle3.Eagle3Technique` (see
    :class:`~economize.techniques._draft_layer.DraftLayerDrafter`). Its cost events belong to the module
    ``"eagle3"``."""

    technique: Eagle3Technique

    def greedy_head(self, fused: bool = True) -> GreedyHead:
        """The greedy choice from :meth:`~economize.techniques.eagle3.Eagle3Technique.draft_logits` of a step's
        output: the own head over the draft vocabulary, or the model's (through the fused kernel when ``fused``)."""
        technique = self.technique
        if technique.lm_head is None:
            adapter = technique.adapter
            head = adapter_greedy_head(adapter, before=lambda h: h.to(adapter.dtype)) if fused else None
            return head or LogitsGreedyHead(technique.draft_logits)
        norm = technique.norm
        assert norm is not None
        ids = None if technique._identity_vocab() else technique.draft_ids
        if not fused:
            return LogitsGreedyHead(technique.head_logits, token_ids=ids)
        return FusedGreedyHead(lambda h: norm(h.to(technique.dtype)), technique.lm_head.weight,
                               technique.lm_head.bias, token_ids=ids)
