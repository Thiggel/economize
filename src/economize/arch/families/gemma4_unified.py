"""Gemma-4 "unified" text decoders and their encoder-free multimodal wrapper (``gemma4_unified_text``,
``gemma4_unified``; ``google/gemma-4-12B-it`` ships as ``Gemma4UnifiedForConditionalGeneration``).

The text decoder is the Gemma-4 decoder without per-layer embeddings and without the MoE block: K/V sharing of
the last layers, K = V full-attention layers with a wider head, one rotary table per layer type, final logit
soft-capping and the per-layer output scalar all work as in :mod:`economize.arch.families.gemma4`. Its decoder layer
takes no per-layer input argument, so layers receive keyword arguments only. The wrapper keeps the decoder at
``model.model.language_model`` and embeds images and audio without an encoder tower; text-only use never touches
those embedders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..base import register_adapter
from ..protocols import LayerInputs
from ..spec import PassContext
from .gemma4 import Gemma4Adapter

if TYPE_CHECKING:
    from ...attention.banded import StructuredMask

__all__ = ["Gemma4UnifiedAdapter"]


@register_adapter("gemma4_unified_text", "gemma4_unified")
class Gemma4UnifiedAdapter(Gemma4Adapter):
    """Gemma-4 unified (see the module docstring)."""

    @property
    def uses_per_layer_inputs(self) -> bool:
        return False

    def layer_inputs(
        self,
        index: int,
        hidden: torch.Tensor,
        ctx: PassContext,
        cache: object,
        mask: torch.Tensor | StructuredMask | None,
    ) -> LayerInputs:
        _, kwargs = super().layer_inputs(index, hidden, ctx, cache, mask)
        return (), kwargs
