"""IBM Granite (``granite``, ``granitemoe``, ``granitemoeshared``): muP multipliers."""

from __future__ import annotations

import torch

from ..base import register_adapter
from ..spec import PassContext
from ..standard import StandardDecoderAdapter

__all__ = ["GraniteAdapter"]


@register_adapter("granite", "granitemoe", "granitemoeshared")
class GraniteAdapter(StandardDecoderAdapter):
    """Granite scales the embedding by ``embedding_multiplier`` in the model forward and divides the logits by
    ``logits_scaling``. The residual and attention multipliers live inside the layers."""

    def embed(self, input_ids: torch.Tensor, ctx: PassContext | None = None) -> torch.Tensor:
        return super().embed(input_ids, ctx) * self.config.embedding_multiplier

    def transform_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return super().transform_logits(logits) / self.config.logits_scaling
