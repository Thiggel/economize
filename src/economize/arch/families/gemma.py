"""Gemma (``gemma``)."""

from __future__ import annotations

from ..base import register_adapter
from ..standard import SoftcappedLogitsMixin, StandardDecoderAdapter

__all__ = ["GemmaAdapter"]


@register_adapter("gemma")
class GemmaAdapter(SoftcappedLogitsMixin, StandardDecoderAdapter):
    """Gemma (the embedding scale is inside ``embed_tokens``)."""
