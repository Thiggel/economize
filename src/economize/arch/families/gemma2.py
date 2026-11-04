"""Gemma-2 (``gemma2``)."""

from __future__ import annotations

from ..base import register_adapter
from ..standard import SoftcappedLogitsMixin, StandardDecoderAdapter

__all__ = ["Gemma2Adapter"]


@register_adapter("gemma2")
class Gemma2Adapter(SoftcappedLogitsMixin, StandardDecoderAdapter):
    """Gemma-2 (logit soft-capping; the attention soft-capping is inside the layer)."""
