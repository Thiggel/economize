"""Ministral-8B (``ministral``)."""

from __future__ import annotations

from ..base import register_adapter
from ..standard import StandardDecoderAdapter

__all__ = ["MinistralAdapter"]


@register_adapter("ministral")
class MinistralAdapter(StandardDecoderAdapter):
    """Ministral-8B (``layer_types`` mixes sliding and full layers)."""
