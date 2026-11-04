"""OLMo-3 (``olmo3``)."""

from __future__ import annotations

from ..base import register_adapter
from ..standard import StandardDecoderAdapter

__all__ = ["Olmo3Adapter"]


@register_adapter("olmo3")
class Olmo3Adapter(StandardDecoderAdapter):
    """OLMo-3 (sliding and full layers, one rotary table per layer type)."""
