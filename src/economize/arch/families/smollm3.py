"""SmolLM3 (``smollm3``)."""

from __future__ import annotations

from ..base import register_adapter
from ..standard import StandardDecoderAdapter

__all__ = ["SmolLM3Adapter"]


@register_adapter("smollm3")
class SmolLM3Adapter(StandardDecoderAdapter):
    """SmolLM3 (NoPE layers skip the rotary inside the attention)."""
