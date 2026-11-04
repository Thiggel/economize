"""Phi-3 / Phi-4 (``phi3``)."""

from __future__ import annotations

from ..base import register_adapter
from ..standard import StandardDecoderAdapter

__all__ = ["Phi3Adapter"]


@register_adapter("phi3")
class Phi3Adapter(StandardDecoderAdapter):
    """Phi-3 / Phi-4 (fused ``qkv_proj`` and ``gate_up_proj``)."""
