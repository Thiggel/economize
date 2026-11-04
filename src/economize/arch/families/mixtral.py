"""Mixtral (``mixtral``)."""

from __future__ import annotations

from ..base import register_adapter
from ..standard import StandardDecoderAdapter

__all__ = ["MixtralAdapter"]


@register_adapter("mixtral")
class MixtralAdapter(StandardDecoderAdapter):
    """Mixtral (sparse MoE MLP)."""
