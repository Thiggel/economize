"""OLMoE (``olmoe``)."""

from __future__ import annotations

from ..base import register_adapter
from ..standard import StandardDecoderAdapter

__all__ = ["OlmoeAdapter"]


@register_adapter("olmoe")
class OlmoeAdapter(StandardDecoderAdapter):
    """OLMoE."""
