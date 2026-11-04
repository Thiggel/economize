"""Ministral-3 and Mistral-3 text decoders (``ministral3``)."""

from __future__ import annotations

from ..base import register_adapter
from ..standard import StandardDecoderAdapter

__all__ = ["Ministral3Adapter"]


@register_adapter("ministral3")
class Ministral3Adapter(StandardDecoderAdapter):
    """Ministral-3 and Mistral-3 text decoders.

    The Llama-4 query scaling inside the attention reads ``position_ids``, which economize passes per row.
    """
