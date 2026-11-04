"""Mistral (``mistral``)."""

from __future__ import annotations

from ..base import register_adapter
from ..standard import StandardDecoderAdapter

__all__ = ["MistralAdapter"]


@register_adapter("mistral")
class MistralAdapter(StandardDecoderAdapter):
    """Mistral (sliding window on every layer when ``sliding_window`` is set).

    Mistral-3 wrappers (``mistral3``) resolve here or to ``ministral3`` through their text config.
    """

    layer_layout = "llama"
