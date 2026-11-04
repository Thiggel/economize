"""Llama 2 / 3 / 3.x (``llama``)."""

from __future__ import annotations

from ..base import register_adapter
from ..standard import StandardDecoderAdapter

__all__ = ["LlamaAdapter"]


@register_adapter("llama")
class LlamaAdapter(StandardDecoderAdapter):
    """Llama 2 / 3 / 3.x."""

    layer_layout = "llama"
