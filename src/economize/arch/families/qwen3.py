"""Qwen3 (``qwen3``)."""

from __future__ import annotations

from ..base import register_adapter
from ..standard import StandardDecoderAdapter

__all__ = ["Qwen3Adapter"]


@register_adapter("qwen3")
class Qwen3Adapter(StandardDecoderAdapter):
    """Qwen3 (query/key norms inside the attention)."""

    layer_threshold_config_fields = ("max_window_layers",)
    layer_layout = "llama"
