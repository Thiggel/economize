"""Qwen2 and Qwen2.5 (``qwen2``)."""

from __future__ import annotations

from ..base import register_adapter
from ..standard import StandardDecoderAdapter

__all__ = ["Qwen2Adapter"]


@register_adapter("qwen2")
class Qwen2Adapter(StandardDecoderAdapter):
    """Qwen2 and Qwen2.5 (sliding layers only with ``use_sliding_window``)."""

    layer_threshold_config_fields = ("max_window_layers",)
    layer_layout = "llama"
