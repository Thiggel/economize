"""DeepSeek-V3 and Kimi-K2 (``deepseek_v3``)."""

from __future__ import annotations

from ..base import register_adapter
from ..standard import StandardDecoderAdapter

__all__ = ["DeepseekV3Adapter"]


@register_adapter("deepseek_v3")
class DeepseekV3Adapter(StandardDecoderAdapter):
    """DeepSeek-V3 multi-head latent attention: keys and values of different widths go through ``update`` and get
    buffers of their own shapes."""

    layer_threshold_config_fields = ("first_k_dense_replace",)
