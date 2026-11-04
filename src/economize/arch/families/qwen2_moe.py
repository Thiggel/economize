"""Qwen1.5-MoE and Qwen2-MoE (``qwen2_moe``)."""

from __future__ import annotations

from ..base import register_adapter
from ..standard import SparseStepMoeMixin, StandardDecoderAdapter

__all__ = ["Qwen2MoeAdapter"]


@register_adapter("qwen2_moe")
class Qwen2MoeAdapter(SparseStepMoeMixin, StandardDecoderAdapter):
    """Qwen1.5-MoE and Qwen2-MoE (shared expert inside the MLP)."""

    layer_threshold_config_fields = ("max_window_layers",)
