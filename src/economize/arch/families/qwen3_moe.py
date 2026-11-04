"""Qwen3-MoE (``qwen3_moe``)."""

from __future__ import annotations

from ..base import register_adapter
from ..standard import SparseStepMoeMixin, StandardDecoderAdapter

__all__ = ["Qwen3MoeAdapter"]


@register_adapter("qwen3_moe")
class Qwen3MoeAdapter(SparseStepMoeMixin, StandardDecoderAdapter):
    """Qwen3-MoE."""
