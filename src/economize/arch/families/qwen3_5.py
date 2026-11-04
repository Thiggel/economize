"""Qwen3.5 dense and MoE, text and multimodal (``qwen3_5``, ``qwen3_5_text``, ``qwen3_5_moe``,
``qwen3_5_moe_text``) and Qwen3-Next (``qwen3_next``).

Three of every four layers are gated delta nets (linear attention) that keep a conv window and a recurrent state;
the rest are full attention layers. The model therefore uses a :class:`~economize.runtime.HybridCache`, drafts are
verified as chains, and rejected tokens are undone by restoring a snapshot and replaying the accepted ones.
Rotary tables are multimodal (MRoPE): text tokens use the same position on all three axes.
"""

from __future__ import annotations

import torch

from ..base import register_adapter
from ..standard import SparseStepMoeMixin, StandardDecoderAdapter

__all__ = ["Qwen3NextAdapter", "Qwen3_5Adapter"]


@register_adapter("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text")
class Qwen3_5Adapter(StandardDecoderAdapter):
    """Qwen3.5 (see the module docstring)."""

    def position_embeddings(self, like: torch.Tensor, positions: torch.Tensor, rope: str = "default") -> object:
        positions = positions.to(like.device)
        return self.rotary_emb(like, positions[None].expand(3, *positions.shape))


@register_adapter("qwen3_next")
class Qwen3NextAdapter(SparseStepMoeMixin, StandardDecoderAdapter):
    """Qwen3-Next: the Qwen3.5 layer pattern with a standard rotary table."""
