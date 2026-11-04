"""The EAGLE-3 draft decoder layer, in plain PyTorch (in its weights' dtype, its own attention over its own keys and
values)."""

from __future__ import annotations

import torch

from ...modules import GatedMLP, RMSNorm
from .._draft_layer import DraftAttention, DraftLayer

__all__ = ["Eagle3Layer"]


class Eagle3Layer(DraftLayer):
    """One EAGLE-3 draft layer (SafeAILab/EAGLE ``cnets.py``, SpecForge ``LlamaForCausalLMEagle3``).

    For the source ``g`` and the token embedding ``e`` at a position::

        r = g (or norm_h(g) with norm_before_residual)
        a = r + o_proj(Attention(q, k, v of [norm_e(e); norm_h(g)]))
        y = a + MLP(norm_post(a))

    Parameter names follow the official checkpoints (``midlayer.*``): ``input_layernorm`` normalizes the
    embedding, ``hidden_norm`` the source.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        eps: float,
        bias: bool = False,
        norm_before_residual: bool = False,
    ) -> None:
        super().__init__()
        self.norm_before_residual = norm_before_residual
        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.hidden_norm = RMSNorm(hidden_size, eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.self_attn = DraftAttention(2 * hidden_size, hidden_size, num_heads, num_kv_heads, head_dim, bias)
        self.mlp = GatedMLP(hidden_size, intermediate_size)

    def _input(self, source: torch.Tensor, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.hidden_norm.weight  # the embedding may sit on another device under ``device_map``
        dtype = weight.dtype
        source, embedding = source.to(weight.device, dtype), embedding.to(weight.device)
        normed = self.hidden_norm(source)
        residual = normed if self.norm_before_residual else source
        return residual, torch.cat((self.input_layernorm(embedding.to(dtype)), normed), -1)
