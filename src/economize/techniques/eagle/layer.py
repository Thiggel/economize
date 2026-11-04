"""The EAGLE-1/2 draft decoder layer, in plain PyTorch (in its weights' dtype, its own attention over its own keys
and values)."""

from __future__ import annotations

import torch
from torch import nn

from ...modules import GatedMLP, RMSNorm
from .._draft_layer import DraftAttention, DraftLayer

__all__ = ["EagleLayer"]


class EagleLayer(DraftLayer):
    """One EAGLE-1 draft layer with its input map ``fc`` (SafeAILab/EAGLE ``cnets1.py``: ``Model.fc`` and the
    decoder layer of index 0, which has no input norm).

    For the source ``f`` and the token embedding ``e`` at a position::

        x = fc([e; f])
        a = x + o_proj(Attention(q, k, v of x))
        y = a + MLP(norm_post(a))

    Parameter names follow the official checkpoints: ``fc`` there, ``self_attn``, ``mlp`` and
    ``post_attention_layernorm`` under ``layers.0``.
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
        fc_bias: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = DraftAttention(hidden_size, hidden_size, num_heads, num_kv_heads, head_dim, bias)
        self.mlp = GatedMLP(hidden_size, intermediate_size)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.fc = nn.Linear(2 * hidden_size, hidden_size, bias=fc_bias)

    def _input(self, source: torch.Tensor, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.fc.weight  # the embedding may sit on another device under ``device_map``
        x = self.fc(torch.cat((embedding.to(weight.device, weight.dtype), source.to(weight.device, weight.dtype)), -1))
        return x, x
