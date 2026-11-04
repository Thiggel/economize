"""The DFlash draft decoder layer, in plain PyTorch (in its weights' dtype, attending to the projected context and
to the block)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ...modules import GatedMLP, RMSNorm, apply_rotary

__all__ = ["DFlashLayer"]


def _rotate(x: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Rotate heads ``[B, H, S, D]`` by a table ``[B, S, D']`` (cast to ``x``'s dtype, as the reference does)."""
    cos, sin = rotary
    return apply_rotary(x, cos[:, None].to(x.dtype), sin[:, None].to(x.dtype))


class _Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int, eps: float,
                 bias: bool) -> None:
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)


class DFlashLayer(nn.Module):
    """One DFlash draft layer (``Qwen3DFlashDecoderLayer`` of z-lab/dflash, ``speculators``' ``DFlashDraftModel``).

    For block states ``h`` and the fused context ``c`` (one vector per committed token)::

        x = input_layernorm(h)
        q = rope(q_norm(q_proj(x)))
        k = rope(k_norm(k_proj([c; x]))),  v = v_proj([c; x])
        a = h + o_proj(Attention(q, k, v))
        y = a + MLP(post_attention_layernorm(a))

    The context enters ``k_proj`` and ``v_proj`` without a norm of the layer (``hidden_norm`` of the drafter
    normalizes it once for all layers). Parameter names follow the official checkpoints (``layers.N.*``). The
    context's keys and values (:meth:`context_kv`) are computed once per committed token and cached by the drafter.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, num_kv_heads: int, head_dim: int,
                 eps: float, bias: bool = False) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.self_attn = _Attention(hidden_size, num_heads, num_kv_heads, head_dim, eps, bias)
        self.mlp = GatedMLP(hidden_size, intermediate_size)

    @property
    def num_heads(self) -> int:
        return self.self_attn.num_heads

    @property
    def num_kv_heads(self) -> int:
        return self.self_attn.num_kv_heads

    @property
    def head_dim(self) -> int:
        return self.self_attn.head_dim

    def context_kv(self, context: torch.Tensor,
                   rotary: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotated keys and values ``[B, H_kv, S, D]`` of fused context vectors ``[B, S, d]``."""
        attn = self.self_attn
        context = self._input(context)
        k = attn.k_norm(self._heads(attn.k_proj(context), attn.num_kv_heads))
        return _rotate(k, rotary), self._heads(attn.v_proj(context), attn.num_kv_heads)

    def forward(self, hidden: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor], context_keys: torch.Tensor,
                context_values: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        """The layer output ``[B, S, d]`` of block states ``[B, S, d]``.

        Args:
            hidden: The block's states.
            rotary: ``(cos, sin)`` ``[B, S, D']`` of the block's positions.
            context_keys: Rotated context keys ``[B, H_kv, N, D]`` (:meth:`context_kv`).
            context_values: Context values ``[B, H_kv, N, D]``.
            visible: Bool mask broadcastable to ``[B, H, S, N + S]`` over ``[context; block]`` keys, with at least
                one visible key per query.
        """
        attn = self.self_attn
        hidden = self._input(hidden)
        x = self.input_layernorm(hidden)
        q = _rotate(attn.q_norm(self._heads(attn.q_proj(x), attn.num_heads)), rotary)
        k = _rotate(attn.k_norm(self._heads(attn.k_proj(x), attn.num_kv_heads)), rotary)
        v = self._heads(attn.v_proj(x), attn.num_kv_heads)
        keys = torch.cat([context_keys.to(k.dtype), k], 2)
        values = torch.cat([context_values.to(v.dtype), v], 2)
        repeat = self.num_heads // self.num_kv_heads
        if repeat > 1:
            keys, values = keys.repeat_interleave(repeat, 1), values.repeat_interleave(repeat, 1)
        attended = F.scaled_dot_product_attention(q, keys, values, attn_mask=visible)
        batch, _, length, _ = attended.shape
        hidden = hidden + attn.o_proj(attended.transpose(1, 2).reshape(batch, length, -1))
        return hidden + self.mlp(self.post_attention_layernorm(hidden))

    def _input(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.input_layernorm.weight  # the inputs may sit on another device under ``device_map``
        return x.to(weight.device, weight.dtype)

    def _heads(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, heads, self.head_dim).transpose(1, 2)
