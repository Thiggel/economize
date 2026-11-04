"""The prefix attention layer of Hydra++: one decoder layer of its own over the model's last hidden states."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ...modules import GatedMLP, RMSNorm, apply_rotary

__all__ = ["PrefixLayer"]


def _rotate(x: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Rotate heads ``[B, H, S, D]`` by a table ``[B, S, D']``."""
    cos, sin = rotary
    return apply_rotary(x, cos[:, None].to(x.dtype), sin[:, None].to(x.dtype))


class PrefixLayer(nn.Module):
    """A Llama decoder layer (the reference's one-layer ``LlamaModel``) whose input is the model's last hidden
    state instead of a token embedding::

        a = x + o_proj(Attention(q, k, v of norm_in(x)))
        y = a + MLP(norm_post(a))        (then norm(y) with an output norm)

    Attention is split into :meth:`keys_values`, :meth:`query` and :meth:`attend` / :meth:`finish`, so the drafter
    can keep the keys and values of past tokens in its own cache and query only each row's last token.

    Args:
        hidden_size: Width ``d`` of the model.
        intermediate_size: Width of the gated MLP.
        num_heads, num_kv_heads, head_dim: Attention geometry.
        eps: Epsilon of the norms.
        bias: Biases on the query, key and value maps.
        output_norm: End with an RMSNorm (the reference model's final norm).
    """

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, num_kv_heads: int, head_dim: int,
                 eps: float, bias: bool = False, output_norm: bool = True) -> None:
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.mlp = GatedMLP(hidden_size, intermediate_size)
        self.norm = RMSNorm(hidden_size, eps) if output_norm else None

    def keys_values(self, x: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotated keys and values ``[B, H_kv, S, D]`` of the inputs ``[B, S, d]``."""
        x = self.input_layernorm(x)
        return _rotate(self._heads(self.k_proj(x), self.num_kv_heads), rotary), self._heads(self.v_proj(x),
                                                                                           self.num_kv_heads)

    def query(self, x: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Rotated queries ``[B, H, S, D]`` of the inputs ``[B, S, d]``."""
        return _rotate(self._heads(self.q_proj(self.input_layernorm(x)), self.num_heads), rotary)

    def attend(self, q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
               visible: torch.Tensor | None) -> torch.Tensor:
        """Attention output ``[B, H, S, D]`` over ``keys``/``values`` ``[B, H_kv, N, D]``; ``visible`` is a bool mask
        broadcastable to ``[B, H, S, N]``, ``None`` for causal attention over ``S == N`` tokens."""
        repeat = self.num_heads // self.num_kv_heads
        if repeat > 1:
            keys, values = keys.repeat_interleave(repeat, 1), values.repeat_interleave(repeat, 1)
        return F.scaled_dot_product_attention(q, keys, values, attn_mask=visible, is_causal=visible is None)

    def finish(self, x: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        """The layer output ``[B, S, d]`` from its inputs and the attention output ``[B, H, S, D]``."""
        batch, _, length, _ = attended.shape
        x = x + self.o_proj(attended.transpose(1, 2).reshape(batch, length, -1))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x if self.norm is None else self.norm(x)

    def forward(self, x: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Causal pass over whole sequences ``[B, S, d]`` (training)."""
        return self.finish(x, self.attend(self.query(x, rotary), *self.keys_values(x, rotary), None))

    def _heads(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, heads, self.head_dim).transpose(1, 2)
