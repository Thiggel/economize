"""The Kangaroo adapter: one attention layer and two RMS norms between the shallow sub-network and the LM head."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ...modules import RMSNorm, apply_rotary

__all__ = ["KangarooAdapter"]


def _rotate(x: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Rotate heads ``[B, H, S, D]`` by a table ``[B, S, D']``."""
    cos, sin = rotary
    return apply_rotary(x, cos[:, None].to(x.dtype), sin[:, None].to(x.dtype))


class KangarooAdapter(nn.Module):
    """Kangaroo's adapter network (``kangaroo/adapter.py`` of the reference code).

    For the shallow state ``h`` (output of the model's layer ``l``) at a position::

        a = h + o_proj(Attention(q, k, v of input_layernorm(h)))
        y = norm(a)

    and ``y`` goes to the model's LM head. The attention is causal over the adapter's own keys and values, one per
    position. As in :class:`~economize.techniques.eagle3.Eagle3Layer`, attention is split into
    :meth:`project`, :meth:`attend` and :meth:`finish` so that a drafter can keep the keys and values in a cache.
    """

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int, eps: float,
                 bias: bool = False) -> None:
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.norm = RMSNorm(hidden_size, eps)

    def keys_values(self, hidden: torch.Tensor,
                    rotary: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotated keys and values ``[B, H_kv, S, D]`` of shallow states ``[B, S, d]``."""
        x = self.input_layernorm(self._input(hidden))
        return _rotate(self._heads(self.k_proj(x), self.num_kv_heads), rotary), self._heads(self.v_proj(x),
                                                                                            self.num_kv_heads)

    def project(self, hidden: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Residual ``[B, S, d]`` and rotated queries, keys and values ``[B, H, S, D]`` / ``[B, H_kv, S, D]``."""
        residual = self._input(hidden)
        x = self.input_layernorm(residual)
        q = _rotate(self._heads(self.q_proj(x), self.num_heads), rotary)
        k = _rotate(self._heads(self.k_proj(x), self.num_kv_heads), rotary)
        return residual, q, k, self._heads(self.v_proj(x), self.num_kv_heads)

    def attend(self, q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
               visible: torch.Tensor | None) -> torch.Tensor:
        """Attention output ``[B, H, S, D]`` over ``keys``/``values`` ``[B, H_kv, N, D]``: ``visible`` is a bool mask
        broadcastable to ``[B, H, S, N]``, or ``None`` for causal attention over the ``S == N`` positions."""
        repeat = self.num_heads // self.num_kv_heads
        if repeat > 1:
            keys, values = keys.repeat_interleave(repeat, 1), values.repeat_interleave(repeat, 1)
        return F.scaled_dot_product_attention(q, keys, values, attn_mask=visible, is_causal=visible is None)

    def finish(self, residual: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        """The normalized adapter output ``[B, S, d]`` that the LM head reads."""
        batch, _, length, _ = attended.shape
        return self.norm(residual + self.o_proj(attended.transpose(1, 2).reshape(batch, length, -1)))

    def forward(self, hidden: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """The causal pass over a sequence of shallow states ``[B, S, d]`` (training)."""
        residual, q, k, v = self.project(hidden, rotary)
        return self.finish(residual, self.attend(q, k, v, None))

    def _input(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = self.q_proj.weight  # the shallow states may sit on another device under ``device_map``
        return hidden.to(weight.device, weight.dtype)

    def _heads(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, heads, self.head_dim).transpose(1, 2)
