"""The attention and MLP of a one-layer draft decoder, in plain PyTorch (in its weights' dtype, its own attention
over its own keys and values). Subclasses decide how the source and the token embedding enter the layer."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

from ...modules import GatedMLP, RMSNorm, apply_rotary

__all__ = ["DraftAttention", "DraftLayer"]


def _rotate(x: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Rotate heads ``[B, H, S, D]`` by a table ``[B, S, D']`` (cast to ``x``'s dtype, as the reference does)."""
    cos, sin = rotary
    return apply_rotary(x, cos[:, None].to(x.dtype), sin[:, None].to(x.dtype))


class DraftAttention(nn.Module):
    """The q/k/v maps of ``input_size``-wide layer inputs and the output map back to ``hidden_size``."""

    def __init__(self, input_size: int, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int,
                 bias: bool) -> None:
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.q_proj = nn.Linear(input_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(input_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(input_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)


class DraftLayer(nn.Module, ABC):
    """One draft decoder layer: ``a = r + o_proj(Attention(x))`` and ``y = a + MLP(norm_post(a))``.

    :meth:`_input` turns the source and the token embedding into the residual ``r`` and the attention input ``x``.
    Subclasses register ``self_attn`` (a :class:`DraftAttention`), ``mlp`` and ``post_attention_layernorm`` under
    the names of their official checkpoints, in the order those checkpoints' code builds them. Attention is split
    into :meth:`project` (queries, keys and values, rotated) and :meth:`attend` / :meth:`finish`, so the drafter can
    store keys and values in its own cache.
    """

    self_attn: DraftAttention
    mlp: GatedMLP
    post_attention_layernorm: RMSNorm

    @property
    def num_heads(self) -> int:
        return self.self_attn.num_heads

    @property
    def num_kv_heads(self) -> int:
        return self.self_attn.num_kv_heads

    @property
    def head_dim(self) -> int:
        return self.self_attn.head_dim

    @abstractmethod
    def _input(self, source: torch.Tensor, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The residual ``[B, S, d]`` and the attention input ``[B, S, d_in]`` of sources and embeddings
        ``[B, S, d]``, on the layer's device and in its dtype."""

    def keys_values(
        self, source: torch.Tensor, embedding: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotated keys and values ``[B, H_kv, S, D]`` for sources and embeddings ``[B, S, d]`` (no attention)."""
        x = self._input(source, embedding)[1]
        attn = self.self_attn
        k = self._heads(attn.k_proj(x), attn.num_kv_heads)
        v = self._heads(attn.v_proj(x), attn.num_kv_heads)
        return _rotate(k, rotary), v

    def project(
        self, source: torch.Tensor, embedding: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Residual ``[B, S, d]`` and rotated queries, keys and values ``[B, H, S, D]`` / ``[B, H_kv, S, D]``."""
        residual, x = self._input(source, embedding)
        attn = self.self_attn
        q = _rotate(self._heads(attn.q_proj(x), attn.num_heads), rotary)
        k = _rotate(self._heads(attn.k_proj(x), attn.num_kv_heads), rotary)
        v = self._heads(attn.v_proj(x), attn.num_kv_heads)
        return residual, q, k, v

    def attend(self, q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        """Attention output ``[B, H, S, D]`` of queries over ``keys``/``values`` ``[B, H_kv, N, D]``; ``visible`` is a
        bool mask broadcastable to ``[B, H, S, N]`` with at least one visible key per query."""
        repeat = self.num_heads // self.num_kv_heads
        if repeat > 1:
            keys, values = keys.repeat_interleave(repeat, 1), values.repeat_interleave(repeat, 1)
        return F.scaled_dot_product_attention(q, keys, values, attn_mask=visible)

    def attend_unrolled(
        self,
        q: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        visible: torch.Tensor,
        own_keys: list[torch.Tensor],
        own_values: list[torch.Tensor],
    ) -> torch.Tensor:
        """Attention for the training-time test: query ``t`` sees the context keys ``visible[t]`` (``[S, N]``) and
        the keys ``own_keys[j][:, :, t]`` of its own earlier steps (one per step, ``[B, H_kv, S, D]`` each).

        This is :meth:`attend` over ``[keys; own_keys]`` with a block-diagonal mask on the own keys, computed in
        ``O(S * (N + steps))`` memory instead of ``O(S * N * steps)``.
        """
        repeat = self.num_heads // self.num_kv_heads
        expand = (lambda x: x.repeat_interleave(repeat, 1)) if repeat > 1 else (lambda x: x)
        scale = q.shape[-1] ** -0.5
        scores = torch.matmul(q, expand(keys).transpose(-1, -2)) * scale
        scores = scores.masked_fill(~visible, float("-inf"))
        own = [(q * expand(k)).sum(-1, keepdim=True) * scale for k in own_keys]
        probs = torch.cat([scores, *own], -1).softmax(-1)
        width = scores.shape[-1]
        out = torch.matmul(probs[..., :width], expand(values))
        for j, v in enumerate(own_values):
            out = out + probs[..., width + j : width + j + 1] * expand(v)
        return out

    def finish(self, residual: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        """The layer output ``[B, S, d]`` from the residual and the attention output ``[B, H, S, D]``."""
        batch, _, length, _ = attended.shape
        x = residual + self.self_attn.o_proj(attended.transpose(1, 2).reshape(batch, length, -1))
        return x + self.mlp(self.post_attention_layernorm(x))

    def _heads(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, heads, self.head_dim).transpose(1, 2)
