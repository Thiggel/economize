"""One multi-token-prediction module: two input norms, a projection, one decoder layer and an output norm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from ...modules import trainable_copy
from .config import MTPLayout

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...runtime import LayerContext

__all__ = ["MTPModule", "copy_layer"]


def copy_layer(layer: nn.Module, layer_idx: int | None = None) -> nn.Module:
    """A trainable copy of ``layer`` in its own dtype (kept for compatibility: :func:`economize.modules.trainable_copy`
    with ``dtype=None``)."""
    return trainable_copy(layer, None, layer_idx)


class MTPModule(nn.Module):
    """``norm(block(proj(cat(enorm(E[x]), hnorm(h)))))`` with the block's own KV cache entries.

    Submodule names follow the DeepSeek-V3 checkpoints (``enorm``, ``hnorm``, ``eh_proj``, ``block``, ``norm``);
    :mod:`economize.techniques.vendor_mtp.formats` maps them to the other vendors' names.

    Args:
        adapter: The model's adapter (its final norm initializes the three norms).
        block: The decoder layer (a :func:`~economize.modules.trainable_copy`).
        block_layer: The model layer (1-indexed) whose attention pattern and rotary table the block uses.
        layout: Input order and position-0 rule.
    """

    def __init__(self, adapter: ArchAdapter, block: nn.Module, block_layer: int, layout: MTPLayout) -> None:
        super().__init__()
        d = adapter.hidden_size
        self.enorm = trainable_copy(adapter.final_norm, None)
        self.hnorm = trainable_copy(adapter.final_norm, None)
        self.eh_proj = nn.Linear(2 * d, d, bias=False)
        nn.init.normal_(self.eh_proj.weight, std=0.02)
        self.block = block
        self.norm = trainable_copy(adapter.final_norm, None)
        self.block_layer = block_layer
        self.layout = layout

    def forward(self, adapter: ArchAdapter, state: torch.Tensor, embedding: torch.Tensor,
                ctx: LayerContext) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the module on states and token embeddings ``[B, S, d]`` within ``ctx`` (whose positions are the
        entries' positions and whose slots receive the block's K/V).

        Returns:
            The block output before and after the module's norm, ``[B, S, d]`` each.
        """
        dtype = self.eh_proj.weight.dtype
        if self.layout.zero_first_embedding:
            first = ctx.positions.to(embedding.device)[..., None] == 0
            embedding = embedding.masked_fill(first, 0.0)
        e, h = self.enorm(embedding.to(dtype)), self.hnorm(state.to(dtype))
        mixed = self.eh_proj(torch.cat([e, h] if self.layout.concat_order == "emb_hidden" else [h, e], -1))
        with ctx.cache.writing_at(ctx.slots):
            raw = adapter.call_layer(self.block_layer, self.block, mixed, ctx)
        return raw, self.norm(raw)
