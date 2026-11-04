"""Hydra heads: a widening residual block over the base state and the drafted embeddings, then Medusa blocks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..medusa.heads import ResBlock

__all__ = ["HydraHead", "InputBlock"]


class InputBlock(nn.Module):
    """``res(x) + SiLU(W x + b)`` from ``x`` ``[..., n d]`` to ``[..., d]``, the first block of a head
    (the reference's ``ResBlock(hidden_size, num_condition)``).

    ``W`` starts at zero. ``res`` is a linear map with PyTorch's default init, as in the reference, or with
    ``identity=True`` the selection of the first ``d`` inputs with a zero bias, so a new head starts as Medusa's.

    Args:
        hidden_size: Width ``d`` of the model.
        inputs: ``n``, the number of ``d`` wide inputs (the base state plus the embeddings).
        identity: Initialize ``res`` to pass the base state through.
        zero_bias: Initialize ``b`` to zero instead of PyTorch's default.
    """

    def __init__(self, hidden_size: int, inputs: int, identity: bool = False, zero_bias: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(inputs * hidden_size, hidden_size)
        self.res = nn.Linear(inputs * hidden_size, hidden_size)
        with torch.no_grad():
            self.linear.weight.zero_()
            if zero_bias:
                self.linear.bias.zero_()
            if identity:
                self.res.weight.zero_()
                self.res.weight[:, :hidden_size].copy_(torch.eye(hidden_size))
                self.res.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(x) + F.silu(self.linear(x))


class HydraHead(nn.Module):
    """Head ``j`` of Hydra: an :class:`InputBlock` over ``j + 1`` inputs, ``num_layers - 1`` residual blocks, then an
    optional vocabulary projection with bias.

    Args:
        hidden_size: Width ``d`` of the model.
        vocab_size: Rows of the projection (ignored without one).
        inputs: ``j + 1``.
        num_layers: Blocks, the input block included.
        projection: Give the head its own projection to the vocabulary (otherwise it returns hidden states).
        shared: Initialize for the model's own final norm and LM head (identity input block, zero biases).
    """

    def __init__(self, hidden_size: int, vocab_size: int, inputs: int, num_layers: int, projection: bool,
                 shared: bool = False) -> None:
        super().__init__()
        blocks: list[nn.Module] = [InputBlock(hidden_size, inputs, identity=shared, zero_bias=shared)]
        blocks += [ResBlock(hidden_size, zero_bias=shared) for _ in range(num_layers - 1)]
        self.blocks = nn.ModuleList(blocks)
        self.proj = nn.Linear(hidden_size, vocab_size) if projection else None

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Output of the blocks for the inputs ``x`` ``[..., (j + 1) d]``."""
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        return x if self.proj is None else self.proj(x)
