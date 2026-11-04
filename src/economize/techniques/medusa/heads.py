"""Medusa heads: residual SiLU blocks and an optional vocabulary projection."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["MedusaHead", "ResBlock"]


class ResBlock(nn.Module):
    """``x + SiLU(W x + b)`` with ``W`` initialized to zero, so a new block is the identity."""

    def __init__(self, hidden_size: int, zero_bias: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        with torch.no_grad():
            self.linear.weight.zero_()
            if zero_bias:
                self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.silu(self.linear(x))


class MedusaHead(nn.Module):
    """One Medusa head: ``num_layers`` residual blocks, then an optional bias-free projection to the vocabulary.

    Args:
        hidden_size: Width ``d`` of the model.
        vocab_size: Rows of the projection (ignored without one).
        num_layers: Residual blocks.
        projection: Give the head its own projection to the vocabulary (otherwise it returns hidden states).
        zero_bias: Initialize the blocks' biases to zero instead of PyTorch's default.
    """

    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int, projection: bool,
                 zero_bias: bool = False) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(ResBlock(hidden_size, zero_bias) for _ in range(num_layers))
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False) if projection else None

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Output of the residual blocks for ``x`` ``[..., d]``."""
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        return x if self.proj is None else self.proj(x)
