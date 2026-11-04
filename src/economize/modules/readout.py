"""Readout adapter for hidden states taken from an intermediate layer."""

from __future__ import annotations

import torch
from torch import nn

from .norm import rms_normalize
from .swiglu import SwiGLU

__all__ = ["ReadoutAdapter"]


class ReadoutAdapter(nn.Module):
    """Maps the output of an intermediate layer toward the final residual stream before the model's norm and head.

    ``h + linear(n) + mlp(n)`` with ``n = rms_normalize(h)``, computed in float32. Both branches start at zero, so
    an untrained adapter is the identity.

    Args:
        hidden_size: Width of the residual stream.
        width: Hidden width of the gated branch (default: ``hidden_size``).
    """

    def __init__(self, hidden_size: int, width: int | None = None) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp = SwiGLU(hidden_size, width or hidden_size, hidden_size, out_init_std=0.0)
        nn.init.zeros_(self.linear.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        h = hidden.float()
        n = rms_normalize(h)
        return h + self.linear(n) + self.mlp(n)
