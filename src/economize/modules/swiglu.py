"""Gated feed-forward blocks: :class:`SwiGLU` (predictors and readout adapters, with a near-zero output init) and
:class:`GatedMLP` (the Llama/Qwen MLP with Hugging Face parameter names, for draft layers loaded from published
checkpoints)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["GatedMLP", "SwiGLU"]


class SwiGLU(nn.Module):
    """``out(silu(gate(x)) * value(x))`` with bias-free linear maps.

    Args:
        in_features: Input width.
        hidden_features: Width of the gated hidden layer.
        out_features: Output width.
        init_std: Standard deviation of the normal init of ``gate`` and ``value``.
        out_init_std: Standard deviation of the init of ``out``; ``0`` gives a zero init, so the block starts as
            the zero function.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        init_std: float = 0.02,
        out_init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(in_features, hidden_features, bias=False)
        self.value = nn.Linear(in_features, hidden_features, bias=False)
        self.out = nn.Linear(hidden_features, out_features, bias=False)
        self.init_std, self.out_init_std = init_std, out_init_std
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.gate.weight, std=self.init_std)
        nn.init.normal_(self.value.weight, std=self.init_std)
        if self.out_init_std:
            nn.init.normal_(self.out.weight, std=self.out_init_std)
        else:
            nn.init.zeros_(self.out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.gate(x)) * self.value(x))


class GatedMLP(nn.Module):
    """``down_proj(silu(gate_proj(x)) * up_proj(x))`` with bias-free linear maps and the parameter names of Hugging
    Face's ``LlamaMLP``, so published draft-layer weights load as they are (PyTorch's default init).

    Args:
        hidden_size: Input and output width.
        intermediate_size: Width of the gated hidden layer.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
