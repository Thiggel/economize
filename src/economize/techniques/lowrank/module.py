"""The two-factor linear map that replaces a compressed ``nn.Linear``."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .._weights.transaction import replace_module  # re-exported: it moved to the shared weight helpers

__all__ = ["LowRankLinear", "replace_module"]


class LowRankLinear(nn.Module):
    """``y = up(down(x))`` with ``down: in -> rank`` (no bias) and ``up: rank -> out`` (the original bias).

    Both factors are ordinary ``nn.Linear`` modules, so other weight techniques (quantization, LoRA) and kernel
    backends see two plain linear maps named ``<name>.down`` and ``<name>.up``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=bias, device=device, dtype=dtype)

    @classmethod
    @torch.no_grad()
    def from_factors(cls, up: Tensor, down: Tensor, bias: Tensor | None, like: Tensor) -> LowRankLinear:
        """A module holding ``up [out, k]``, ``down [k, in]`` and ``bias``, on the device and dtype of ``like``."""
        module = cls(down.shape[1], up.shape[0], up.shape[1], bias is not None, like.device, like.dtype)
        module.down.weight.copy_(down)
        module.up.weight.copy_(up)
        if bias is not None:
            module.up.bias.copy_(bias)
        return module.requires_grad_(like.requires_grad)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.down(x))

    def dense_weight(self) -> Tensor:
        """The product ``up @ down`` ``[out, in]`` (for inspection; the forward never forms it)."""
        return self.up.weight @ self.down.weight

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}"


