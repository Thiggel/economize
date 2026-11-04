"""RMS normalisation used by the added components: parameter-free (:func:`rms_normalize`) and with a learnable
weight (:class:`RMSNorm`, the Llama/Qwen norm)."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["RMSNorm", "rms_normalize"]


def rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scale ``x`` to unit root mean square over the last dimension (RMSNorm without a weight).

    The statistic is computed in float32 and cast back to ``x.dtype`` before the product, which matches the
    reference implementation bit for bit.
    """
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


class RMSNorm(nn.Module):
    """``weight * x / sqrt(mean(x^2) + eps)``, the Llama/Qwen norm (Hugging Face ``LlamaRMSNorm``): the statistic and
    the normalization in float32, cast back to the input dtype before the weight.

    Args:
        size: Width of the normalized dimension.
        eps: Added to the mean square.
    """

    def __init__(self, size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}"
