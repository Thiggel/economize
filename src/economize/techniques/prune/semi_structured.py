"""The 2:4 speed path: PyTorch semi-structured sparse tensors (cuSPARSELt / CUTLASS kernels on sm80+).

Masked dense weights compute the same function as the dense model at the same speed. Only 2:4 weights stored as
``torch.sparse.SparseSemiStructuredTensor`` run on sparse tensor cores; NVIDIA and the Wanda paper report about
1.2-1.6x for the linear layers at large token counts and little to nothing at batch-1 decoding, where the GEMV is
bound by memory traffic of the compressed values plus metadata. The storage kernel is chosen from
:data:`~.semi_kernels.SEMI_STRUCTURED_KERNELS` (``"auto"`` keeps, per weight shape, whichever of dense, cuSPARSELt and
CUTLASS is fastest at decode token counts).
"""

from __future__ import annotations

import torch
from torch import nn

from ...errors import ConfigError
from .masks import check_nm

__all__ = ["semi_structured_available", "to_semi_structured"]


def semi_structured_available(device: torch.device | str | None = None) -> bool:
    """Whether ``torch.sparse.to_sparse_semi_structured`` can run on ``device`` (CUDA, compute capability >= 8.0)."""
    if not torch.cuda.is_available():
        return False
    target = torch.device(device or "cuda")
    if target.type != "cuda":
        return False
    return torch.cuda.get_device_capability(target)[0] >= 8 and hasattr(torch.sparse, "to_sparse_semi_structured")


def to_semi_structured(linear: nn.Linear, kernel: str = "default") -> None:
    """Replace a 2:4-pruned ``linear.weight`` by its form for the registered semi-structured ``kernel``.

    ``"default"`` is ``torch.sparse.to_sparse_semi_structured``; ``"auto"`` may keep the masked dense weight where
    the sparse kernels are slower (inference only).
    """
    from .semi_kernels import SEMI_STRUCTURED_KERNELS

    weight = linear.weight
    if not semi_structured_available(weight.device):
        raise ConfigError("semi-structured 2:4 sparsity needs a CUDA GPU with compute capability >= 8.0")
    if weight.dtype not in (torch.float16, torch.bfloat16):
        raise ConfigError(f"semi-structured 2:4 sparsity needs fp16 or bf16 weights, got {weight.dtype}")
    if not check_nm(weight, 2, 4):
        raise ConfigError("weight does not satisfy the 2:4 pattern")
    stored = SEMI_STRUCTURED_KERNELS.get(kernel).convert(weight.detach())
    linear.weight = nn.Parameter(stored, requires_grad=False)
