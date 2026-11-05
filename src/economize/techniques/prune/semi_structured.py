"""The 2:4 speed path: PyTorch semi-structured sparse tensors (cuSPARSELt / CUTLASS kernels on sm80+).

Masked dense weights compute the same function as the dense model at the same speed. Only 2:4 weights stored as
``torch.sparse.SparseSemiStructuredTensor`` run on sparse tensor cores; NVIDIA and the Wanda paper report about
1.2-1.6x for the linear layers at large token counts and little to nothing at batch-1 decoding, where the GEMV is
bound by memory traffic of the compressed values plus metadata. The storage kernel is chosen from
:data:`~.semi_kernels.SEMI_STRUCTURED_KERNELS` (``"auto"`` keeps, per weight shape, whichever of dense, cuSPARSELt and
CUTLASS is fastest at decode token counts).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from torch import nn

from ...errors import ConfigError
from .masks import check_nm

if TYPE_CHECKING:
    from .config import PruneConfig

__all__ = [
    "NO_SPARSE_FORMAT",
    "REFERENCE_HINT",
    "check_storable",
    "semi_structured_available",
    "storage_reason",
    "to_semi_structured",
]

#: How to ask for the simulated masked-dense reference (appended to every refusal).
REFERENCE_HINT = "pass semi_structured=False for the masked dense reference (simulated: no memory or speed saving)"
#: Why a pattern other than 2:4 has no real storage (``{pattern}`` is the configured pattern).
NO_SPARSE_FORMAT = (
    "no sparse weight format runs {pattern} sparsity faster or in less memory than the dense weights (sparse tensor "
    "cores run only 2:4); use sparsity='2:4' or " + REFERENCE_HINT
)


def storage_reason(config: PruneConfig) -> str | None:
    """Why ``config``'s pruned weights cannot be stored in a sparse format here (``None``: they can, or the
    configuration is the explicit masked-dense reference).

    Only the installed software is checked (the device is checked when the weights are converted, by
    :func:`check_storable`).

    Parameters
    ----------
    config : PruneConfig
        The pruning configuration.

    Returns
    -------
    str or None
        The reason, with the way out.
    """
    if config.reference:
        return None
    if not config.sparse_storage:
        return NO_SPARSE_FORMAT.format(pattern=config.pattern)
    if not hasattr(torch.sparse, "to_sparse_semi_structured"):
        return (f"this PyTorch ({torch.__version__}) has no torch.sparse.to_sparse_semi_structured (2.1+ needed); "
                + REFERENCE_HINT)
    return None


def check_storable(config: PruneConfig, linears: Iterable[nn.Linear]) -> None:
    """Raise before any weight changes if ``config``'s sparse storage cannot hold the weights of ``linears``.

    Parameters
    ----------
    config : PruneConfig
        The pruning configuration.
    linears : Iterable[nn.Linear]
        The maps that will be pruned.

    Raises
    ------
    ConfigError
        :func:`storage_reason` refuses the configuration, or a weight is not an fp16 / bf16 weight on a CUDA GPU of
        compute capability 8.0 or higher (what the semi-structured kernels need).
    """
    reason = storage_reason(config)
    if reason is not None:
        raise ConfigError(f"prune(sparsity={config.sparsity!r}): {reason}")
    if not config.sparse_storage:
        return
    for linear in linears:
        weight = linear.weight
        if not semi_structured_available(weight.device):
            raise ConfigError(f"semi-structured 2:4 storage needs a CUDA GPU with compute capability >= 8.0, the "
                              f"weights are on {weight.device}; move the model to such a GPU or {REFERENCE_HINT}")
        if weight.dtype not in (torch.float16, torch.bfloat16):
            raise ConfigError(f"semi-structured 2:4 storage needs fp16 or bf16 weights, got {weight.dtype}; cast "
                              f"the model or {REFERENCE_HINT}")


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
