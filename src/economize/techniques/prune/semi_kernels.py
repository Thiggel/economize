"""Registry of storage kernels for 2:4-pruned weights, and the timing policy that picks one per weight shape.

Every entry turns a dense weight that satisfies the 2:4 pattern into the tensor ``nn.Linear`` should hold: a
``torch.sparse`` semi-structured tensor (cuSPARSELt or CUTLASS sparse tensor cores) or the masked dense weight
itself (cuBLAS). The sparse kernels are not uniformly faster. On an A100 at 1-8 decode tokens and Gemma-4-31B shapes
the cuSPARSELt GEMM runs the wide MLP up/gate maps (21504 x 5376) about 1.4-1.6x faster than dense, and the attention
maps and the down map 0.4-0.8x as fast. ``"auto"`` therefore times
every available kernel on the shape of each weight at decode token counts and keeps the fastest, so a pruned model
is never slower than its masked dense form::

    PruneConfig(sparsity="2:4", semi_structured=True, semi_structured_kernel="auto")

A new kernel is added by registration::

    from economize.techniques.prune import SEMI_STRUCTURED_KERNELS, SemiStructuredKernel

    SEMI_STRUCTURED_KERNELS.register("my_kernel", SemiStructuredKernel(convert=my_convert, available=my_check))
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ...core.metadata import Saving, Simulation
from ...registry import Registry

__all__ = [
    "AUTO_CANDIDATES",
    "AUTO_TOKENS",
    "SEMI_STRUCTURED_KERNELS",
    "SemiStructuredKernel",
    "auto_choice",
    "clear_auto_cache",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemiStructuredKernel:
    """One way to store (and multiply with) a 2:4-pruned weight.

    Attributes:
        convert: Maps a 2:4 dense CUDA weight (fp16 / bf16, shape ``(out, in)``) to the tensor the linear layer
            stores; it may raise when the shape is unsupported.
        available: Whether the kernel can run on a CUDA device.
        description: One line for listings.
        simulation: Set when the kernel keeps the dense weight (no memory or compute saving; see
            :class:`~economize.Simulation`).
    """

    convert: Callable[[torch.Tensor], torch.Tensor]
    available: Callable[[torch.device], bool]
    description: str = ""
    simulation: Simulation | None = None


SEMI_STRUCTURED_KERNELS: Registry[SemiStructuredKernel] = Registry(
    "semi-structured kernel", "economize.semi_structured_kernels"
)

#: Kernels ``"auto"`` times, in order of preference on ties.
AUTO_CANDIDATES: tuple[str, ...] = ("dense", "cusparselt", "cutlass")
#: Token counts (rows of the activation) ``"auto"`` times each kernel at; the summed time decides.
AUTO_TOKENS: tuple[int, ...] = (1, 8)

_AUTO_CACHE: dict[tuple[object, ...], str] = {}


def _sm80(device: torch.device) -> bool:
    return (
        torch.cuda.is_available()
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] >= 8
        and hasattr(torch.sparse, "to_sparse_semi_structured")
    )


def _cusparselt_available(device: torch.device) -> bool:
    backend = getattr(torch.backends, "cusparselt", None)
    return _sm80(device) and backend is not None and bool(backend.is_available())


def _cutlass_convert(weight: torch.Tensor) -> torch.Tensor:
    return torch.sparse.SparseSemiStructuredTensorCUTLASS.from_dense(weight)


def _cusparselt_convert(weight: torch.Tensor) -> torch.Tensor:
    return torch.sparse.SparseSemiStructuredTensorCUSPARSELT.from_dense(weight)


SEMI_STRUCTURED_KERNELS.register(
    "default",
    SemiStructuredKernel(
        convert=torch.sparse.to_sparse_semi_structured,
        available=_sm80,
        description="torch.sparse.to_sparse_semi_structured: PyTorch's default sparse backend (cuSPARSELt if built)",
    ),
)
SEMI_STRUCTURED_KERNELS.register(
    "cusparselt",
    SemiStructuredKernel(
        convert=_cusparselt_convert, available=_cusparselt_available, description="cuSPARSELt sparse tensor cores"
    ),
)
SEMI_STRUCTURED_KERNELS.register(
    "cutlass",
    SemiStructuredKernel(convert=_cutlass_convert, available=_sm80, description="CUTLASS sparse tensor cores"),
)
SEMI_STRUCTURED_KERNELS.register(
    "dense",
    SemiStructuredKernel(
        convert=lambda weight: weight,
        available=lambda device: device.type == "cuda",
        description="keep the masked dense weight (cuBLAS); same function, dense speed",
        simulation=Simulation(Saving.WEIGHT_MEMORY | Saving.COMPUTE, "keeps the masked dense weight"),
    ),
)


def _time_ms(weight: torch.Tensor, inputs: Sequence[torch.Tensor], reps: int) -> float:
    """Median ms of one round of ``F.linear`` over ``inputs`` (eager, CUDA events, after warmup)."""

    def run() -> None:
        for x in inputs:
            F.linear(x, weight)

    for _ in range(3):
        run()
    torch.cuda.synchronize(weight.device)
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(4):
            run()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 4)
    return statistics.median(times)


def auto_choice(
    weight: torch.Tensor,
    candidates: Sequence[str] = AUTO_CANDIDATES,
    tokens: Sequence[int] = AUTO_TOKENS,
    reps: int = 10,
) -> str:
    """Name of the fastest of ``candidates`` for ``weight``'s shape, dtype and device at ``tokens`` rows.

    Kernels that are unavailable or fail to convert or run on the shape are skipped; ``"dense"`` always works, so a
    choice always exists. The result is cached per (shape, dtype, device name, candidates, tokens) for the process.
    """
    device = weight.device
    key = (tuple(weight.shape), weight.dtype, torch.cuda.get_device_name(device), tuple(candidates), tuple(tokens))
    if key in _AUTO_CACHE:
        return _AUTO_CACHE[key]
    inputs = [torch.randn(m, weight.shape[1], device=device, dtype=weight.dtype) for m in tokens]
    timings: dict[str, float] = {}
    for name in candidates:
        kernel = SEMI_STRUCTURED_KERNELS.get(name)
        if not kernel.available(device):
            continue
        try:
            stored = kernel.convert(weight.detach())
            timings[name] = _time_ms(stored, inputs, reps)
        except (RuntimeError, ValueError, NotImplementedError) as error:
            logger.debug("semi-structured kernel %r unusable for %s: %s", name, tuple(weight.shape), error)
        finally:
            stored = None  # free a sparse copy before the next candidate
    best = min(timings, key=timings.__getitem__) if timings else "dense"
    logger.info(
        "semi-structured kernel for %s %s: %s (%s)",
        tuple(weight.shape),
        weight.dtype,
        best,
        ", ".join(f"{k} {v * 1e3:.1f} us" for k, v in timings.items()),
    )
    _AUTO_CACHE[key] = best
    return best


def clear_auto_cache() -> None:
    """Forget the cached ``"auto"`` choices (e.g. after changing :data:`AUTO_CANDIDATES`)."""
    _AUTO_CACHE.clear()


def _auto_convert(weight: torch.Tensor) -> torch.Tensor:
    return SEMI_STRUCTURED_KERNELS.get(auto_choice(weight)).convert(weight)


SEMI_STRUCTURED_KERNELS.register(
    "auto",
    SemiStructuredKernel(
        convert=_auto_convert,
        available=lambda device: device.type == "cuda",
        description="time dense / cuSPARSELt / CUTLASS per weight shape at decode token counts and keep the fastest",
    ),
)
