"""Launch parameters of the row-masked GEMV timed on the device, as the default ``auto`` MLP backend uses them.

The heuristic of :func:`.kernel.default_row_config` (32 channels by 256 inputs, 4 warps) runs the up map of a
Llama-3-8B-shaped MLP (14336 by 4096) at 1.44x the speed of the dense cuBLAS GEMV on an H100 NVL at one decode row
and 50% kept channels, and the 4864 by 896 up map of a 0.5B model at 1.04x; the best config of a grid runs them at
1.72x and 1.44x. The best config differs by device and by map shape, so :class:`RowTuner` times a grid once per
(weight shape, dtype, device name) and caches the fastest config.

Calls with several rows use the ``tl.dot`` path of the kernel. The tuner times its best case (every row keeps the
same channels) at :attr:`RowTuner.probe_rows` rows against the dense matmul; when the dense matmul still wins, the map
runs the kernel for single-row calls only. Timing replays a CUDA graph of several calls over weight copies that
together exceed the L2 cache, so the result is the bandwidth-bound decode time. :func:`clear_row_tuning_cache`
forgets the stored choices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..sparsity.triton_autotune import _graph_ms
from .kernel import RowGemvConfig, default_row_config, row_masked_gemv

__all__ = ["RowTuner", "RowTuning", "clear_row_tuning_cache"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RowTuning:
    """The tuned launch parameters of the row-masked GEMV for one weight shape.

    Attributes:
        one_row: Config of single-row calls.
        many_rows: Config of calls with 2..16 rows, or ``None`` when the dense matmul is faster there.
        one_row_speedup: Time of the dense matmul over the kernel's time at one row and 50% kept channels.
        many_rows_speedup: The same at :attr:`RowTuner.probe_rows` rows that keep the same channels.
    """

    one_row: RowGemvConfig
    many_rows: RowGemvConfig | None
    one_row_speedup: float
    many_rows_speedup: float


_CACHE: dict[tuple[object, ...], RowTuning] = {}

Grid = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
"""``(block_n, block_k, num_warps)`` value lists of a config grid."""


def clear_row_tuning_cache() -> None:
    """Forget every stored :class:`RowTuning`."""
    _CACHE.clear()


def _grid(
    block_n: tuple[int, ...], block_k: tuple[int, ...], num_warps: tuple[int, ...], k: int
) -> list[RowGemvConfig]:
    return [
        RowGemvConfig(bn, bk, warps, 2)
        for bn in block_n
        for bk in block_k
        for warps in num_warps
        if bn * bk <= 32768 and bk <= max(k, 16)
    ]


@dataclass(frozen=True)
class RowTuner:
    """Times :func:`.kernel.row_masked_gemv` configs per weight shape (see the module docstring).

    Attributes:
        grid_one_row: ``(block_n, block_k, num_warps)`` value lists of the single-row grid.
        grid_many_rows: The same for the ``tl.dot`` path (calls with several rows).
        probe_rows: Row count at which the ``tl.dot`` path is compared with the dense matmul.
        calls: Kernel calls per timed graph.
        reps: Graph replays per config (the median counts).
    """

    grid_one_row: Grid = ((8, 16, 32, 64), (128, 256, 512), (2, 4, 8))
    grid_many_rows: Grid = ((16, 32, 64), (32, 64, 128), (4,))
    probe_rows: int = 8
    calls: int = 8
    reps: int = 7

    def tune(self, weight: Tensor) -> RowTuning:
        """The :class:`RowTuning` of ``weight``'s shape (``[out, in]``), dtype and device, timed on first use."""
        n, k = weight.shape
        key = (n, k, weight.dtype, torch.cuda.get_device_name(weight.device), self)
        if key not in _CACHE:
            with torch.no_grad():
                _CACHE[key] = self._time(weight.detach())
        return _CACHE[key]

    def _time(self, weight: Tensor) -> RowTuning:
        n, k = weight.shape
        device = weight.device
        l2 = getattr(torch.cuda.get_device_properties(device), "L2_cache_size", 40 << 20)
        copies = max(1, min(4, -(-2 * l2 // (weight.numel() * weight.element_size()))))
        weights = [weight.contiguous().clone() for _ in range(copies)]
        gen = torch.Generator(device=device).manual_seed(0)

        def time(x: Tensor, mask: Tensor, config: RowGemvConfig | None) -> float:
            def run() -> None:
                for i in range(self.calls):
                    w = weights[i % copies]
                    if config is None:
                        F.linear(x, w)
                    else:
                        row_masked_gemv(x, w, mask, config)

            return _graph_ms(run, self.reps) / self.calls

        def best(x: Tensor, mask: Tensor, grid: list[RowGemvConfig]) -> tuple[RowGemvConfig, float]:
            timings: dict[RowGemvConfig, float] = {}
            for config in [default_row_config(x.shape[0]), *grid]:
                if config in timings:
                    continue
                try:
                    timings[config] = time(x, mask, config)
                except RuntimeError as error:  # out of resources for this tile, etc.
                    logger.debug("row GEMV config %s fails on %dx%d: %s", config, n, k, error)
            choice = min(timings, key=timings.__getitem__)
            return choice, timings[choice]

        shared = torch.rand(1, n, device=device, generator=gen) < 0.5
        x1 = torch.randn(1, k, device=device, generator=gen).to(weight.dtype)
        one, one_ms = best(x1, shared, _grid(*self.grid_one_row, k=k))
        one_speedup = time(x1, shared, None) / one_ms

        rows = self.probe_rows
        xm = torch.randn(rows, k, device=device, generator=gen).to(weight.dtype)
        mask = shared.expand(rows, n).contiguous()  # every row keeps the same channels: the kernel's best case
        many, many_ms = best(xm, mask, _grid(*self.grid_many_rows, k=k))
        many_speedup = time(xm, mask, None) / many_ms
        weights.clear()
        tuning = RowTuning(one, many if many_speedup > 1.0 else None, one_speedup, many_speedup)
        logger.info(
            "row GEMV autotune %dx%d %s: 1 row %s (%.2fx dense), %d rows %s (%.2fx dense)",
            n,
            k,
            weight.dtype,
            one,
            one_speedup,
            rows,
            tuning.many_rows,
            many_speedup,
        )
        return tuning
