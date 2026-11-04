"""The ``"triton_autotune"`` TEAL backend: the ``"triton"`` sparse GEMV with launch parameters timed on the device.

The heuristic of :func:`.teal_gemv.default_config` was tuned on an L40. On an A100 the same config runs the
Gemma-4-31B maps at 0.73-1.1x the speed of the dense cuBLAS GEMV at one decode row and 50% sparsity, while the best
config of a grid runs them at 1.44-1.87x (measured with a config sweep
over the Triton kernel). The best config differs by device and by map shape, so this backend times a grid
(:attr:`AutotuneTritonBackend.grid_one_row`) once per (weight shape, dtype, device name) when it is installed, and keeps
the fastest config for every map of that shape.

Calls with several rows use the ``tl.dot`` path of the kernel. They pay off only when the rows keep a common set of
channels. The backend times that best case (rows with one magnitude pattern, 50% sparsity) at
:attr:`AutotuneTritonBackend.probe_rows` rows against the dense matmul; when the dense matmul still wins, the map runs
the kernel for single-row calls only and every larger call takes the dense masked path. Numbers are those of the
``"triton"`` backend, which equal the ``"dense"`` backend up to bfloat16 accumulation order.

Timing replays a CUDA graph of several calls over weight copies that together exceed the L2 cache, so the result is
the bandwidth-bound decode time and not the CPU launch time. Tuning a 60-layer model with five map shapes takes
about a minute, mostly Triton compilation. A subclass changes the grids or the probe by overriding the class
attributes; :func:`clear_tuning_cache` forgets the stored choices.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .backends import TEAL_BACKENDS
from .teal_gemv import MAX_ROWS, GemvConfig, default_config, teal_gemv
from .triton_backend import TritonBackend, _KernelPatch

__all__ = ["AutotuneTritonBackend", "Tuning", "clear_tuning_cache"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tuning:
    """The tuned launch parameters of one weight shape.

    Attributes:
        one_row: Config of single-row calls.
        many_rows: Config of calls with 2..16 rows, or ``None`` when the dense matmul is faster there.
        one_row_speedup: Time of the dense matmul on the same (transposed) storage over the kernel's time, at one
            row and 50% sparsity. That dense call is the path the map takes when the kernel does not serve a call.
        many_rows_speedup: The same at :attr:`AutotuneTritonBackend.probe_rows` rows with shared channels.
    """

    one_row: GemvConfig
    many_rows: GemvConfig | None
    one_row_speedup: float
    many_rows_speedup: float


_CACHE: dict[tuple[object, ...], Tuning] = {}


def clear_tuning_cache() -> None:
    """Forget every stored :class:`Tuning`."""
    _CACHE.clear()


def _graph_ms(fn: Callable[[], object], reps: int) -> float:
    """Median ms of one replay of a CUDA graph of ``fn`` (after eager and graph warmup)."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    del graph
    return statistics.median(times)


def _grid(
    block_n: Sequence[int], block_k: Sequence[int], split_k: Sequence[int], num_warps: Sequence[int], k: int
) -> list[GemvConfig]:
    return [
        GemvConfig(bn, bk, split, warps, 2)
        for bn in block_n
        for bk in block_k
        for split in split_k
        for warps in num_warps
        if bn * bk <= 32768 and k // split >= bk
    ]


@TEAL_BACKENDS.register("triton_autotune")
class AutotuneTritonBackend(TritonBackend):
    """The ``"triton"`` backend with per-shape launch parameters timed on the device (see the module docstring).

    Attributes:
        grid_one_row: ``(block_n, block_k, split_k, num_warps)`` value lists of the single-row grid.
        grid_many_rows: The same for the ``tl.dot`` path (calls with several rows).
        probe_rows: Row count at which the ``tl.dot`` path is compared with the dense matmul.
        calls: Kernel calls per timed graph.
        reps: Graph replays per config (the median counts).
    """

    grid_one_row: tuple[tuple[int, ...], ...] = ((32, 64, 128, 256), (64, 128), (4, 8, 16, 32, 64), (2, 4))
    grid_many_rows: tuple[tuple[int, ...], ...] = ((64, 128), (32, 64), (2, 4, 8, 16, 32), (4,))
    probe_rows: int = 8
    calls: int = 8
    reps: int = 7

    def configure(self, patch: _KernelPatch) -> None:
        """Look up or time the :class:`Tuning` of the map's shape and set the patch's configs and row cutoff."""
        tuning = self.tune(patch.module.weight)
        configs = {1: tuning.one_row}
        if tuning.many_rows is None:
            patch.max_rows = min(patch.max_rows, 1)
        else:
            configs.update(dict.fromkeys(range(2, MAX_ROWS + 1), tuning.many_rows))
        patch.configs = configs

    def tune(self, weight: Tensor) -> Tuning:
        """The :class:`Tuning` of ``weight``'s shape, dtype and device, timed on first use and cached."""
        n, k = weight.shape
        key = (
            n,
            k,
            weight.dtype,
            torch.cuda.get_device_name(weight.device),
            self.grid_one_row,
            self.grid_many_rows,
            self.probe_rows,
        )
        if key not in _CACHE:
            with torch.no_grad():
                _CACHE[key] = self._time(weight.detach())
        return _CACHE[key]

    def _time(self, weight: Tensor) -> Tuning:
        n, k = weight.shape
        device = weight.device
        l2 = getattr(torch.cuda.get_device_properties(device), "L2_cache_size", 40 << 20)
        copies = max(1, min(4, -(-2 * l2 // (weight.numel() * weight.element_size()))))
        weights_t = [weight.t().contiguous() for _ in range(copies)]  # [k, n] row-major, as the patch stores it
        gen = torch.Generator(device=device).manual_seed(0)

        def time(x: Tensor, threshold: Tensor, config: GemvConfig | None) -> float:
            def run() -> None:
                for i in range(self.calls):
                    w = weights_t[i % copies]
                    if config is None:
                        F.linear(x, w.t())
                    else:
                        teal_gemv(x, w, threshold, config)

            return _graph_ms(run, self.reps) / self.calls

        def best(
            x: Tensor, threshold: Tensor, grid: list[GemvConfig], fallback: GemvConfig
        ) -> tuple[GemvConfig, float]:
            timings: dict[GemvConfig, float] = {}
            for config in [fallback, *grid]:
                if config in timings:
                    continue
                try:
                    timings[config] = time(x, threshold, config)
                except RuntimeError as error:  # out of resources for this tile, etc.
                    logger.debug("TEAL config %s fails on %dx%d: %s", config, n, k, error)
            choice = min(timings, key=timings.__getitem__)
            return choice, timings[choice]

        x1 = torch.randn(1, k, device=device, generator=gen).to(weight.dtype)
        t1 = torch.quantile(x1.float().abs(), 0.5).reshape(()).contiguous()
        one, one_ms = best(x1, t1, _grid(*self.grid_one_row, k=k), default_config(1, n, k, device))
        one_speedup = time(x1, t1, None) / one_ms

        rows = self.probe_rows
        magnitude = torch.randn(1, k, device=device, generator=gen).abs()
        signs = torch.randint(0, 2, (rows, k), device=device, generator=gen) * 2 - 1
        xm = (magnitude * signs).to(weight.dtype)  # every row keeps the same channels: the kernel's best case
        tm = torch.quantile(magnitude.float(), 0.5).reshape(()).contiguous()
        many, many_ms = best(xm, tm, _grid(*self.grid_many_rows, k=k), default_config(rows, n, k, device))
        many_speedup = time(xm, tm, None) / many_ms
        weights_t.clear()  # free the copies before the next shape
        tuning = Tuning(one, many if many_speedup > 1.0 else None, one_speedup, many_speedup)
        logger.info(
            "TEAL autotune %dx%d %s: 1 row %s (%.2fx dense), %d rows %s (%.2fx dense)",
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
