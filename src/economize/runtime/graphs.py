"""CUDA graph capture of fixed-shape steps (draft steps, single-token verification).

A decoding step launches hundreds of small kernels; replaying a captured CUDA graph removes the launch overhead.
:class:`StaticStep` wraps a function of tensors and keeps one graph per shape bucket::

    step = StaticStep(lambda tokens, positions: draft_step(tokens, positions))
    logits = step(tokens, positions)  # eager twice, then captured, then replayed

The wrapped function must be free of host synchronization (``.item()``, ``.tolist()``, data-dependent shapes) and
must allocate its persistent state (cache buffers) before capture; the warm-up calls take care of lazy allocation.
On CPU, or with ``enabled=False``, the function runs eagerly.

Steps that share a memory pool and drop their graphs together (every graph of a decoder when its KV cache is rebuilt)
are created by a :class:`GraphPool`, which renews the pool on each reset.
"""

from __future__ import annotations

import contextlib
import gc
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch

__all__ = ["GraphPool", "StaticStep", "capture"]

_Key = tuple[tuple[tuple[int, ...], torch.dtype, torch.device], ...]


def capture(graph: torch.cuda.CUDAGraph, fn: Callable[[], Any], device: torch.device, *, pool: Any = None) -> Any:
    """Record ``fn()`` into ``graph`` on a fresh stream of ``device`` and return its output (the graph's buffers).

    Every capture of the library goes through here, so all of them share the same safeguards:

    * the device is synchronized first, so no kernel of earlier work still reads memory the capture reuses;
    * torch's default capture stream is created once, on the device of the first capture in the process: a stream
      of ``device`` keeps graphs of models on other devices from coming out empty;
    * torch.cuda.graph collects garbage before the capture. A collection during it (triggered by the Python objects
      the captured function allocates) can free CUDA memory of unrelated dead objects, e.g. the graph pool of an
      earlier decoder, and a cudaFree (an unmap under ``expandable_segments``) invalidates the capture ("Invalid
      stream capture status"), so the collector is off while recording;
    * under torch.distributed the NCCL watchdog thread keeps calling the CUDA API (event queries) while the captured
      collectives are recorded. The default "global" mode turns those calls into capture errors or a hang, so a
      split model captures in "thread_local" mode (as Megatron does);
    * a capture that fails is followed by a synchronization before the error propagates, so that a caller falling
      back to eager execution starts from a quiet device.

    Args:
        graph: The (empty) graph to record into.
        fn: The work to record, a function without arguments.
        device: The CUDA device the work runs on.
        pool: Memory pool shared with other graphs (``torch.cuda.graph_pool_handle()``); ``None`` gives the graph a
            private pool.

    Returns:
        What ``fn()`` returned during the capture.
    """
    torch.cuda.synchronize(device)
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    mode = "thread_local" if distributed else "global"
    enabled = gc.isenabled()
    gc.disable()
    try:
        with torch.cuda.graph(graph, pool=pool, stream=torch.cuda.Stream(device), capture_error_mode=mode):
            return fn()
    except BaseException:
        with contextlib.suppress(RuntimeError):  # a sticky CUDA error would hide the capture's own error
            torch.cuda.synchronize(device)
        raise
    finally:
        if enabled:
            gc.enable()


@dataclass
class _Captured:
    graph: torch.cuda.CUDAGraph
    inputs: tuple[torch.Tensor, ...]
    output: Any


class StaticStep:
    """A function of tensors replayed from a CUDA graph per distinct input signature (shapes, dtypes, devices).

    Outputs of a replay are the graph's static buffers: they are overwritten by the next call with the same
    signature, so callers clone what they keep.

    Args:
        fn: The step. Positional tensor arguments only; everything else goes into the closure.
        warmup: Eager calls per signature before capture (lazy allocations happen here).
        enabled: ``False`` always runs ``fn`` eagerly (debugging, CPU).
        pool: Memory pool shared with other graphs (``torch.cuda.graph_pool_handle()``).
    """

    def __init__(self, fn: Callable[..., Any], *, warmup: int = 2, enabled: bool = True, pool: Any = None) -> None:
        self.fn = fn
        self.warmup = warmup
        self.enabled = enabled
        self.pool = pool
        self._graphs: dict[_Key, _Captured] = {}
        self._calls: dict[_Key, int] = {}

    def __call__(self, *inputs: torch.Tensor) -> Any:
        if not self.enabled or not inputs or inputs[0].device.type != "cuda":
            return self.fn(*inputs)
        key: _Key = tuple((tuple(t.shape), t.dtype, t.device) for t in inputs)
        captured = self._graphs.get(key)
        if captured is not None:
            for static, value in zip(captured.inputs, inputs):
                static.copy_(value)
            captured.graph.replay()
            return captured.output
        calls = self._calls.get(key, 0)
        self._calls[key] = calls + 1
        if calls < self.warmup:
            return self._side_stream(self.fn, *inputs)
        return self._capture(key, inputs)

    def reset(self) -> None:
        """Drop every captured graph (after the wrapped function's persistent state was reallocated)."""
        self._graphs.clear()
        self._calls.clear()

    @property
    def num_graphs(self) -> int:
        return len(self._graphs)

    def _capture(self, key: _Key, inputs: tuple[torch.Tensor, ...]) -> Any:
        static = tuple(t.clone() for t in inputs)
        graph = torch.cuda.CUDAGraph()
        output = capture(graph, lambda: self.fn(*static), inputs[0].device, pool=self.pool)
        self._graphs[key] = _Captured(graph, static, output)
        graph.replay()
        return output

    @staticmethod
    def _side_stream(fn: Callable[..., Any], *inputs: torch.Tensor) -> Any:
        stream = torch.cuda.Stream(inputs[0].device)
        stream.wait_stream(torch.cuda.current_stream(inputs[0].device))
        with torch.cuda.stream(stream):
            output = fn(*inputs)
        torch.cuda.current_stream(inputs[0].device).wait_stream(stream)
        return output


class GraphPool:
    """The steps of one owner that share a CUDA graph memory pool and drop their graphs together.

    :meth:`reset` drops every graph and switches the steps to a fresh pool. Recording into the old pool is not
    possible: once all its graphs are destroyed, torch keeps the pool only while a tensor allocated in it is still
    alive (a replay output held by a caller), and a capture into such a pool fails an internal assertion
    (``use_count > 0`` in ``CUDACachingAllocator``). The old pool's memory is returned when its last tensor dies.

    Args:
        device: Device the pool handles are created on (``None``: the current device).
        enabled: ``False`` (CPU, debugging) makes every step run eagerly and creates no pool.
    """

    def __init__(self, device: torch.device | None = None, *, enabled: bool = True) -> None:
        self.device = device
        self.enabled = enabled
        self.steps: list[StaticStep] = []
        self.handle: Any = self._new_handle()

    def step(self, fn: Callable[..., Any], **kwargs: Any) -> StaticStep:
        """``fn`` as a :class:`StaticStep` on this pool (keyword arguments as for :class:`StaticStep`)."""
        step = StaticStep(fn, pool=self.handle, enabled=self.enabled and kwargs.pop("enabled", True), **kwargs)
        self.steps.append(step)
        return step

    def reset(self) -> None:
        """Drop the graphs of every step and record the next ones into a fresh pool."""
        for step in self.steps:
            step.reset()
        self.handle = self._new_handle()
        for step in self.steps:
            step.pool = self.handle

    @property
    def num_graphs(self) -> int:
        """Graphs recorded so far (over all steps)."""
        return sum(step.num_graphs for step in self.steps)

    def _new_handle(self) -> Any:
        if not self.enabled:
            return None
        with self._on_device():
            return torch.cuda.graph_pool_handle()

    @contextlib.contextmanager
    def _on_device(self) -> Iterator[None]:
        if self.device is None or self.device.type != "cuda":
            yield
            return
        with torch.cuda.device(self.device):
            yield
