"""Measured sublayer latencies of KnapSpec: ``t_mlp`` and ``t_attn(n) = slope * n + base`` on the decoding GPU.

KnapSpec weighs every sublayer by its latency in a one-token draft step. The paper measures the MLP once and the
attention at several context lengths per hardware and fits the attention linearly in the context length.
:func:`measure_latency` does the same on the model as it decodes: it times layer calls with only the attention (or
only the MLP) of a ``[B, 1, d]`` query, replayed from a CUDA graph when it can be captured (the draft steps replay
graphs too), else eagerly between CUDA events (``time.perf_counter`` on the CPU). The query writes its K/V to the
cache's trash slot, so the cache keeps its contents. The fit averages the attention over the distinct attention
patterns of the model (Gemma's sliding and global layers) weighted by how many layers use each.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...runtime.context import ContextLayout
from ...runtime.graphs import capture

if TYPE_CHECKING:
    from ...runtime import KVCache, LayerCall, LayerRunner
    from .sublayer import SublayerSkipper

__all__ = ["SublayerLatency", "measure_latency"]


@dataclass(frozen=True)
class SublayerLatency:
    """Latency of one sublayer in a one-token draft step, in seconds.

    Attributes:
        mlp: Latency of an MLP sublayer (its norm, the MLP and the residual add).
        attention_slope: Growth of an attention sublayer's latency per token of context.
        attention_base: Latency of an attention sublayer at context length 0.
    """

    mlp: float
    attention_slope: float
    attention_base: float

    def attention(self, length: float) -> float:
        """Latency of an attention sublayer at context length ``length``."""
        return self.attention_slope * length + self.attention_base

    def as_tuple(self) -> tuple[float, float, float]:
        """``(mlp, attention_slope, attention_base)``, the form of :attr:`KnapSpecConfig.latency`."""
        return self.mlp, self.attention_slope, self.attention_base


def _timer(fn: Callable[[], object], repeats: int, device: torch.device) -> float:
    """Seconds per call of ``fn`` (already warmed up).

    On CUDA the calls are recorded into one graph through :func:`economize.runtime.graphs.capture` (the safeguards
    of the decoder's own captures: no garbage collection while recording, a quiet device before and after) in a
    private pool, which is destroyed before returning once the device is idle.
    """
    if device.type != "cuda":
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        return (time.perf_counter() - start) / repeats
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream(device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    run: Callable[[], object]
    try:
        capture(graph, lambda: [fn() for _ in range(repeats)], device)
        run, calls = graph.replay, repeats
    except RuntimeError:  # a layer that cannot be captured: time it eagerly
        run, calls = fn, 1
    try:
        run()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        begin.record()
        for _ in range(repeats if calls == 1 else 3):
            run()
        end.record()
        torch.cuda.synchronize(device)
        return begin.elapsed_time(end) / 1e3 / (repeats if calls == 1 else 3 * repeats)
    finally:
        del run, graph
        torch.cuda.synchronize(device)


def measure_latency(
    runner: LayerRunner, cache: KVCache, skipper: SublayerSkipper, lengths: Sequence[int], repeats: int
) -> SublayerLatency:
    """Time the sublayers of the model behind ``runner`` with ``cache`` (whose buffers exist: after a prefill).

    Context lengths beyond the cache's capacity are dropped (at least two must remain, else the attention is
    taken as flat). Every timed call is one layer on a ``[B, 1, d]`` query, ``B`` the cache's active batch size.
    """
    adapter = runner.adapter
    L = adapter.num_layers
    batch = cache.active_batch_size
    device = torch.device(adapter.device)
    attention_layers = [i for i in range(1, L + 1) if skipper.skippable[2 * (i - 1)]]
    mlp_layers = [i for i in range(1, L + 1) if skipper.skippable[2 * i - 1]]
    by_spec = Counter(adapter.attention_spec(i) for i in attention_layers)
    probes = {spec: next(i for i in attention_layers if adapter.attention_spec(i) == spec) for spec in by_spec}
    usable = sorted({min(int(n), cache.capacity) for n in lengths})

    def timed(index: int, call: LayerCall, length: int) -> float:
        positions = torch.full((batch, 1), length - 1, dtype=torch.long, device=device)
        layout = ContextLayout(positions, torch.full_like(positions, cache.trash_slot))
        ctx = runner.context_from_layout(cache, layout)
        hidden = runner.embed(torch.zeros(batch, 1, dtype=torch.long, device=device), ctx)
        hidden = hidden + 0.1 * torch.randn_like(hidden)

        def fn() -> torch.Tensor:
            return runner.run(hidden, ctx, index, index, overrides={index: call})[0]

        fn()  # builds the context's mask and rotary tables outside any capture
        return _timer(fn, repeats, device)

    mlp = (sum(timed(mlp_layers[0], skipper.mlp_only(mlp_layers[0]), length) for length in usable) / len(usable)
           if mlp_layers else 0.0)
    if not probes:
        return SublayerLatency(max(mlp, 1e-9), 0.0, max(mlp, 1e-9))
    points = []
    for length in usable:
        total = sum(count * timed(probes[spec], skipper.attention_only(probes[spec]), length)
                    for spec, count in by_spec.items())
        points.append((length, total / sum(by_spec.values())))
    if len(points) < 2:
        return SublayerLatency(max(mlp, 1e-9), 0.0, max(points[0][1], 1e-9))
    x = torch.tensor([p[0] for p in points], dtype=torch.float64)
    y = torch.tensor([p[1] for p in points], dtype=torch.float64)
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
    slope = max(slope, 0.0)
    base = float(y.mean() - slope * x.mean())
    return SublayerLatency(max(mlp, 1e-9), slope, max(base, 1e-9))
