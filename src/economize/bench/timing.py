"""Wall-clock timing of model calls and generations: warmup, repeated measurements, medians.

On a single CUDA device a call is timed with CUDA events around it (the host may run ahead of the device, the events
do not); on several devices or on the CPU every device is synchronised and the host clock is read. Both report the
time from launching the call until all its work has finished.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch

from ..errors import ConfigError

__all__ = ["GenerationTiming", "PromptTiming", "Timing", "benchmark_generation", "count_tokens", "time_call"]


@dataclass(frozen=True)
class Timing:
    """Repeated measurements of one call, in seconds."""

    samples: tuple[float, ...]

    @property
    def median(self) -> float:
        return statistics.median(self.samples)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.samples)

    @property
    def minimum(self) -> float:
        return min(self.samples)

    @property
    def spread(self) -> float:
        """Interquartile range (0 for fewer than two samples)."""
        if len(self.samples) < 2:
            return 0.0
        low, _, high = statistics.quantiles(self.samples, n=4, method="inclusive")
        return high - low

    def to_dict(self) -> dict[str, Any]:
        return {"median": self.median, "mean": self.mean, "min": self.minimum, "iqr": self.spread,
                "samples": list(self.samples)}


def _devices(devices: Iterable[torch.device | str | int] | None) -> list[torch.device]:
    if devices is None:
        return [torch.device("cuda", torch.cuda.current_device())] if torch.cuda.is_available() else []
    result = []
    for device in devices:
        device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
        if device.type == "cuda":
            result.append(device if device.index is not None else torch.device("cuda", torch.cuda.current_device()))
    return list(dict.fromkeys(result))


def time_call(fn: Callable[[], Any], *, warmup: int = 1, repeats: int = 5,
              devices: Iterable[torch.device | str | int] | None = None) -> tuple[Timing, Any]:
    """Time ``fn()`` ``repeats`` times after ``warmup`` untimed calls.

    Args:
        fn: The call to time.
        warmup: Untimed calls first (compilation, CUDA graph capture, allocator growth).
        repeats: Timed calls.
        devices: The devices ``fn`` computes on (default: the current CUDA device, if any). CUDA events time a call
            on one device; with several the host clock times it after synchronising all of them. A call that
            computes on two GPUs, such as :class:`~economize.distributed.AsyncSpeculativeDecoder` with its drafter on
            a second device, must name both (``devices=[target_device, drafter_device]``): the default times only
            the current device and misses work still queued on the other.

    Returns:
        The timing and the result of the last call.
    """
    if repeats < 1 or warmup < 0:
        raise ConfigError(f"need repeats >= 1 and warmup >= 0, got {repeats}, {warmup}")
    cuda = _devices(devices)
    result = None
    for _ in range(warmup):
        result = fn()
    samples = []
    for _ in range(repeats):
        for device in cuda:
            torch.cuda.synchronize(device)
        if len(cuda) == 1:
            with torch.cuda.device(cuda[0]):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                result = fn()
                end.record()
                end.synchronize()
            samples.append(start.elapsed_time(end) / 1000)
        else:
            began = time.perf_counter()
            result = fn()
            for device in cuda:
                torch.cuda.synchronize(device)
            samples.append(time.perf_counter() - began)
    return Timing(tuple(samples)), result


def count_tokens(output: Any) -> int:
    """Generated tokens in a generation's output: an int, a token tensor (all its elements, so a ``[B, N]`` batch
    counts ``B * N`` including any padding), a sequence of token ids, an object with a ``tokens`` attribute
    (``DecodeResult``, ``SSDResult``) or a sequence of such outputs (the results of ``decode_batch``), summed.

    Raises:
        TypeError: The output is text or of another type whose tokens cannot be counted.
    """
    if isinstance(output, int):
        return output
    tokens = getattr(output, "tokens", output)
    if isinstance(tokens, torch.Tensor):
        return int(tokens.numel())
    if isinstance(tokens, Sequence) and not isinstance(tokens, (str, bytes)):
        if all(isinstance(token, int) for token in tokens):
            return len(tokens)
        return sum(count_tokens(item) for item in tokens)
    raise TypeError(f"cannot count the tokens of {type(output).__name__}")


@dataclass(frozen=True)
class PromptTiming:
    """One prompt: tokens generated and the timing of its generation."""

    tokens: int
    timing: Timing


@dataclass(frozen=True)
class GenerationTiming:
    """A generation benchmark over a prompt set.

    Throughput is total tokens over the sum of the per-prompt medians, so long generations weigh more, as they do
    for a user.
    """

    prompts: tuple[PromptTiming, ...] = field(default_factory=tuple)

    @property
    def tokens(self) -> int:
        return sum(p.tokens for p in self.prompts)

    @property
    def seconds(self) -> float:
        return sum(p.timing.median for p in self.prompts)

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 else float("inf")

    @property
    def seconds_per_token(self) -> float:
        return self.seconds / max(1, self.tokens)

    def speedup_over(self, baseline: GenerationTiming) -> float:
        """Throughput relative to ``baseline`` (above 1 is faster).

        Both timings must cover the same prompts (checked by their number); the token counts may differ, e.g. for a
        lossy method that stops at another place.

        Raises:
            ConfigError: The two timings have different numbers of prompts.
        """
        if len(self.prompts) != len(baseline.prompts):
            raise ConfigError(f"speedup_over compares timings of the same prompts, got {len(self.prompts)} prompts "
                              f"against a baseline of {len(baseline.prompts)}")
        return self.tokens_per_second / baseline.tokens_per_second

    def to_dict(self) -> dict[str, Any]:
        return {"tokens": self.tokens, "seconds": self.seconds, "tokens_per_second": self.tokens_per_second,
                "prompts": [{"tokens": p.tokens, **p.timing.to_dict()} for p in self.prompts]}


def benchmark_generation(generate: Callable[[torch.Tensor], Any], prompts: Sequence[torch.Tensor], *,
                         warmup: int = 1, repeats: int = 3, warmup_per_prompt: int = 0,
                         devices: Iterable[torch.device | str | int] | None = None) -> GenerationTiming:
    """Time ``generate(prompt)`` on every prompt.

    Args:
        generate: Runs one generation and returns its output (see :func:`count_tokens`), e.g.
            ``lambda ids: model.decode(ids, 256)``.
        prompts: Token ids per prompt.
        warmup: Untimed generations of the first prompt before the first measurement.
        repeats: Timed generations per prompt (the median counts).
        warmup_per_prompt: Untimed generations of every prompt before its measurements, for decoders that compile,
            capture CUDA graphs or grow buffers per prompt length (the median of three or more repeats also drops
            one such slow run).
        devices: The devices the generation computes on (see :func:`time_call`). Name every GPU of a multi-device
            generation, e.g. ``devices=[target_device, drafter_device]`` for SSD.

    Raises:
        ConfigError: No prompts, or a negative warmup.
    """
    if not prompts:
        raise ConfigError("benchmark_generation needs at least one prompt")
    if warmup < 0 or warmup_per_prompt < 0:
        raise ConfigError(f"need warmup >= 0 and warmup_per_prompt >= 0, got {warmup}, {warmup_per_prompt}")
    devices = list(devices) if devices is not None else None
    for _ in range(warmup):
        generate(prompts[0])
    timings = []
    for prompt in prompts:
        timing, output = time_call(partial(generate, prompt), warmup=warmup_per_prompt, repeats=repeats,
                                   devices=devices)
        timings.append(PromptTiming(count_tokens(output), timing))
    return GenerationTiming(tuple(timings))
