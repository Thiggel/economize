"""The timing harness: warmup and repeats, medians, token counting and throughput, on the CPU and with CUDA events."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
import torch

from economize.bench import GenerationTiming, PromptTiming, Timing, benchmark_generation, count_tokens, time_call
from economize.errors import ConfigError
from helpers import VOCAB, make_tiny_model


def test_timing_statistics() -> None:
    timing = Timing((0.4, 0.1, 0.3, 0.2, 1.0))
    assert timing.median == 0.3 and timing.minimum == 0.1 and timing.mean == pytest.approx(0.4)
    assert timing.spread == pytest.approx(0.2)  # quartiles 0.2 and 0.4
    assert Timing((0.5,)).spread == 0
    assert timing.to_dict()["samples"] == [0.4, 0.1, 0.3, 0.2, 1.0]


def test_time_call_warms_up_and_repeats() -> None:
    calls = []

    def work() -> int:
        calls.append(None)
        time.sleep(0.01)
        return len(calls)

    timing, last = time_call(work, warmup=2, repeats=3, devices=["cpu"])
    assert len(calls) == 5 and last == 5 and len(timing.samples) == 3
    assert timing.minimum >= 0.009


def test_time_call_rejects_bad_counts() -> None:
    with pytest.raises(ConfigError):
        time_call(lambda: None, repeats=0)
    with pytest.raises(ConfigError):
        time_call(lambda: None, warmup=-1)


def test_count_tokens() -> None:
    assert count_tokens(7) == 7
    assert count_tokens(torch.zeros(1, 9)) == 9 and count_tokens(torch.tensor(3)) == 1
    assert count_tokens(SimpleNamespace(tokens=torch.arange(4))) == 4
    assert count_tokens([1, 2, 3]) == 3
    with pytest.raises(TypeError):
        count_tokens(1.5)


def test_benchmark_generation() -> None:
    seen = []

    def generate(prompt: torch.Tensor) -> torch.Tensor:
        seen.append(prompt.numel())
        return torch.zeros(2 * prompt.numel())

    prompts = [torch.arange(3), torch.arange(5)]
    result = benchmark_generation(generate, prompts, warmup=1, repeats=2, devices=["cpu"])
    assert seen == [3, 3, 3, 5, 5]
    assert result.tokens == 16 and [p.tokens for p in result.prompts] == [6, 10]
    assert result.seconds == pytest.approx(sum(p.timing.median for p in result.prompts))
    assert result.tokens_per_second == pytest.approx(16 / result.seconds)
    json.dumps(result.to_dict())
    with pytest.raises(ConfigError):
        benchmark_generation(generate, [])


def test_speedup_over() -> None:
    slow = GenerationTiming((PromptTiming(10, Timing((2.0,))),))
    fast = GenerationTiming((PromptTiming(20, Timing((1.0,))),))
    assert fast.speedup_over(slow) == pytest.approx(4.0)
    assert slow.seconds_per_token == pytest.approx(0.2)
    assert GenerationTiming().tokens_per_second == float("inf")


def test_times_a_real_generation() -> None:
    model = make_tiny_model()
    prompts = [torch.randint(0, VOCAB, (8,), generator=torch.Generator().manual_seed(s)) for s in range(2)]

    def generate(ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model.generate(ids[None], max_new_tokens=6, do_sample=False)[0, ids.numel():]

    result = benchmark_generation(generate, prompts, repeats=2, devices=[])
    assert result.tokens == 12 and result.tokens_per_second > 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cuda_events_and_several_devices() -> None:
    a = torch.randn(1024, 1024, device="cuda")
    timing, result = time_call(lambda: a @ a, warmup=1, repeats=5)
    assert result.device.type == "cuda" and 0 < timing.median < 1
    if torch.cuda.device_count() > 1:
        b = torch.randn(1024, 1024, device="cuda:1")
        timing, _ = time_call(lambda: (a @ a, b @ b), repeats=3, devices=[0, 1])
        assert len(timing.samples) == 3 and timing.minimum > 0


def test_benchmark_generation_can_warm_up_every_prompt() -> None:
    """A decoder that compiles or captures per prompt shape pays that once per new prompt length; ``warmup_per_prompt``
    keeps it out of the timed repeats (with ``repeats=2`` the median would include it)."""
    from economize.bench import benchmark_generation

    seen: list[int] = []
    prompts = [torch.arange(3), torch.arange(5)]
    benchmark_generation(lambda ids: seen.append(ids.numel()) or ids, prompts, warmup=1, warmup_per_prompt=1,
                         repeats=2, devices=["cpu"])
    assert seen == [3, 3, 3, 3, 5, 5, 5]


def test_speedup_over_refuses_timings_of_different_prompt_sets() -> None:
    """A throughput ratio is a speedup only when both timings ran the same prompts."""
    from economize.bench import GenerationTiming, PromptTiming, Timing
    from economize.errors import ConfigError

    one = GenerationTiming((PromptTiming(10, Timing((2.0,))),))
    two = GenerationTiming((PromptTiming(10, Timing((0.5,))), PromptTiming(10, Timing((0.5,)))))
    with pytest.raises(ConfigError, match="prompt"):
        two.speedup_over(one)
