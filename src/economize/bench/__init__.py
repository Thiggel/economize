"""Benchmarking: wall-clock timing, standard prompt suites and an analytic cost model.

* :mod:`~economize.bench.timing` times model calls and generations with warmup and repeated measurements (CUDA events
  on one device, synchronised host clocks on several) and reports medians and throughput.
* :mod:`~economize.bench.prompts` loads Spec-Bench, GSM8K and HumanEval prompts through ``datasets``; further suites
  are added by registration in :data:`PROMPT_SOURCES`.
* :mod:`~economize.bench.cost_model` prices the cost log of a generation (FLOPs and bytes per call, a roofline time per
  call on an accelerator preset) and compares it with plain decoding of the same output. Model families whose
  configs need special reading register a reader in :data:`SHAPE_READERS`.

A typical speed measurement::

    from economize.bench import benchmark_generation, load_prompts

    prompts = load_prompts("spec-bench", limit=80).tokenize(tokenizer, chat=True)
    plain = benchmark_generation(lambda ids: model.generate(ids[None].cuda(), max_new_tokens=256)[0, ids.numel():],
                                 prompts)
    fast = benchmark_generation(lambda ids: model.decode(ids.cuda(), 256), prompts)
    print(fast.speedup_over(plain))
"""

from .cost_model import (
    HARDWARE,
    SHAPE_READERS,
    CallCost,
    CallSpec,
    CostModel,
    CostReport,
    Hardware,
    ModelShape,
    ShapeReader,
    get_hardware,
)
from .prompts import PROMPT_SOURCES, SPEC_BENCH_URL, PromptSet, load_prompts
from .timing import GenerationTiming, PromptTiming, Timing, benchmark_generation, count_tokens, time_call

__all__ = [
    "HARDWARE",
    "PROMPT_SOURCES",
    "SHAPE_READERS",
    "SPEC_BENCH_URL",
    "CallCost",
    "CallSpec",
    "CostModel",
    "CostReport",
    "GenerationTiming",
    "Hardware",
    "ModelShape",
    "PromptSet",
    "PromptTiming",
    "ShapeReader",
    "Timing",
    "benchmark_generation",
    "count_tokens",
    "get_hardware",
    "load_prompts",
    "time_call",
]
