"""Driver of the vLLM GPU tests, run in a subprocess (vLLM owns the GPU and starts its own processes).

    python vllm_driver.py export MODEL OUT ADAPTATION [RUN]   # export a jump drafter (RUN: a trained economize run)
    python vllm_driver.py generate MODEL OUT.json [DRAFTER K] [--eager] [--dtype D] [--max-tokens N]

``generate`` decodes :data:`PROMPTS` greedily, with speculative decoding when a drafter is given, and writes the
token ids, the decoding speed and vLLM's speculative decoding counters as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

PROMPTS = [
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips to her "
    "friends in May. How many clips did Natalia sell altogether in April and May?\nAnswer:",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "Photosynthesis is the process by which",
    "In machine learning, overfitting means",
]


def export(model_name: str, out: str, adaptation: str, run: str | None) -> None:
    import torch
    import transformers

    import economize
    from economize.backends.vllm import export_drafter

    hf = transformers.AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    if run:
        model = economize.load(hf, run)
    else:  # an untrained drafter; small noise makes the LoRA updates and the layer copies differ from the target
        layers = hf.config.num_hidden_layers
        config = economize.JumpConfig(target_layer=layers // 2, source_layer=layers // 2 + 2,
                                   exit_layer=layers // 2 + 3, adaptation=adaptation, lora_rank=4)
        model = economize.apply(hf, config)
        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for name, param in model.techniques["jump"].named_parameters():
                if name.startswith(("lora.", "blocks.")) and param.dim() == 2:
                    param.add_(torch.randn(param.shape, generator=generator) * 1e-3)
    export_drafter(model, out)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    from economize.backends.vllm import speculative_config

    options: dict[str, Any] = dict(
        dtype=args.dtype, enforce_eager=args.eager, max_model_len=1024, seed=0, disable_log_stats=False,
        gpu_memory_utilization=float(os.environ.get("ECONOMIZE_VLLM_GPU_UTIL", "0.4")),
    )
    if args.drafter:
        options["speculative_config"] = speculative_config(args.drafter, args.k)
    llm = LLM(args.model, **options)
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True)
    llm.generate(PROMPTS[:1], SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)  # warm-up
    start = time.perf_counter()
    outputs = llm.generate(PROMPTS, params, use_tqdm=False)
    seconds = time.perf_counter() - start
    counters: dict[str, Any] = {}
    for metric in llm.get_metrics():
        if "spec_decode" in metric.name:
            counters[metric.name.split(":")[-1]] = list(metric.values) if hasattr(metric, "values") else metric.value
    ids = [list(output.outputs[0].token_ids) for output in outputs]
    return {"ids": ids, "seconds": seconds, "tokens_per_second": sum(map(len, ids)) / seconds, "spec": counters}


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    exporting = commands.add_parser("export")
    exporting.add_argument("model")
    exporting.add_argument("out")
    exporting.add_argument("adaptation")
    exporting.add_argument("run", nargs="?")
    generating = commands.add_parser("generate")
    generating.add_argument("model")
    generating.add_argument("out")
    generating.add_argument("drafter", nargs="?")
    generating.add_argument("k", nargs="?", type=int, default=3)
    generating.add_argument("--eager", action="store_true")
    generating.add_argument("--dtype", default="float32")
    generating.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()
    if args.command == "export":
        export(args.model, args.out, args.adaptation, args.run)
    else:
        with open(args.out, "w") as handle:
            json.dump(generate(args), handle)


if __name__ == "__main__":
    main()
