"""Time drafters against plain decoding on a standard prompt suite, with modelled costs.

    python examples/benchmark.py --drafter runs/eagle3/final --drafter lookup --prompts spec-bench --cuda-graphs

Each ``--drafter`` is an economize checkpoint (local or Hub) or the name of a training-free drafter (``lookup``).
Reports the measured speedup over ``transformers``' greedy ``generate``, the accepted drafts per cycle and the
cost that ``economize.bench.CostModel`` assigns to the logged calls on a given GPU. A drafter with a network of its
own (an EAGLE-3 layer, a draft model) is priced as decoder layers of the target's width, as many as its calls run.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import economize
from economize.bench import CallSpec, CostModel, ModelShape, benchmark_generation, load_prompts
from economize.decoding import CostLog
from economize.techniques.lookup import LookupConfig

TRAINING_FREE = {"lookup": lambda: LookupConfig(max_ngram=3, max_proposal=10)}


def module_costs(config: Any, logs: list[CostLog]) -> CostModel:
    """A cost model of ``config`` that prices every separate network in ``logs`` as layers of the target's width."""
    depth: dict[str, int] = {}
    for log in logs:
        for event in log.events:
            if event.module is not None:
                depth[event.module] = max(depth.get(event.module, 1), *(event.layers or (event.last_layer,)))
    shape = ModelShape.from_config(config)
    modules = {name: CallSpec(shape=replace(shape, num_layers=layers, layer_types=shape.layer_types[:layers],
                                            layer_parts=shape.layer_parts[:layers]))
               for name, layers in depth.items()}
    return CostModel(shape, modules=modules)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--drafter", action="append", default=[], help="repeatable (default: lookup)")
    parser.add_argument("--prompts", default="gsm8k", choices=["gsm8k", "humaneval", "spec-bench"])
    parser.add_argument("--limit", type=int, default=20, help="number of prompts")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--cycle-length", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cuda-graphs", action="store_true", help="needs a CUDA device")
    parser.add_argument("--hardware", default="a100-80gb", help="GPU of the cost model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [p.to(args.device) for p in load_prompts(args.prompts, limit=args.limit).tokenize(tokenizer, chat=False)]
    n = args.max_new_tokens

    plain = benchmark_generation(lambda p: base.generate(p[None], max_new_tokens=n, do_sample=False)[0, p.numel():],
                                 prompts, repeats=args.repeats)
    print(f"{'drafter':32s} {'speedup':>8s} {'accepted':>9s} {'modelled':>9s}  (modelled: roofline on {args.hardware})")
    print(f"{'none (transformers generate)':32s} {1.0:8.2f}")

    decoding = economize.DecodeConfig(cycle_length=args.cycle_length, cuda_graphs=args.cuda_graphs)
    for name in args.drafter or ["lookup"]:
        if name in TRAINING_FREE:
            model = economize.apply(base, TRAINING_FREE[name](), decoding=decoding)
        else:
            model = economize.load(base, name, decoding=decoding)
        timing = benchmark_generation(lambda p, model=model: model.decode(p, n).tokens, prompts, repeats=args.repeats,
                                      warmup_per_prompt=1 if args.cuda_graphs else 0)
        results = [model.decode(p, n) for p in prompts]
        costs = module_costs(base.config, [r.cost for r in results])
        accepted = sum(r.mean_accepted for r in results) / len(results)
        modelled = sum(costs.relative(r.cost, r.prompt_length, len(r.tokens), hardware=args.hardware)["speedup"]
                       for r in results) / len(results)
        print(f"{name:32s} {timing.speedup_over(plain):8.2f} {accepted:9.2f} {modelled:9.2f}")
        model.detach()


if __name__ == "__main__":
    main()
