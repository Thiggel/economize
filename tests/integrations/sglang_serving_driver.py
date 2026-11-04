"""Driver of the SGLang serving GPU tests, run in a subprocess (SGLang owns the GPU).

    python sglang_serving_driver.py RECIPE OUT.json [--model M] [--dtype D] [--k K] [--max-tokens N]
                                    [--run RUN] [--workdir DIR] [--attention-backend B] [--assistant A]

A recipe builds an EfficientModel on ``--model`` (:data:`RECIPES`), :func:`~economize.backends.sglang.serving.to_sglang`
writes it to ``--workdir/RECIPE``, and an offline ``sglang.Engine`` decodes :data:`PROMPTS` greedily with the plan.
The JSON holds the plan, the token ids, the :data:`TOP` most likely tokens at every position (the near-tie check of
the tests) and SGLang's speculative counters (``spec_verify_ct``, ``spec_accept_length``).
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

PROMPTS = [
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips to her "
    "friends in May. How many clips did Natalia sell altogether in April and May?\nAnswer:",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "Photosynthesis is the process by which",
    "In machine learning, overfitting means",
    "The French Revolution began in 1789 when",
    "Here is a recipe for a simple tomato soup:\n1.",
    "import numpy as np\n\ndef softmax(x):\n",
    "The three laws of thermodynamics state that",
]

#: Number of most likely tokens recorded per position.
TOP = 5


def _perturbed(technique: torch.nn.Module, std: float) -> None:
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in technique.parameters():
            if parameter.requires_grad or parameter.dim() == 2:
                parameter.add_(torch.randn(parameter.shape, generator=generator).to(parameter) * std)


def _recipes() -> dict[str, Callable[[Any, argparse.Namespace], Any]]:
    import economize
    from economize.techniques.draft_model.config import DraftModelConfig
    from economize.techniques.eagle3 import Eagle3Config
    from economize.techniques.early_exit import EarlyExitConfig
    from economize.techniques.jump import JumpConfig
    from economize.techniques.kv_quant.fp8 import FP8KVConfig
    from economize.techniques.lookup.config import LookupConfig
    from economize.techniques.quant.config import QuantConfig

    def plain(base: Any, args: argparse.Namespace) -> Any:
        return economize.apply(base)

    def drafter(name: str, config: Any, std: float = 0.02, *extra: Any) -> Callable[[Any, argparse.Namespace], Any]:
        def build(base: Any, args: argparse.Namespace) -> Any:
            model = economize.apply(base, *extra, config)
            _perturbed(model.techniques[name], std)
            return model
        return build

    def trained(base: Any, args: argparse.Namespace) -> Any:
        from economize.core.serialization import attach, read_checkpoint

        if not args.run:
            raise SystemExit("this recipe needs --run (a trained jump drafter of the model)")
        return attach(base, read_checkpoint(args.run), check_teacher=False)

    def assistant(base: Any, args: argparse.Namespace) -> Any:
        from economize.techniques.gemma4_assistant import Gemma4AssistantDraftConfig

        if not args.assistant:
            raise SystemExit("this recipe needs --assistant (a Gemma-4 assistant checkpoint)")
        return economize.apply(base, Gemma4AssistantDraftConfig(model=args.assistant))

    layers = 24  # Qwen2.5-0.5B
    return {
        "plain": plain,
        "jump": drafter("jump", JumpConfig(target_layer=12, source_layer=18), 0.005),
        "jump_exit": drafter("jump", JumpConfig(target_layer=12, source_layer=18, exit_layer=20), 0.005),
        "jump_lora": drafter("jump", JumpConfig(target_layer=12, source_layer=18, adaptation="lora"), 0.005),
        "jump_full": drafter("jump", JumpConfig(target_layer=12, source_layer=18, adaptation="full"), 0.002),
        "early_exit": drafter("early_exit", EarlyExitConfig(exit_layer=layers - 4), 0.002),
        "eagle3": drafter("eagle3", Eagle3Config(attention_bias=False)),
        "lookup": lambda base, args: economize.apply(base, LookupConfig(max_ngram=4, min_ngram=2)),
        "draft_model": lambda base, args: economize.apply(base, DraftModelConfig(model=args.model,
                                                                                 dtype=args.dtype)),
        "quant_jump": drafter("jump", JumpConfig(target_layer=12, source_layer=18), 0.005,
                              QuantConfig(format="int4", group_size=32)),
        "kv_fp8": lambda base, args: economize.apply(base, FP8KVConfig()),
        "trained": trained,
        "gemma4_assistant": assistant,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe")
    parser.add_argument("out")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--run", default=None)
    parser.add_argument("--assistant", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--attention-backend", default=None, help="e.g. triton, torch_native, flashinfer")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    args = parser.parse_args()

    import transformers

    from economize.backends.sglang import serving

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    base = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).eval()
    model = _recipes()[args.recipe](base, args)
    workdir = Path(args.workdir or Path(args.out).with_suffix(""))
    plan = serving.to_sglang(model, workdir / args.recipe, num_speculative_tokens=args.k, dtype=args.dtype)
    result: dict[str, Any] = {"recipe": args.recipe, "dtype": args.dtype, "plan": plan.to_dict()}
    del model, base

    extra: dict[str, Any] = {}
    if args.attention_backend:
        extra["attention_backend"] = args.attention_backend
    if plan.model_path != args.model and plan.tokenizer is None:
        extra["tokenizer_path"] = args.model
    engine = plan.engine(mem_fraction_static=float(os.environ.get("ECONOMIZE_SGLANG_MEM", "0.5")),
                         context_length=1024, random_seed=0, disable_cuda_graph=args.disable_cuda_graph,
                         log_level="warning", **extra)
    try:
        ids = [tokenizer(prompt).input_ids for prompt in PROMPTS]
        engine.generate(input_ids=ids[:1], sampling_params={"temperature": 0, "max_new_tokens": 8})  # warm-up
        params = {"temperature": 0, "max_new_tokens": args.max_tokens, "ignore_eos": True}
        outputs = engine.generate(input_ids=ids, sampling_params=params, return_logprob=True, top_logprobs_num=TOP)
    finally:
        engine.shutdown()
    result["ids"] = [list(out["output_ids"]) for out in outputs]
    result["top"] = [
        [sorted(([int(token), float(lp)] for lp, token, *_ in position), key=lambda t: -t[1])[:TOP]
         for position in out["meta_info"]["output_top_logprobs"]]
        for out in outputs
    ]
    result["first_logprob"] = [float(out["meta_info"]["output_token_logprobs"][0][0]) for out in outputs]
    result["spec"] = [{key: value for key, value in out["meta_info"].items() if key.startswith("spec_")}
                      for out in outputs]
    Path(args.out).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
