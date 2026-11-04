"""Serve a trained EAGLE-3 drafter with vLLM's speculative decoding.

    pip install "economize[vllm]"
    python examples/serve_vllm.py --drafter runs/eagle3/final --export qwen-eagle3
    vllm serve Qwen/Qwen2.5-0.5B --speculative-config \
        "$(python examples/serve_vllm.py --drafter runs/eagle3/final --export qwen-eagle3 --export-only)"

The export writes the head in vLLM's EAGLE-3 layout, so vLLM drafts with its own EAGLE-3 implementation. Sampling
stays lossless at any temperature: ``speculative_config`` sets vLLM's probabilistic drafting and standard rejection
sampling. Serving needs a CUDA device.
"""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM

import economize
from economize.backends import vllm as economize_vllm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--drafter", required=True, help="economize checkpoint of a drafter (local or Hub)")
    parser.add_argument("--export", default="vllm-drafter", help="folder the vLLM drafter is written to")
    parser.add_argument("--draft-tokens", type=int, default=5, help="draft tokens per verification (cycle length - 1)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--export-only", action="store_true", help="only export and print the speculative_config")
    args = parser.parse_args()

    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    economize_vllm.export_drafter(economize.load(base, args.drafter), args.export)
    speculative_config = economize_vllm.speculative_config(args.export, args.draft_tokens)
    print(json.dumps(speculative_config))
    del base
    if args.export_only:
        return

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, speculative_config=speculative_config, dtype="bfloat16")
    prompts = ["The capital of France is", "def fibonacci(n):", "Photosynthesis converts"]
    for output in llm.generate(prompts, SamplingParams(temperature=args.temperature, max_tokens=128)):
        print(repr(output.prompt), "->", repr(output.outputs[0].text))


if __name__ == "__main__":
    main()
