"""Lossless sampling and batched decoding.

    python examples/sampling.py --temperature 0.7 --top-p 0.9

The ``rejection`` verifier accepts sampled drafts so that the output keeps the model's own sampling distribution.
``decode_batch`` decodes prompts of different lengths together, each with its own sampler and stop condition.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import economize
from economize.techniques.eagle3 import Eagle3Config

PROMPTS = ("The capital of France is", "def fibonacci(n):", "Photosynthesis converts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--drafter", help="economize checkpoint (local or Hub); default: untrained EAGLE-3 head")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    decoding = economize.DecodeConfig(cycle_length=4, verifier="rejection")
    if args.drafter:
        model = economize.load(base, args.drafter, decoding=decoding)
    else:
        model = economize.apply(base, Eagle3Config(), decoding=decoding)

    # Through transformers' generate API.
    ids = tokenizer(PROMPTS[0], return_tensors="pt").input_ids.to(args.device)
    out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=args.temperature,
                         top_p=args.top_p)
    print("generate:", tokenizer.decode(out[0, ids.shape[1]:]), "\n")

    # Batched, one sampler per row.
    prompts = [tokenizer(p, return_tensors="pt").input_ids[0].to(args.device) for p in PROMPTS]
    selectors = [economize.TopPSampler(temperature=args.temperature, top_p=args.top_p, seed=args.seed + i,
                                       device=args.device) for i in range(len(prompts))]
    results = model.decode_batch(prompts, args.max_new_tokens, selectors=selectors,
                                 stops=[tokenizer.eos_token_id] * len(prompts))
    for prompt, result in zip(PROMPTS, results):
        print(f"{prompt!r} -> {tokenizer.decode(result.tokens)!r}  ({result.mean_accepted:.2f} accepted per cycle)")


if __name__ == "__main__":
    main()
