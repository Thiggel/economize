"""Lossless generation with a drafter, compared with the model alone.

    python examples/generate.py                                    # untrained EAGLE-3 head (lossless, slower)
    python examples/generate.py --drafter runs/eagle3/final        # a trained drafter (see train_drafter.py)
    python examples/generate.py --drafter me/qwen2.5-0.5b-eagle3   # or one from the Hugging Face Hub

The drafter only proposes tokens: with greedy verification the output is the output of the model alone.
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import economize
from economize.techniques.eagle3 import Eagle3Config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--drafter", help="economize checkpoint (local or Hub); default: untrained EAGLE-3 head")
    parser.add_argument("--prompt", default="Natalia sold clips to 48 of her friends in April, and then she sold half "
                                            "as many clips in May. How many clips did Natalia sell altogether?")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--cycle-length", type=int, default=6, help="one exact pass + up to N-1 draft steps")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(args.device)

    start = time.perf_counter()
    reference = base.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)[0, ids.shape[1]:]
    base_seconds = time.perf_counter() - start

    decoding = economize.DecodeConfig(cycle_length=args.cycle_length)
    if args.drafter:
        model = economize.load(base, args.drafter, decoding=decoding)
    else:
        model = economize.apply(base, Eagle3Config(), decoding=decoding)

    start = time.perf_counter()
    result = model.decode(ids[0], args.max_new_tokens)
    seconds = time.perf_counter() - start

    print(tokenizer.decode(result.tokens))
    n = min(len(reference), len(result.tokens))
    print(f"\nidentical to the model alone: {torch.equal(reference[:n].cpu(), result.tokens[:n].cpu())}")
    print(f"accepted drafts per cycle:    {result.mean_accepted:.2f}")
    print(f"wall-clock speedup:           {base_seconds / seconds:.2f}x (eager PyTorch, see benchmark.py)")
    for key, value in result.cost_summary().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
