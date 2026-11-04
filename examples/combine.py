"""Stack weight compression, a compressed KV cache and a trained drafter in one model.

    python examples/prepare_data.py --out data
    python examples/combine.py --steps 2000 --out runs/combined

Weight transforms apply first, then the drafter trains on the compressed model it will draft for. Decoding is
lossless with respect to the compressed model: the drafter changes the speed, the weight and KV techniques change
the output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import economize
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.kv_quant import KIVIConfig
from economize.techniques.prune import PruneConfig
from economize.techniques.quant import DraftQuantConfig, QuantConfig
from economize.train import fit, fixed_windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--data", type=Path, default=Path("data"), help="directory with train.bin and val.bin")
    parser.add_argument("--calibration", default="128x2048", help="calibration sequences x tokens")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("runs/combined"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa").to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    n, s = map(int, args.calibration.split("x"))
    calibration = fixed_windows(args.data / "train.bin", n, s)

    model = economize.apply(
        base,
        PruneConfig(method="wanda", sparsity="2:4"),                    # 2:4 sparse weights
        QuantConfig(format="int4", method="gptq", group_size=128),       # then int4 GPTQ
        KIVIConfig(bits=4),                                               # 4-bit KV cache
        Eagle3Config(),                                                   # an EAGLE-3 drafter
        DraftQuantConfig(format="int8"),                                 # int8 weights for the drafter only
        calibration=calibration,
        decoding=economize.DecodeConfig(cycle_length=6, min_confidence=0.3),
    )
    print("attached:", ", ".join(model.techniques))

    fit(model, str(args.data / "train.bin"), steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len,
        seed=0, output_dir=args.out)

    prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May."
    ids = tokenizer(prompt, return_tensors="pt").input_ids[0].to(args.device)
    result = model.decode(ids, args.max_new_tokens, stop=tokenizer.eos_token_id)
    print(tokenizer.decode(result.tokens))
    print(f"\naccepted drafts per cycle: {result.mean_accepted:.2f}")
    for key, value in result.cost_summary().items():
        print(f"  {key}: {value}")
    print(f"  kv_compression: {model.techniques['kivi'].cost_summary(result)['kv_compression']}")


if __name__ == "__main__":
    main()
