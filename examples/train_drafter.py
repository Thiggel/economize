"""Train a drafter on a frozen model and save it as an economize checkpoint.

    python examples/prepare_data.py --out data
    python examples/train_drafter.py --technique eagle3 --data data --steps 2000 --out runs/eagle3
    python examples/train_drafter.py --technique early_exit --data data --steps 2000 --out runs/early_exit
    python examples/train_drafter.py --technique eagle3 --self-distill 2000 ...   # train on the model's own answers
    accelerate launch examples/train_drafter.py --accelerate ...                   # data-parallel on several GPUs

Only the drafter's parameters train; the base weights stay untouched, so decoding with the result stays lossless.
Load the result with ``economize.load(base, "runs/eagle3/final")`` or ``python examples/generate.py --drafter ...``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import economize
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.early_exit import EarlyExitConfig
from economize.techniques.medusa import MedusaConfig
from economize.train import fit, fixed_windows, generate_self_distillation

TECHNIQUES = {
    "eagle3": lambda: Eagle3Config(),
    "medusa": lambda: MedusaConfig(num_heads=4),
    "early_exit": lambda: EarlyExitConfig(exit_layer=6, head="readout"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--technique", choices=sorted(TECHNIQUES), default="eagle3")
    parser.add_argument("--data", type=Path, default=Path("data"), help="directory with train.bin and val.bin")
    parser.add_argument("--self-distill", type=int, metavar="N",
                        help="train on the model's answers to the first N GSM8K questions instead of train.bin")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, help="default: the objective's own learning rate")
    parser.add_argument("--eval-windows", type=int, default=64)
    parser.add_argument("--eval-seq-len", type=int,
                        help="length of the validation windows (default: --seq-len)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/drafter"))
    parser.add_argument("--accelerate", action="store_true", help="run under accelerate launch")
    parser.add_argument("--push-to-hub", metavar="REPO", help="upload the trained drafter, e.g. me/qwen-eagle3")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    if not args.accelerate:
        base.to(args.device)
    model = economize.apply(base, TECHNIQUES[args.technique]())
    print(f"{args.technique}: {sum(p.numel() for p in model.technique_parameters()):,} trainable parameters")

    data: object = str(args.data / "train.bin")
    if args.self_distill:
        from datasets import load_dataset

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        rows = load_dataset("openai/gsm8k", "main", split=f"train[:{args.self_distill}]")
        data = generate_self_distillation(base, tokenizer, [row["question"] for row in rows],
                                          max_new_tokens=args.seq_len // 2, batch_size=args.batch_size)

    eval_data = fixed_windows(args.data / "val.bin", args.eval_windows, args.eval_seq_len or args.seq_len)
    result = fit(model, data, steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len, lr=args.lr,
                 seed=args.seed, eval_data=eval_data,
                 output_dir=args.out, accelerate=args.accelerate)

    print(f"final loss {result.losses[-1]:.4f}, saved to {result.final_path}")
    for name, value in (result.eval or {}).items():
        shown = [round(float(v), 4) for v in value[:4]] if isinstance(value, (list, tuple)) else value
        print(f"  eval {name}: {shown}")
    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub, private=True)


if __name__ == "__main__":
    main()
