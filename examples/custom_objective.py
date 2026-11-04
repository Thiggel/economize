"""Add a new training objective by registration and train a drafter with it.

    python examples/prepare_data.py --out data
    python examples/custom_objective.py --steps 2000

The objective here trains an early-exit drafter with cross-entropy on the model's greedy token. An objective names
the parameters it trains, what it needs from the frozen teacher pass, and its loss; ``fit`` does the rest
(optimizer, schedule, mixed precision, checkpoints, several GPUs).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

import economize
from economize.techniques.early_exit import EarlyExitConfig
from economize.train import (
    Objective,
    ObjectiveOutput,
    OptimDefaults,
    ParamGroup,
    TeacherRequest,
    fit,
    register_objective,
)


@register_objective("exit_top1_ce")
@dataclass
class ExitTop1CE(Objective):
    defaults: ClassVar[OptimDefaults] = OptimDefaults(lr=1e-3, warmup_steps=10)

    def parameter_groups(self, model):
        return [ParamGroup("exit", list(model.techniques["early_exit"].parameters()))]

    def teacher_request(self, model):
        return model.techniques["early_exit"].train_request(model).merge(TeacherRequest(logits=True))

    def loss(self, model, batch, teacher):
        draft = model.techniques["early_exit"].train_logits(model, batch, teacher)[1]
        ce = F.cross_entropy(draft.flatten(0, 1).float(), teacher.logits.argmax(-1).flatten())
        return ObjectiveOutput(ce, {"ce": ce.detach()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--data", type=Path, default=Path("data"), help="directory with train.bin")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device)
    model = economize.apply(base, EarlyExitConfig(exit_layer=6))
    result = fit(model, str(args.data / "train.bin"), "exit_top1_ce", steps=args.steps, batch_size=args.batch_size,
                 seq_len=args.seq_len, log_every=max(1, args.steps // 20))
    print(f"cross-entropy: {result.losses[0]:.3f} -> {result.losses[-1]:.3f}")


if __name__ == "__main__":
    main()
