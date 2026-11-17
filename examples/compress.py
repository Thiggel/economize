"""Prune, factorize, quantize or sparsify a model's weights and measure the perplexity it costs.

    python examples/prepare_data.py --out data
    python examples/compress.py --method wanda-2:4
    python examples/compress.py --method gptq-int4 --method teal-40      # several, applied in order

These techniques rewrite the base weights once, when they attach (lossy). Most need calibration data; the papers
use 128 sequences of 2048 tokens. 2:4 pruning stores semi-structured sparse tensors, which a later low-rank or
quantization step cannot rewrite, so a 2:4 pruning followed by one of those keeps masked dense weights instead.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

import economize
from economize.techniques.lowrank import LowRankConfig
from economize.techniques.prune import DepthPruneConfig, PruneConfig
from economize.techniques.quant import QuantConfig
from economize.techniques.sparsity import TealConfig
from economize.train import fixed_windows

METHODS = {
    "shortgpt-4": lambda: DepthPruneConfig(method="shortgpt", num_blocks=4),
    "sleb-4": lambda: DepthPruneConfig(method="sleb", num_blocks=4),
    # Unstructured sparsity has no sparse kernel: the pruned weights stay masked dense (the simulated reference).
    "magnitude-50": lambda: PruneConfig(method="magnitude", sparsity=0.5, semi_structured=False),
    "wanda-2:4": lambda: PruneConfig(method="wanda", sparsity="2:4"),
    "sparsegpt-50": lambda: PruneConfig(method="sparsegpt", sparsity=0.5, semi_structured=False),
    "svd_llm-20": lambda: LowRankConfig(method="svd_llm", compression=0.2),
    "rtn-int4": lambda: QuantConfig(format="int4", method="rtn", group_size=128),
    "gptq-int4": lambda: QuantConfig(format="int4", method="gptq", group_size=128, act_order=True),
    "awq-int4": lambda: QuantConfig(format="int4", method="awq", group_size=128),
    "fp8": lambda: QuantConfig(format="fp8", group_size=None),
    # The sparse TEAL kernel needs a 16-bit CUDA model; on a CPU the masked dense reference gives the same perplexity.
    "teal-40": lambda: TealConfig(sparsity=0.4, backend="auto" if torch.cuda.is_available() else "dense"),
}


def configs(methods: list[str]) -> list[economize.TechniqueConfig]:
    """The configurations of ``methods`` in order; 2:4 pruning followed by low rank or quantization stays dense."""
    built: list[economize.TechniqueConfig] = [METHODS[m]() for m in methods]
    for i, config in enumerate(built):
        rewritten_later = any(isinstance(c, (LowRankConfig, QuantConfig)) for c in built[i + 1 :])
        if isinstance(config, PruneConfig) and config.semi_structured is None and rewritten_later:
            built[i] = dataclasses.replace(config, semi_structured=False)
    return built


@torch.no_grad()
def perplexity(model: torch.nn.Module, windows: torch.Tensor, device: str) -> float:
    losses = [model(row[None].to(device), labels=row[None].to(device)).loss.float().item() for row in windows]
    return math.exp(sum(losses) / len(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--method", action="append", choices=sorted(METHODS), help="repeatable (default: wanda-2:4)")
    parser.add_argument("--data", type=Path, default=Path("data"), help="directory with train.bin and val.bin")
    parser.add_argument("--calibration", default="128x2048", help="calibration sequences x tokens")
    parser.add_argument("--eval", default="32x1024", help="perplexity windows x tokens of val.bin")
    parser.add_argument("--save", type=Path, help="save the compressed model's economize checkpoint here")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device)
    n, s = map(int, args.calibration.split("x"))
    calibration = fixed_windows(args.data / "train.bin", n, s)
    n_eval, s_eval = map(int, args.eval.split("x"))
    windows = fixed_windows(args.data / "val.bin", n_eval, s_eval)

    before = perplexity(base, windows, args.device)
    methods = args.method or ["wanda-2:4"]
    model = economize.apply(base, *configs(methods), calibration=calibration)
    after = perplexity(model, windows, args.device)   # EfficientModel forwards like the model it wraps
    print(f"{' + '.join(methods)}: perplexity {before:.3f} -> {after:.3f} ({after / before - 1:+.1%})")
    if args.save:
        model.save_pretrained(args.save)


if __name__ == "__main__":
    main()
