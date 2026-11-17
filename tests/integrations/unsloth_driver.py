"""Driver of :mod:`test_unsloth`: economize on a model loaded with Unsloth's ``FastLanguageModel``.

Run as ``python unsloth_driver.py MODEL OUT.json [--4bit]`` in a fresh process, because Unsloth must be imported
before ``transformers``. Every case is recorded in ``OUT.json`` with its outcome:

* ``fast_path``: economize on the model as Unsloth returns it, with Unsloth's patched attention.
* ``plain``, ``lookup``, ``early_exit_lora``, ``eagle3``, ``unsloth_peft``: the same model after its modules are
  routed back to the ``transformers`` forward, with the tokens of lossless decoding and the training losses.

``reference`` is greedy ``generate`` on the ``transformers`` forward and ``patched_reference`` greedy ``generate``
with Unsloth's patched forward.
"""

from __future__ import annotations

import argparse
import json
import traceback
from collections.abc import Callable
from typing import Any

from unsloth import FastLanguageModel  # must be imported before transformers

# isort: split
import torch

import economize
from economize.integrations import peft as economize_peft
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.early_exit import EarlyExitConfig
from economize.techniques.lookup import LookupConfig
from economize.train import fit

PROMPT = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
          "How many clips did Natalia sell altogether? Natalia sold 48 clips in April. In May she sold")
NEW = 32
DECODING = economize.DecodeConfig(cycle_length=4)


def use_transformers_forward(model: torch.nn.Module) -> None:
    """Unsloth runs its patched forwards only on modules that carry ``_unsloth_fast_path``."""
    for module in model.modules():
        module._unsloth_fast_path = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--4bit", dest="four_bit", action="store_true")
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()

    base, tokenizer = FastLanguageModel.from_pretrained(args.model, max_seq_length=1024, dtype=torch.bfloat16,
                                                        load_in_4bit=args.four_bit)
    FastLanguageModel.for_inference(base)
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids.cuda()

    @torch.no_grad()
    def greedy() -> list[int]:
        return base.generate(ids, max_new_tokens=NEW, do_sample=False)[0, ids.shape[1]:].tolist()

    report: dict[str, Any] = {"patched_reference": greedy()}

    def case(name: str, fn: Callable[[], dict[str, Any]]) -> None:
        try:
            report[name] = {"ok": True, **fn()}
        except Exception as error:
            report[name] = {"ok": False, "error": f"{type(error).__name__}: {error}", "tb": traceback.format_exc()}

    def decode(model: economize.EfficientModel) -> dict[str, Any]:
        result = model.decode(ids, NEW)
        generated = model.generate(ids, max_new_tokens=NEW)[0, ids.shape[1]:].tolist()
        return {"tokens": result.tokens.tolist(), "generate": generated, "mean_accepted": result.mean_accepted}

    def attached(*configs: Any, train: bool = False) -> dict[str, Any]:
        model = economize.apply(base, *configs, decoding=DECODING if configs else None)
        try:
            out: dict[str, Any] = {}
            if train:
                generator = torch.Generator().manual_seed(0)
                data = torch.randint(0, 20000, (64, 128), generator=generator)
                result = fit(model, data, steps=args.steps, batch_size=4, seq_len=128, seed=0)
                out["losses"] = [float(loss) for loss in result.losses]
            return {**out, **decode(model)}
        finally:
            model.detach()

    case("fast_path", lambda: attached())
    use_transformers_forward(base)
    report["reference"] = greedy()
    case("plain", lambda: attached())
    case("lookup", lambda: attached(LookupConfig()))
    case("early_exit_lora", lambda: attached(EarlyExitConfig(exit_layer=12, adaptation="lora"), train=True))
    case("eagle3", lambda: attached(Eagle3Config(), train=True))

    def unsloth_peft() -> dict[str, Any]:
        policy = FastLanguageModel.get_peft_model(base, r=8, lora_alpha=16, lora_dropout=0,
                                                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                                  use_gradient_checkpointing=False)
        FastLanguageModel.for_inference(policy)
        use_transformers_forward(policy)
        model = economize_peft.apply(policy, EarlyExitConfig(exit_layer=12), decoding=DECODING)
        try:
            return decode(model)
        finally:
            model.detach()

    case("unsloth_peft", unsloth_peft)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle)


if __name__ == "__main__":
    main()
