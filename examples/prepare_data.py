"""Tokenize a Hugging Face dataset into the flat token files that training and calibration read.

    python examples/prepare_data.py                                       # GSM8K question + answer
    python examples/prepare_data.py --dataset HuggingFaceFW/fineweb-edu --config sample-10BT \\
        --template "{text}" --limit 200000 --out data/fineweb

Writes ``<out>/train.bin`` and ``<out>/val.bin`` (uint32 token ids, documents separated by the EOS token), the
format of nanoGPT-style pretraining shards. Training draws random windows of ``train.bin``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="whose tokenizer to use")
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--config", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--template", default="{question}\n{answer}", help="format string over the dataset columns")
    parser.add_argument("--limit", type=int, help="use only the first LIMIT rows")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--out", default="data", type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    split = args.split if args.limit is None else f"{args.split}[:{args.limit}]"
    rows = load_dataset(args.dataset, args.config, split=split)
    texts = [args.template.format(**row) + tokenizer.eos_token for row in rows]
    tokens = np.concatenate([np.asarray(ids, dtype=np.uint32) for ids in tokenizer(texts).input_ids])

    cut = int(len(tokens) * (1 - args.val_fraction))
    args.out.mkdir(parents=True, exist_ok=True)
    tokens[:cut].tofile(args.out / "train.bin")
    tokens[cut:].tofile(args.out / "val.bin")
    print(f"{len(rows)} documents, {cut} training and {len(tokens) - cut} validation tokens in {args.out}/")


if __name__ == "__main__":
    main()
