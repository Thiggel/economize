"""Command line: build or describe a suffix index (``python -m economize.techniques.retrieval -h``)."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from . import build_from_spans
from .index import SuffixIndex


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m economize.techniques.retrieval", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="index records of a flat token file")
    build.add_argument("out", type=Path, help="folder the index is written to")
    build.add_argument("--tokens", type=Path, required=True, help="flat binary token file (or .npy)")
    build.add_argument("--dtype", default="uint32", help="token dtype of a binary file (default uint32)")
    build.add_argument("--spans", type=Path, help="[R, 2] int64 .npy of (start, end) records (default: whole file)")
    build.add_argument("--depth", type=int, default=64, help="tokens the suffix order is exact on (default 64)")
    info = commands.add_parser("info", help="print an index's metadata")
    info.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "info":
        index = SuffixIndex.load(args.path, mmap=True)
        print(json.dumps({"tokens": len(index), "records": index.num_records, "depth": index.depth,
                          "vocab": len(index.buckets) - 1}, indent=2))
        return
    if args.tokens.suffix == ".npy":
        tokens = np.load(args.tokens, mmap_mode="r")
    else:
        tokens = np.memmap(args.tokens, dtype=np.dtype(args.dtype), mode="r")
    spans = np.load(args.spans) if args.spans else np.array([[0, len(tokens)]])
    start = time.perf_counter()
    index = build_from_spans(tokens, spans, args.depth)
    index.save(args.out)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"indexed {len(index)} tokens of {index.num_records} records (depth {index.depth}) into {args.out} "
          f"in {time.perf_counter() - start:.1f}s, peak memory {peak:.1f} GiB")


if __name__ == "__main__":
    main()
