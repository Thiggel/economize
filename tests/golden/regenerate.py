"""Regenerate the committed golden tensors under ``tests/golden/data/``.

Usage (from the repository root)::

    PYTHONPATH=src:tests python tests/golden/regenerate.py                # every case
    PYTHONPATH=src:tests python tests/golden/regenerate.py medusa lookup  # some cases
    PYTHONPATH=src:tests python tests/golden/regenerate.py --check        # compare only, write nothing

Regenerate a case only after a change that is meant to alter its numbers, and say why in the commit message. The
script prints, for every case, how far the new tensors are from the committed ones and whether a forward-pass case
differs from the dense model at all (a case equal to ``dense`` tests nothing).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import golden.catalog  # noqa: F401  (registers the cases)
from golden.cases import CASES, DATA, decode_after, load_reference, max_difference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="cases to regenerate (default: all)")
    parser.add_argument("--check", action="store_true", help="compare with the committed tensors, write nothing")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.names) - set(CASES))
    if unknown:
        parser.error(f"unknown cases {unknown}; known: {sorted(CASES)}")
    dense = {"logits": CASES["dense"].run()["logits"], "tokens": decode_after()["tokens"]}
    DATA.mkdir(parents=True, exist_ok=True)
    for name in args.names or sorted(CASES):
        case = CASES[name]
        tensors = case.run()
        path = DATA / f"{name}.pt"
        old = load_reference(name) if path.exists() else None
        change = "new" if old is None else f"max |diff| vs committed {max_difference(tensors, old):.3e}"
        trivial = name != "dense" and "acceptance" not in tensors and any(
            torch.equal(tensors[k], dense[k]) for k in dense if k in tensors
        )
        print(f"{name:24s} {change}{'  (EQUAL TO DENSE)' if trivial else ''}")
        if not args.check:
            torch.save({k: v.contiguous() for k, v in tensors.items()}, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
