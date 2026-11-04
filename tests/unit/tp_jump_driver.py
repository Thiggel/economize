"""Driver of the split jump-decoding test: ``torchrun --nproc-per-node 2 tp_jump_driver.py FOLDER OUT.json``.

FOLDER holds the tiny base model (``base/``), one economize checkpoint per variant (names in ``variants.json``) and the
prompts (``prompts.json``). Every rank loads the base split over the launch, attaches each checkpoint and decodes
the prompts in lossless and lossy mode; rank 0 writes the tokens to OUT.json.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import torch


def main() -> None:
    folder, out = Path(sys.argv[1]), Path(sys.argv[2])
    import economize
    from economize import DecodeConfig
    from economize.distributed.tp import load_tensor_parallel

    prompts = [torch.tensor(p) for p in json.loads((folder / "prompts.json").read_text())]
    result: dict[str, object] = {}
    for name in json.loads((folder / "variants.json").read_text()):
        base = load_tensor_parallel(str(folder / "base"), dtype=torch.float32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = economize.load(base, folder / name)
        ids = [p.to(model.device) for p in prompts]
        with torch.no_grad():
            spec = model.decode_batch(ids, 16, decoding=DecodeConfig(cycle_length=4))
            lossy = model.decode_batch(ids, 16, decoding=DecodeConfig(cycle_length=4, lossless=False))
        result[name] = {
            "spec": [r.tokens.tolist() for r in spec],
            "lossy": [r.tokens.tolist() for r in lossy],
            "tp_size": model.adapter.tensor_parallel_size,
            "warnings": [str(w.message) for w in caught if issubclass(w.category, economize.TeacherMismatchWarning)],
        }
    if torch.distributed.get_rank() == 0:
        out.write_text(json.dumps(result))
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
