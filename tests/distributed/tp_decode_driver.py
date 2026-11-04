"""Driver of the tensor-parallel decoding test, launched with
``torchrun --nproc-per-node P tp_decode_driver.py MODEL CHECKPOINTS OUT``.

Every process loads the model in MODEL split over the launch, and for every economize checkpoint under CHECKPOINTS
(one directory per case) attaches its techniques, decodes the prompt of the test with cycles of 1 and 4 tokens
(without and, on CUDA, with CUDA graphs) and saves the checkpoint again under OUT/<case>. Rank 0 writes the tokens and
the acceptance of every run to OUT/result.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def main() -> None:
    import economize
    from economize.distributed.tp import load_tensor_parallel

    folder, checkpoints, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    model = load_tensor_parallel(folder, dtype=torch.float32)
    model.generation_config.eos_token_id = None
    ids = torch.arange(3, 15).to(model.device)
    graphs = [False, True] if torch.cuda.is_available() else [False]
    result: dict[str, list] = {}
    for case in sorted(path.name for path in checkpoints.iterdir()):
        efficient = economize.load(model, checkpoints / case)
        for cycle in (1, 4):
            for graphed in graphs:
                decoded = efficient.decode(ids, 16, decoding=economize.DecodeConfig(cycle_length=cycle,
                                                                                     cuda_graphs=graphed))
                result[f"{case}/{cycle}/{int(graphed)}"] = [decoded.tokens.tolist(), decoded.acceptance]
        economize.save(efficient, out / case)
        efficient.detach()
    if torch.distributed.get_rank() == 0:
        (out / "result.json").write_text(json.dumps(result))
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
