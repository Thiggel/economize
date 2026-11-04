"""Driver of the tensor-parallel test, launched with ``torchrun --nproc-per-node 2 tp_driver.py FOLDER OUT.json``.

Every rank loads the model in FOLDER split over the launch with :func:`economize.distributed.tp.load_tensor_parallel`,
runs one forward pass and a short greedy ``generate``; rank 0 writes the logits, the generated tokens and the shard
shapes to OUT.json.
"""

from __future__ import annotations

import json
import sys

import torch


def main() -> None:
    folder, out = sys.argv[1], sys.argv[2]
    from economize.distributed.tp import is_tensor_parallel, load_tensor_parallel

    model = load_tensor_parallel(folder, dtype=torch.float32)
    rank = torch.distributed.get_rank()
    ids = torch.arange(3, 15).view(1, -1).to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0].float().cpu()
        generated = model.generate(ids, max_new_tokens=8, do_sample=False)[0, ids.shape[1]:].cpu()
    q_proj = model.model.layers[0].self_attn.q_proj.weight
    local = q_proj.to_local() if hasattr(q_proj, "to_local") else q_proj
    if rank == 0:
        with open(out, "w") as handle:
            json.dump({"logits": logits.tolist(), "generated": generated.tolist(), "split": is_tensor_parallel(model),
                       "q_rows": local.shape[0], "world_size": torch.distributed.get_world_size()}, handle)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
