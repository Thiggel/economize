"""economize's own decoding on a model split by tensor parallelism: jump and early-exit drafters with frozen, LoRA
and fully trained layers, chained verification and (on CUDA) graphs.

Techniques are applied and perturbed on one process and saved; ``tp_decode_driver.py`` loads each checkpoint on the
split model under ``torchrun``, decodes and saves it again. The split decoding must emit the greedy tokens of the
whole model, accept as many drafts as the unsplit decoding does, and save the tensors it loaded.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

import economize
from economize import DecodeConfig, JumpConfig
from economize.core.serialization import WEIGHTS_FILE
from economize.techniques.early_exit import EarlyExitConfig
from helpers import make_tiny_model, perturb, reference_greedy, subprocess_env

DRIVER = Path(__file__).with_name("tp_decode_driver.py")

CASES = {
    "jump-frozen": JumpConfig(target_layer=1, source_layer=2, exit_layer=5),
    "jump-lora": JumpConfig(target_layer=1, source_layer=2, exit_layer=5, adaptation="lora", lora_rank=2),
    "jump-full": JumpConfig(target_layer=1, source_layer=2, exit_layer=5, adaptation="full"),
    "exit-lora": EarlyExitConfig(exit_layer=4, adaptation="lora", lora_rank=2),
    "exit-full": EarlyExitConfig(exit_layer=4, adaptation="full"),
}


def _run(tmp_path: Path, processes: int, devices: str) -> None:
    model = make_tiny_model(heads=2 * processes, kv_heads=processes)
    folder, checkpoints, out = tmp_path / "tiny", tmp_path / "checkpoints", tmp_path / "out"
    model.save_pretrained(folder)
    ids = torch.arange(3, 15)
    greedy = reference_greedy(model, ids, 16).tolist()
    expected: dict[str, list] = {}
    for seed, (case, config) in enumerate(CASES.items()):
        efficient = economize.apply(model, config)
        for technique in efficient.techniques.values():
            perturb(technique, 0.05, seed)
        for cycle in (1, 4):
            decoded = efficient.decode(ids, 16, decoding=DecodeConfig(cycle_length=cycle))
            assert decoded.tokens.tolist() == greedy, case
            expected[f"{case}/{cycle}"] = [list(pair) for pair in decoded.acceptance]
        economize.save(efficient, checkpoints / case)
        efficient.detach()
    env = subprocess_env()
    env["CUDA_VISIBLE_DEVICES"] = devices
    env.setdefault("OMP_NUM_THREADS", "2")
    command = [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", str(processes), "--master-port",
               str(28500 + os.getpid() % 1000), str(DRIVER), str(folder), str(checkpoints), str(out)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=1800, env=env)
    assert done.returncode == 0, done.stderr[-4000:]
    result = json.loads((out / "result.json").read_text())
    assert {key.rsplit("/", 1)[0] for key in result} == set(expected)
    for key, (tokens, acceptance) in result.items():
        assert tokens == greedy, key
        assert [list(pair) for pair in acceptance] == expected[key.rsplit("/", 1)[0]], key
    for case in CASES:
        saved, resaved = load_file(checkpoints / case / WEIGHTS_FILE), load_file(out / case / WEIGHTS_FILE)
        assert saved.keys() == resaved.keys(), case
        for name, tensor in saved.items():
            torch.testing.assert_close(resaved[name], tensor, rtol=0, atol=0, msg=f"{case} {name}")


@pytest.mark.slow
def test_decoding_over_two_cpu_processes(tmp_path: Path) -> None:
    _run(tmp_path, 2, "")


@pytest.mark.gpu
@pytest.mark.parametrize("processes", [2, 4])
def test_decoding_over_gpus(tmp_path: Path, processes: int) -> None:
    if torch.cuda.device_count() < processes:
        pytest.skip(f"needs {processes} CUDA devices")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(processes))))
    _run(tmp_path, processes, ",".join(visible.split(",")[:processes]))
