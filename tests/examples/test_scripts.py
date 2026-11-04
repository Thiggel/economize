"""Runs every script in ``examples/`` on a CPU with tiny settings (``pytest -m slow tests/examples``).

The scripts run as a user would run them, in subprocesses that share one working directory, so that later scripts
read the token files and checkpoints that earlier ones write. ``serve_vllm.py`` and ``benchmark_vllm.py`` need vLLM
on a GPU and run only in their export steps.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
TINY_TRAIN = ["--steps", "2", "--batch-size", "2", "--seq-len", "64"]

#: In order: later scripts read what earlier ones write.
RUNS: list[tuple[str, list[str]]] = [
    ("prepare_data.py", ["--limit", "300", "--out", "data"]),
    ("generate.py", ["--max-new-tokens", "8"]),
    ("sampling.py", ["--max-new-tokens", "8"]),
    ("train_drafter.py", ["--technique", "eagle3", *TINY_TRAIN, "--eval-windows", "2", "--out", "runs/eagle3"]),
    ("train_drafter.py", ["--technique", "early_exit", *TINY_TRAIN, "--eval-windows", "2", "--out", "runs/early_exit"]),
    ("train_drafter.py", ["--technique", "eagle3", "--self-distill", "2", *TINY_TRAIN, "--eval-windows", "2",
                          "--out", "runs/self_distilled"]),
    ("generate.py", ["--drafter", "runs/eagle3/final", "--max-new-tokens", "8"]),
    ("benchmark.py", ["--drafter", "runs/eagle3/final", "--drafter", "runs/early_exit/final", "--drafter", "lookup",
                      "--limit", "2", "--max-new-tokens", "8", "--repeats", "1"]),
    ("compress.py", ["--method", "wanda-2:4", "--method", "rtn-int4", "--calibration", "4x64", "--eval", "2x64"]),
    ("combine.py", ["--calibration", "4x64", *TINY_TRAIN, "--max-new-tokens", "8"]),
    ("custom_objective.py", TINY_TRAIN),
    ("serve_vllm.py", ["--drafter", "runs/eagle3/final", "--export", "vllm-drafter", "--export-only"]),
    ("benchmark_vllm.py", ["--drafter", "eagle3=runs/eagle3/final", "--recipe", "plain", "eagle3+fp8kv", "--k", "3",
                           "--export-device", "cpu", "--export-only"]),
]


def test_every_script_is_tested() -> None:
    assert {path.name for path in EXAMPLES.glob("*.py")} == {name for name, _ in RUNS}


@pytest.mark.slow
def test_scripts_run(tmp_path: Path) -> None:
    pytest.importorskip("datasets")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(EXAMPLES.parent / "src"), env.get("PYTHONPATH")]))
    for script, args in RUNS:
        done = subprocess.run([sys.executable, str(EXAMPLES / script), *args], cwd=tmp_path, env=env,
                              capture_output=True, text=True, timeout=1800)
        assert done.returncode == 0, f"{script} {' '.join(args)}\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}"
