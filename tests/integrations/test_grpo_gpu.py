"""TRL GRPO with speculative vLLM rollouts (colocate mode): the rollouts take the speculative path and stay exact.

Run with ``pytest -m gpu tests/integrations/test_grpo_gpu.py`` in an environment with TRL 1.14 or later, vLLM 0.30
or later and a CUDA device (``ECONOMIZE_VLLM_GPU_UTIL``, default 0.3, is the engine's memory share). Each case trains
``Qwen/Qwen2.5-0.5B`` (``ECONOMIZE_VLLM_TEST_MODEL``) for a few GRPO steps in :mod:`grpo_driver`.

* An untrained LoRA drafter with re-distillation every step: vLLM verifies its drafts (the counters move), every
  policy sync re-merges the LoRA updates into the new layers, and the drafter's re-distilled tensors arrive in the
  engine with every sync.
* Plain rollouts as the reference: TRL's gap between the engine's sampling log-probabilities and the trainer's
  recomputed ones is the same with and without speculation, so the rollouts' log-probabilities come from the
  target's verification, never from the drafter.
* With ``ECONOMIZE_VLLM_TEST_RUN`` set to a trained economize run of the model, its drafts are also accepted.
"""

from __future__ import annotations

import importlib.util
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from helpers import subprocess_env

MODEL = os.environ.get("ECONOMIZE_VLLM_TEST_MODEL", "Qwen/Qwen2.5-0.5B")
RUN = os.environ.get("ECONOMIZE_VLLM_TEST_RUN")
DRIVER = Path(__file__).with_name("grpo_driver.py")
STEPS = 3


def _cached(model: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(model, "config.json"), str) or Path(model).is_dir()


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(importlib.util.find_spec("vllm") is None, reason="needs vLLM"),
    pytest.mark.skipif(importlib.util.find_spec("trl") is None, reason="needs TRL"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
    pytest.mark.skipif(not _cached(MODEL), reason=f"{MODEL} is not in the Hugging Face cache"),
]


def _train(tmp: Path, name: str, *args: str) -> dict[str, Any]:
    out = tmp / f"{name}.json"
    env = subprocess_env()
    env.setdefault("HF_HUB_OFFLINE", "1")
    done = subprocess.run([sys.executable, str(DRIVER), MODEL, str(out), "--steps", str(STEPS), *args],
                          capture_output=True, text=True, timeout=1800, env=env)
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(out.read_text())


def _steps(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in result["log_history"] if "loss" in entry]


def _logp_gap(result: dict[str, Any]) -> float:
    return statistics.mean(entry["sampling/sampling_logp_difference/mean"] for entry in _steps(result))


@pytest.fixture(scope="module")
def workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("grpo")


@pytest.fixture(scope="module")
def plain(workdir: Path) -> dict[str, Any]:
    return _train(workdir, "plain")


def test_lora_drafter_follows_the_policy(workdir: Path, plain: dict[str, Any]) -> None:
    result = _train(workdir, "lora", "--adaptation", "lora", "--redistill-every", "1")
    steps = _steps(result)
    assert len(steps) == STEPS and result["spec"]["spec_decode_num_drafts"] > 0
    assert all(entry["spec/drafts"] > 0 for entry in steps)  # every rollout batch was drafted and verified
    assert result["refreshes"] >= STEPS - 1  # one re-merge per policy sync
    assert result["redistill_rounds"] == STEPS
    assert result["drafter_gap"] == 0 and result["drafter_gap_to_export"] > 0
    assert _logp_gap(result) < max(2 * _logp_gap(plain), 0.05)


@pytest.mark.skipif(RUN is None, reason="set ECONOMIZE_VLLM_TEST_RUN to a trained economize run of the model")
def test_trained_drafter_is_accepted(workdir: Path, plain: dict[str, Any]) -> None:
    result = _train(workdir, "trained", "--adaptation", "full", "--run", str(RUN))
    totals = result["spec"]
    assert 1 + totals["spec_decode_num_accepted_tokens"] / totals["spec_decode_num_drafts"] > 1.5
    assert all(entry["spec/acceptance_length"] > 1.5 for entry in _steps(result))
    assert _logp_gap(result) < max(2 * _logp_gap(plain), 0.05)
