"""Jump drafters inside vLLM: greedy decoding with speculation equals greedy decoding without it.

Run with ``pytest -m gpu tests/integrations/test_vllm_gpu.py`` in an environment with vLLM (0.30 or later, V2 model
runner) and a CUDA device. The target is ``ECONOMIZE_VLLM_TEST_MODEL`` (default ``Qwen/Qwen2.5-0.5B``, which must be
in the Hugging Face cache). Decoding runs in float32, where vLLM's batched verification and its one-token decode
steps round alike; in float16 a few long greedy continuations differ after about 100 tokens with any drafter,
vLLM's own n-gram drafter included, because the two kernels round differently.

Untrained drafters of every adaptation check that the draft loop is lossless whatever it proposes. Setting
``ECONOMIZE_VLLM_TEST_RUN`` to a trained economize run of the target also checks that its drafts are accepted.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from helpers import subprocess_env

MODEL = os.environ.get("ECONOMIZE_VLLM_TEST_MODEL", "Qwen/Qwen2.5-0.5B")
RUN = os.environ.get("ECONOMIZE_VLLM_TEST_RUN")
DRIVER = Path(__file__).with_name("vllm_driver.py")


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
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
    pytest.mark.skipif(not _cached(MODEL), reason=f"{MODEL} is not in the Hugging Face cache"),
]


def _drive(*args: str, timeout: int = 1800) -> None:
    env = subprocess_env()
    env.setdefault("HF_HUB_OFFLINE", "1")
    done = subprocess.run([sys.executable, str(DRIVER), *args], capture_output=True, text=True, timeout=timeout,
                          env=env)
    assert done.returncode == 0, done.stderr[-4000:]


def _generate(tmp: Path, name: str, *args: str) -> dict[str, Any]:
    out = tmp / f"{name}.json"
    _drive("generate", MODEL, str(out), *args)
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("vllm")


@pytest.fixture(scope="module")
def plain(workdir: Path) -> dict[str, Any]:
    return _generate(workdir, "plain")


def _drafter(workdir: Path, adaptation: str, run: str | None = None) -> str:
    folder = workdir / f"drafter-{adaptation}"
    if not folder.exists():
        _drive("export", MODEL, str(folder), adaptation, *([run] if run else []))
    return str(folder)


@pytest.mark.parametrize("adaptation,eager", [("frozen", False), ("lora", True), ("full", True)])
def test_speculation_is_lossless(workdir: Path, plain: dict[str, Any], adaptation: str, eager: bool) -> None:
    """``frozen`` drafts through captured CUDA graphs, the others eagerly (both draft step variants run)."""
    spec = _generate(workdir, f"spec-{adaptation}", _drafter(workdir, adaptation), "3",
                     *(["--eager"] if eager else []))
    assert spec["ids"] == plain["ids"]
    assert spec["spec"]["spec_decode_num_drafts"] > 0  # the speculative path ran


@pytest.mark.skipif(RUN is None, reason="set ECONOMIZE_VLLM_TEST_RUN to a trained economize run of the target")
def test_trained_drafter_is_accepted(workdir: Path, plain: dict[str, Any]) -> None:
    spec = _generate(workdir, "spec-trained", _drafter(workdir, "trained", RUN), "3")
    assert spec["ids"] == plain["ids"]
    counters = spec["spec"]
    acceptance_length = 1 + counters["spec_decode_num_accepted_tokens"] / counters["spec_decode_num_drafts"]
    assert acceptance_length > 1.5, counters
