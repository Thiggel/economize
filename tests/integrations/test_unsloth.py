"""economize on a model loaded with Unsloth's ``FastLanguageModel``, on a CUDA device.

Run with ``pytest -m gpu tests/integrations/test_unsloth.py`` in an environment with ``unsloth`` installed. Each
case loads ``Qwen/Qwen2.5-0.5B`` (``ECONOMIZE_UNSLOTH_TEST_MODEL``) in :mod:`unsloth_driver`, a fresh process,
since Unsloth must be imported before ``transformers``.

Unsloth replaces the attention forward of the model's classes with its own, which ignores the key/value cache that
economize passes, so economize fails on the model as Unsloth returns it. With every module routed back to the
``transformers`` forward, lossless decoding reproduces Unsloth's own greedy tokens, and drafters train on the model.
If Unsloth or economize changes so that the patched forward works, ``test_patched_forward_fails`` fails and
``docs/ecosystem.md`` must be updated.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from helpers import subprocess_env

MODEL = os.environ.get("ECONOMIZE_UNSLOTH_TEST_MODEL", "Qwen/Qwen2.5-0.5B")
DRIVER = Path(__file__).with_name("unsloth_driver.py")


def _cached(model: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(model, "config.json"), str) or Path(model).is_dir()


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(importlib.util.find_spec("unsloth") is None, reason="needs unsloth"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
    pytest.mark.skipif(not _cached(MODEL), reason=f"{MODEL} is not in the Hugging Face cache"),
]


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("unsloth") / "report.json"
    done = subprocess.run([sys.executable, str(DRIVER), MODEL, str(out)], capture_output=True, text=True,
                          timeout=3600, env=subprocess_env())
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(out.read_text())


def test_patched_forward_fails(report: dict[str, Any]) -> None:
    assert not report["fast_path"]["ok"]
    assert report["fast_path"]["error"].startswith("RuntimeError")


@pytest.mark.parametrize("name", ["plain", "lookup", "early_exit_lora", "eagle3", "unsloth_peft"])
def test_transformers_forward_is_lossless(report: dict[str, Any], name: str) -> None:
    case = report[name]
    assert case["ok"], case.get("tb")
    assert case["tokens"] == report["reference"]
    assert case["generate"] == report["reference"]


@pytest.mark.parametrize("name", ["early_exit_lora", "eagle3"])
def test_drafters_train(report: dict[str, Any], name: str) -> None:
    losses = report[name]["losses"]
    assert all(math.isfinite(loss) for loss in losses)
    assert losses[-1] < losses[0]
