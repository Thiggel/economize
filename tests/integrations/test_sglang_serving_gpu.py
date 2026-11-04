"""Servable techniques decoded by SGLang through :mod:`economize.backends.sglang.serving`.

Run with ``pytest -m gpu tests/integrations/test_sglang_serving_gpu.py`` in an environment with SGLang 0.5.21 and a
CUDA device. The target is ``ECONOMIZE_SGLANG_TEST_MODEL`` (default ``Qwen/Qwen2.5-0.5B``, in the Hugging Face
cache); each recipe runs in its own process (:mod:`sglang_serving_driver`).

* Lossless drafters must give the tokens of plain SGLang. SGLang's CUDA kernels (RMSNorm, rotary, sampling) take
  only float16 / bfloat16, so the tests decode in bfloat16 (``ECONOMIZE_SGLANG_TEST_DTYPE``). A verification pass
  and a one-token step round differently, so the two may pick different tokens where the most likely ones are
  within the rounding noise of the dtype; :func:`_lossless` accepts a difference only at such a near tie (``TIE``
  in log-probability) and compares no further on that prompt. The untrained drafters (perturbed predictor,
  perturbed EAGLE-3 head) check that whatever they propose is verified correctly. ``ECONOMIZE_SGLANG_TEST_RUN``
  (a trained jump run of the target) also checks that its drafts are accepted, which only holds when the draft
  model computes the jump step of the economize runtime.
* Fake quantization is served densified: the dequantized base is the target, and the drafter's layers are its copies.
* The fp8 KV cache must give the first token of plain SGLang of the same dtype and agree with it on a prefix.
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

MODEL = os.environ.get("ECONOMIZE_SGLANG_TEST_MODEL", "Qwen/Qwen2.5-0.5B")
RUN = os.environ.get("ECONOMIZE_SGLANG_TEST_RUN")
BACKEND = os.environ.get("ECONOMIZE_SGLANG_TEST_ATTENTION", "triton")
DTYPE = os.environ.get("ECONOMIZE_SGLANG_TEST_DTYPE", "bfloat16")
#: Largest log-probability gap of a near tie. Qwen2.5-0.5B's logits lie between 16 and 32 in magnitude, where one
#: bfloat16 step is 0.125: the parted prompts of our test runs all parted at a gap of one step.
TIE = {"float32": 0.01, "float16": 0.05}.get(DTYPE, 0.25)
DRIVER = Path(__file__).with_name("sglang_serving_driver.py")


def _cached(model: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(model, "config.json"), str) or Path(model).is_dir()


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(importlib.util.find_spec("sglang") is None, reason="needs SGLang"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
    pytest.mark.skipif(not _cached(MODEL), reason=f"{MODEL} is not in the Hugging Face cache"),
]


@pytest.fixture(scope="module")
def workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("sglang_serving")


def _decode(workdir: Path, recipe: str, *args: str, timeout: int = 1800) -> dict[str, Any]:
    if "--dtype" not in args:
        args = ("--dtype", DTYPE, *args)
    out = workdir / f"{recipe}{'-'.join(['', *args]).replace('/', '_')}.json"
    if not out.exists():
        env = subprocess_env()
        env.setdefault("HF_HUB_OFFLINE", "1")
        done = subprocess.run(
            [sys.executable, str(DRIVER), recipe, str(out), "--model", MODEL, "--workdir", str(workdir),
             "--attention-backend", BACKEND, *args],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(out.read_text())


def _agreement(ids: list[list[int]], reference: list[list[int]]) -> float:
    """Mean fraction of each continuation that equals the reference up to the first difference."""
    fractions = []
    for a, b in zip(ids, reference, strict=True):
        n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        fractions.append(n / max(len(b), 1))
    return sum(fractions) / len(fractions)


def _lossless(result: dict[str, Any], reference: dict[str, Any], tie: float = TIE) -> int:
    """Assert ``result`` decodes ``reference``'s tokens up to near ties; the number of prompts that hit one."""
    ties = 0
    for i, (row, ref) in enumerate(zip(result["ids"], reference["ids"], strict=True)):
        n = next((j for j, (a, b) in enumerate(zip(row, ref)) if a != b), None)
        if n is None:
            continue
        logprobs = {int(token): lp for token, lp in reference["top"][i][n]}
        assert row[n] in logprobs and max(logprobs.values()) - logprobs[row[n]] < tie, (
            f"prompt {i} differs at token {n}: {row[n]} for {ref[n]}, which is not a near tie ({logprobs})")
        ties += 1
    if DTYPE == "float32":
        assert ties <= len(reference["ids"]) // 2, f"{ties} of {len(reference['ids'])} prompts hit a near tie"
    return ties


def _accept_length(result: dict[str, Any]) -> float:
    lengths = [row.get("spec_accept_length") for row in result["spec"]]
    assert all(length is not None for length in lengths), "SGLang did not speculate"
    return sum(lengths) / len(lengths)


@pytest.fixture(scope="module")
def plain(workdir: Path) -> dict[str, Any]:
    result = _decode(workdir, "plain")
    assert result["plan"]["techniques"] == {}
    return result


@pytest.mark.parametrize("recipe", ["jump", "jump_exit", "jump_lora", "jump_full", "early_exit", "eagle3",
                                    "lookup", "draft_model"])
def test_lossless_drafter(workdir: Path, plain: dict[str, Any], recipe: str) -> None:
    result = _decode(workdir, recipe)
    assert result["plan"]["server_args"]["speculative_algorithm"]
    _lossless(result, plain)
    assert _accept_length(result) >= 1.0


def test_fake_quant_with_jump(workdir: Path) -> None:
    """The dequantized base is the target; the jump drafter's layers are its copies (densified)."""
    quantized = _decode(workdir, "quant_jump")
    assert quantized["plan"]["techniques"] == {"quant": "densified", "jump": "plugin"}
    assert _accept_length(quantized) >= 1.0


@pytest.mark.skipif(RUN is None, reason="set ECONOMIZE_SGLANG_TEST_RUN to a trained jump run of the target")
def test_trained_jump_drafts_are_accepted(workdir: Path, plain: dict[str, Any]) -> None:
    result = _decode(workdir, "trained", "--run", str(RUN))
    _lossless(result, plain)
    # a wrong drafter (shifted rows, a wrong aux layer, unmerged LoRA) is accepted about as often as an untrained one
    assert _accept_length(result) > 1.8, result["spec"]


def test_fp8_kv_cache(workdir: Path) -> None:
    reference = _decode(workdir, "plain", "--dtype", "bfloat16")
    result = _decode(workdir, "kv_fp8", "--dtype", "bfloat16")
    assert result["plan"]["server_args"]["kv_cache_dtype"] == "fp8_e4m3"
    # unscaled fp8 keys of Qwen2.5-0.5B part from bfloat16 after about a quarter of 64 tokens (0.23 in our runs)
    assert _agreement(result["ids"], reference["ids"]) > 0.1
    first = torch.tensor(result["first_logprob"]) - torch.tensor(reference["first_logprob"])
    assert first.abs().max() < 0.5
