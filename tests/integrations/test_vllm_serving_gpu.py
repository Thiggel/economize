"""Every servable technique, and combinations of them, decoded by vLLM through :mod:`economize.backends.vllm.serving`.

Run with ``pytest -m gpu tests/integrations/test_vllm_serving_gpu.py`` in an environment with vLLM 0.30 and a CUDA
device. The target is ``ECONOMIZE_VLLM_TEST_MODEL`` (default ``Qwen/Qwen2.5-0.5B``, in the Hugging Face cache); each
recipe runs in its own process (:mod:`vllm_serving_driver`).

* Lossless drafters decode in float32 and must give the tokens of plain vLLM. vLLM's float32 kernels are not bit
  identical between a verification step and a one-token step (its Triton attention rounds differently), so the two
  may pick different tokens where the most likely ones are within a few 1e-3 in log-probability. :func:`_lossless`
  accepts a difference only at such a near tie (and compares no further on that prompt). Untrained drafters
  (EAGLE-3, FR-Spec, Medusa with perturbed heads) check that whatever they propose is verified correctly;
  ``ECONOMIZE_VLLM_TEST_RUN`` (a trained jump run of the target) also checks that its drafts are accepted.
  ``ECONOMIZE_VLLM_TEST_FROZEN_RUN`` (a frozen jump run) enables the combination with compressed-tensors weights,
  and ``ECONOMIZE_VLLM_TEST_RUN`` the same combination with a fully adapted jump path (copies served dense).
  The serving layer refuses LoRA jump paths on a compressed-tensors target.
* Weight techniques served densified or natively in float32 (fake quantization, SVD-LLM, Wanda, depth pruning)
  must give the tokens of the economize HF runtime, up to near ties of vLLM's log-probabilities (within 0.02).
* Compressed formats (compressed-tensors int4, fp8 KV cache) run in bfloat16 against a reference of the same
  numbers, and must agree on most tokens and on the first token's log-probability.
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
# a jump run with adaptation="frozen" (its jump path is the quantized target's own layers)
FROZEN_RUN = os.environ.get("ECONOMIZE_VLLM_TEST_FROZEN_RUN")
DRIVER = Path(__file__).with_name("vllm_serving_driver.py")


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


@pytest.fixture(scope="module")
def workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("vllm_serving")


def _decode(workdir: Path, recipe: str, *args: str, timeout: int = 1800) -> dict[str, Any]:
    out = workdir / f"{recipe}{'-'.join(['', *args]).replace('/', '_')}.json"
    if not out.exists():
        env = subprocess_env()
        env.setdefault("HF_HUB_OFFLINE", "1")
        # the driver starts torch's threads before the engine; a forked engine core can then deadlock at start-up
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        done = subprocess.run(
            [sys.executable, str(DRIVER), recipe, str(out), "--model", MODEL, "--workdir", str(workdir), *args],
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


def _near_ties(ids: list[list[int]], reference: list[list[int]], top: list[list[list[Any]]], tie: float) -> int:
    """Assert ``ids`` equal ``reference`` up to near ties; the number of prompts that hit one.

    ``top`` holds ``reference``'s most likely tokens at every position. A prompt may differ at one position, where
    the token of ``ids`` is among them and within ``tie`` of the top log-probability; the continuations are not
    compared after it.
    """
    ties = 0
    for i, (row, ref) in enumerate(zip(ids, reference, strict=True)):
        n = next((j for j, (a, b) in enumerate(zip(row, ref)) if a != b), None)
        if n is None:
            continue
        logprobs = dict(top[i][n])
        assert row[n] in logprobs and max(logprobs.values()) - logprobs[row[n]] < tie, (
            f"prompt {i} differs at token {n}: {row[n]} for {ref[n]}, which is not a near tie ({logprobs})")
        ties += 1
    assert ties <= len(reference) // 2, f"{ties} of {len(reference)} prompts hit a near tie"
    return ties


def _lossless(result: dict[str, Any], reference: dict[str, Any], tie: float = 0.01) -> int:
    """Assert ``result`` decodes ``reference``'s tokens up to near ties of ``reference``'s log-probabilities."""
    return _near_ties(result["ids"], reference["ids"], reference["top"], tie)


@pytest.fixture(scope="module")
def plain(workdir: Path) -> dict[str, Any]:
    result = _decode(workdir, "plain", "--hf")
    # HF and vLLM float32 kernels round differently too (an L40 split a 1.4e-3 tie on one prompt)
    _near_ties(result["hf_ids"], result["ids"], result["top"], 0.01)
    return result


@pytest.mark.parametrize("recipe", ["lookup", "lookup_latest", "lookahead_pool", "rest", "suffix_datastore",
                                    "eagle3", "frspec", "medusa", "dflash", "dflash_mixed", "draft_model",
                                    "early_exit", "early_exit_lora", "layer_skip", "layer_skip_tail",
                                    "kangaroo", "knapspec", "hydra", "hydra_shared", "redrafter",
                                    "redrafter_shared", "hidden_transfer", "dmtd", "free"])
def test_lossless_drafter(workdir: Path, plain: dict[str, Any], recipe: str) -> None:
    result = _decode(workdir, recipe)
    assert result["plan"]["speculative"] is not None
    _lossless(result, plain)
    assert result["spec"].get("spec_decode_num_drafts"), "vLLM did not speculate"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
@pytest.mark.parametrize("recipe", ["hydra", "hydra_shared", "redrafter", "hidden_transfer", "kangaroo"])
def test_lossless_drafter_tensor_parallel(workdir: Path, recipe: str) -> None:
    """The drafters with their own maps or attention heads (replicated maps, sharded heads) under TP=2.

    A wrongly sharded drafter still decodes losslessly, so the acceptance rate must also match the one at TP=1 (up
    to the near ties of float32 kernels that split differently across two GPUs).
    """
    plain_tp = _decode(workdir, "plain", "--tp", "2")
    result = _decode(workdir, recipe, "--tp", "2")
    assert result["plan"]["speculative"] is not None
    _lossless(result, plain_tp)
    single = _decode(workdir, recipe)

    def acceptance(spec: dict[str, Any]) -> float:
        return spec["spec_decode_num_accepted_tokens"] / max(spec["spec_decode_num_draft_tokens"], 1)

    assert result["spec"].get("spec_decode_num_drafts"), "vLLM did not speculate"
    assert abs(acceptance(result["spec"]) - acceptance(single["spec"])) < 0.08, (result["spec"], single["spec"])


@pytest.mark.skipif(importlib.util.find_spec("arctic_inference") is None, reason="needs economize[arctic]")
def test_suffix_decoding(workdir: Path, plain: dict[str, Any]) -> None:
    """vLLM's suffix proposer (Arctic Inference's suffix trees) over the request's own prompt and output."""
    result = _decode(workdir, "suffix")
    assert result["plan"]["speculative"]["method"] == "suffix"
    _lossless(result, plain)
    assert result["spec"].get("spec_decode_num_drafts"), "vLLM did not speculate"


@pytest.mark.skipif(RUN is None, reason="set ECONOMIZE_VLLM_TEST_RUN to a trained jump run of the target")
@pytest.mark.parametrize("recipe", ["jump", "jump_draft_int8"])
def test_trained_jump(workdir: Path, plain: dict[str, Any], recipe: str) -> None:
    result = _decode(workdir, recipe, "--run", str(RUN), "--k", "7")
    _lossless(result, plain)
    spec = result["spec"]
    assert spec["spec_decode_num_accepted_tokens"] > spec["spec_decode_num_drafts"], (
        "a trained drafter should be accepted for more than one token per draft")


@pytest.mark.parametrize("recipe", ["quant_fake_int4", "lowrank", "wanda", "depth_prune"])
def test_weights_match_hf(workdir: Path, recipe: str) -> None:
    result = _decode(workdir, recipe, "--hf")
    if recipe == "lowrank":
        assert result["plan"]["techniques"] == {"lowrank": "plugin"}  # served factored
    _near_ties(result["hf_ids"], result["ids"], result["top"], tie=0.02)  # vLLM's log-probabilities


@pytest.mark.parametrize("recipe", [
    "quant_ct_int4",
    "quant_marlin_int4",  # codes read back from the Marlin weights
    pytest.param("quant_machete_int4", marks=pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="Machete needs sm90")),
])
def test_compressed_tensors_int4(workdir: Path, recipe: str) -> None:
    result = _decode(workdir, recipe, "--hf", "--dtype", "bfloat16")
    assert result["plan"]["engine"].get("quantization", "compressed-tensors") == "compressed-tensors"
    assert _agreement(result["ids"], result["hf_ids"]) > 0.4
    assert max(abs(a - b) for a, b in zip(result["first_logprob"], result["hf_first_logprob"], strict=True)) < 0.1


def _mean(table: list[list[float]]) -> float:
    values = [x for row in table for x in row]
    return sum(values) / len(values)


def _teal_close(result: dict[str, Any], *, margin: float) -> None:
    """vLLM's log-probabilities of its own tokens stay nearer the TEAL HF runtime than the dense model does, and their
    distance from the HF runtime is about HF's own distance from itself under a change of attention kernel.

    TEAL's thresholds turn rounding into flips of activations around ``|x| = t``, and such flips grow through the
    layers, so tokens alone part early; the teacher-forced distances of ``--teacher`` measure every position.
    """
    diff, dense, noise = (_mean(result[key]) for key in ("tf_diff", "tf_dense_diff", "tf_noise_diff"))
    assert diff < margin * dense, (diff, dense)
    assert diff <= 1.5 * noise + 0.05, (diff, noise)


@pytest.mark.parametrize("single", [False, True], ids=["batch", "single"])
def test_teal_matches_hf(workdir: Path, single: bool) -> None:
    """TEAL's thresholded linear maps against the economize HF runtime (float32, the masked dense map, IEEE attention).

    Prompts stay dense and the decoding rows are thresholded (FULL_DECODE_ONLY graphs). The first token comes from the
    dense prompt pass and must agree exactly; the second one, from the first thresholded step, agrees on most prompts.
    """
    result = _decode(workdir, "teal", "--teacher", "--ieee-attention", *(["--single"] if single else []))
    assert result["plan"]["techniques"]["teal"] == "plugin"
    assert max(row[0] for row in result["tf_diff"]) < 1e-3
    exact = sum(row[1] < 1e-3 for row in result["tf_diff"])
    assert exact >= len(result["tf_diff"]) // 2
    assert exact > sum(row[1] < 1e-3 for row in result["tf_dense_diff"])
    _teal_close(result, margin=0.75)


def test_teal_second_half(workdir: Path) -> None:
    """``prefill="second_half"``: the last half of each prompt thresholded in one unchunked pass, as in the HF call."""
    result = _decode(workdir, "teal_second_half", "--teacher", "--ieee-attention")
    assert result["plan"]["engine"]["enable_chunked_prefill"] is False
    assert sum(row[0] < 1e-3 for row in result["tf_diff"]) >= len(result["tf_diff"]) // 2
    _teal_close(result, margin=0.75)


def test_teal_lookup(workdir: Path) -> None:
    """Every row thresholded next to vLLM's n-gram drafter, whose verification passes see the same masked maps."""
    result = _decode(workdir, "teal_lookup", "--teacher", "--ieee-attention")
    assert result["spec"].get("spec_decode_num_drafts"), "vLLM did not speculate"
    _teal_close(result, margin=0.75)


def test_teal_kernel(workdir: Path) -> None:
    """bfloat16: decoding rows run TEAL's sparse GEMV in vLLM and in the HF runtime (``backend="triton"``).

    bfloat16 rounding moves vLLM about as far from the HF runtime as from the dense model on early positions, so the
    margin to the dense model is smaller than in float32.
    """
    result = _decode(workdir, "teal_triton", "--teacher", "--dtype", "bfloat16")
    _teal_close(result, margin=0.9)


@pytest.mark.parametrize("kernel", [False, True], ids=["dense", "triton"])
def test_lowrank_teal(workdir: Path, kernel: bool) -> None:
    """Factored maps with TEAL's thresholds on their inputs and latents (float32 with IEEE attention, or bfloat16
    with TEAL's sparse GEMV on both factors), against the economize HF runtime."""
    if kernel:
        result = _decode(workdir, "lowrank_teal_triton", "--teacher", "--dtype", "bfloat16")
    else:
        result = _decode(workdir, "lowrank_teal", "--teacher", "--ieee-attention")
    assert result["plan"]["techniques"] == {"lowrank": "plugin", "teal": "plugin"}
    _teal_close(result, margin=0.9 if kernel else 0.75)


def test_kv_fp8(workdir: Path) -> None:
    """vLLM's fp8 cache against economize's HF fp8 cache of the same format and unit scales.

    With unit scales an e4m3 cache is noisy on Qwen2.5-0.5B (calibrated scales would help), and the two runtimes
    round at different points, so they part early: measured on an L40, agreement 0.45 (FlashInfer)
    or 0.23 (Triton attention) and up to 0.38 apart in the first token's log-probability, against 0.10 between the
    HF fp8 and bf16 caches. The bounds catch a cache that is read with the wrong format or scale (garbage).
    """
    result = _decode(workdir, "kv_fp8", "--hf", "--dtype", "bfloat16")
    assert result["plan"]["engine"]["kv_cache_dtype"].startswith("fp8")
    assert _agreement(result["ids"], result["hf_ids"]) > 0.15
    assert max(abs(a - b) for a, b in zip(result["first_logprob"], result["hf_first_logprob"], strict=True)) < 0.6


def test_kv_fp8_calibrated(workdir: Path) -> None:
    """Calibrated per-layer fp8 scales: vLLM loads them from the checkpoint and decodes close to economize's HF cache.

    The bounds are those of the unit-scale cache (:func:`test_kv_fp8`); the scales must be the calibrated ones.
    """
    result = _decode(workdir, "kv_fp8_tensor", "--hf", "--dtype", "bfloat16", "--kv-scales")
    assert result["plan"]["engine"]["kv_cache_dtype"] == "fp8_e4m3"
    config = json.loads((Path(result["plan"]["model"]) / "config.json").read_text())
    assert config["quantization_config"]["kv_cache_scheme"]["strategy"] == "tensor"
    scales = result["vllm_kv_scales"]
    assert scales and all(k > 0 and v > 0 for k, v in scales) and any(k != 1.0 for k, _ in scales)
    assert _agreement(result["ids"], result["hf_ids"]) > 0.15
    assert max(abs(a - b) for a, b in zip(result["first_logprob"], result["hf_first_logprob"], strict=True)) < 0.6


@pytest.fixture(scope="module")
def streaming(workdir: Path, plain: dict[str, Any]) -> dict[str, Any]:
    """StreamingLLM on economize's CUSTOM backend against economize's HF runtime (4 sinks, 16 recent tokens)."""
    result = _decode(workdir, "streaming", "--hf")
    assert result["plan"]["engine"]["attention_backend"] == "CUSTOM"
    _near_ties(result["hf_ids"], result["ids"], result["top"], 0.01)
    assert result["ids"] != plain["ids"], "the window must change the continuations"
    return result


def test_streaming(streaming: dict[str, Any]) -> None:
    assert streaming["plan"]["techniques"] == {"kv_evict": "plugin"}


def test_streaming_long_decode(workdir: Path) -> None:
    """192 tokens drop up to 12 pages of 16 from each decoding request's block table; the tokens stay HF's."""
    result = _decode(workdir, "streaming", "--hf", "--max-tokens", "192")
    _near_ties(result["hf_ids"], result["ids"], result["top"], 0.01)
    drafted = _decode(workdir, "streaming_eagle3", "--max-tokens", "192")
    _lossless(drafted, result)


def test_streaming_layer_subset(workdir: Path, streaming: dict[str, Any]) -> None:
    result = _decode(workdir, "streaming_layers", "--hf")
    _near_ties(result["hf_ids"], result["ids"], result["top"], 0.01)
    assert result["ids"] != streaming["ids"]


@pytest.mark.parametrize("recipe", ["streaming_eagle3", "streaming_early_exit"])
def test_streaming_drafter_is_lossless(workdir: Path, streaming: dict[str, Any], recipe: str) -> None:
    """Each verification query reads the window of its own position: the tokens are plain streaming decoding's."""
    result = _decode(workdir, recipe)
    assert result["plan"]["speculative"] is not None
    _lossless(result, streaming)
    assert result["spec"].get("spec_decode_num_drafts"), "vLLM did not speculate"


def test_streaming_kv_fp8(workdir: Path) -> None:
    """A per-token-head fp8 cache under the streaming mask, against economize's HF runtime (bounds of test_kv_fp8)."""
    result = _decode(workdir, "streaming_kv_fp8", "--hf", "--dtype", "bfloat16")
    assert result["plan"]["engine"]["kv_cache_dtype"] == "fp8_per_token_head"
    assert result["plan"]["engine"]["attention_backend"] == "CUSTOM"
    assert _agreement(result["ids"], result["hf_ids"]) > 0.15
    assert max(abs(a - b) for a, b in zip(result["first_logprob"], result["hf_first_logprob"], strict=True)) < 0.6


@pytest.fixture(scope="module")
def kivi(workdir: Path, plain: dict[str, Any]) -> dict[str, Any]:
    """KIVI (4 bits, groups of 16, a residual of 32) on economize's CUSTOM backend against economize's HF runtime.

    The engine runs with CUDA graphs, so decoding replays the backend's captured quantization and attention.
    """
    result = _decode(workdir, "kivi", "--hf")
    assert result["plan"]["engine"]["attention_backend"] == "CUSTOM"
    assert result["plan"]["techniques"] == {"kivi": "plugin"}
    _near_ties(result["hf_ids"], result["ids"], result["top"], 0.01)
    assert result["ids"] != plain["ids"], "the quantization must change the continuations"
    return result


@pytest.mark.parametrize("recipe", ["kivi_2bit", "kivi_layers"])
def test_kivi_variants(workdir: Path, kivi: dict[str, Any], recipe: str) -> None:
    result = _decode(workdir, recipe, "--hf")
    _near_ties(result["hf_ids"], result["ids"], result["top"], 0.01)
    assert result["ids"] != kivi["ids"]


@pytest.mark.parametrize("eager", [False, True], ids=["graphs", "eager"])
def test_kivi_fused_matches_hf(workdir: Path, kivi: dict[str, Any], eager: bool) -> None:
    """The paged KIVI kernel over the packed codes, against economize's HF ``kivi_fused`` runtime."""
    result = _decode(workdir, "kivi_fused", "--hf", *(["--eager"] if eager else []))
    assert result["plan"]["techniques"] == {"kivi_fused": "plugin"}
    _near_ties(result["hf_ids"], result["ids"], result["top"], 0.01)
    # the fused kernel reads the same numbers as the dense path of kivi
    _near_ties(result["ids"], kivi["ids"], kivi["top"], 0.01)


@pytest.mark.parametrize("recipe", ["kivi_eagle3", "kivi_fused_eagle3", "kivi_fused_early_exit"])
def test_kivi_drafter_is_lossless(workdir: Path, kivi: dict[str, Any], recipe: str) -> None:
    """Only committed tokens are quantized, so verification sees plain KIVI decoding's cache at every position."""
    result = _decode(workdir, recipe)
    assert result["plan"]["speculative"] is not None
    _lossless(result, kivi)
    assert result["spec"].get("spec_decode_num_drafts"), "vLLM did not speculate"


@pytest.mark.skipif(RUN is None, reason="set ECONOMIZE_VLLM_TEST_RUN to a trained jump run of the target")
def test_kivi_jump(workdir: Path, kivi: dict[str, Any]) -> None:
    result = _decode(workdir, "kivi_fused_jump", "--run", str(RUN), "--k", "7")
    _lossless(result, kivi)
    spec = result["spec"]
    assert spec["spec_decode_num_accepted_tokens"] > spec["spec_decode_num_drafts"]


@pytest.mark.parametrize("run", [
    pytest.param(FROZEN_RUN, id="frozen", marks=pytest.mark.skipif(
        FROZEN_RUN is None, reason="set ECONOMIZE_VLLM_TEST_FROZEN_RUN to a frozen jump run of the target")),
    pytest.param(RUN, id="adapted", marks=pytest.mark.skipif(
        RUN is None, reason="set ECONOMIZE_VLLM_TEST_RUN to a trained jump run of the target")),
])
def test_combination(workdir: Path, run: str) -> None:
    """compressed-tensors int4 + fp8 KV cache + a jump drafter: the drafter must not change the tokens.

    A frozen drafter runs the quantized target's own layers, a fully adapted one (``ECONOMIZE_VLLM_TEST_RUN``) its
    dense copies. Both sides run in bfloat16 with an fp8 cache (unit scales) and Triton attention, which the plan
    picks for an fp8 cache with a plugin drafter. They part at the noise floor of speculative decoding in these
    numbers (a verification step and a one-token step round differently): on an L40 both drafters decode the same
    tokens, with an agreed prefix of 0.40 against the target on Triton and 0.37 against it on FlashInfer. The bound
    only catches a drafter that corrupts the target's cache or weights, which gives garbage within a few tokens.
    """
    target = _decode(workdir, "combo_target", "--dtype", "bfloat16", "--attention-backend", "TRITON_ATTN")
    result = _decode(workdir, "combo", "--dtype", "bfloat16", "--run", str(run), "--k", "7")
    assert set(result["plan"]["techniques"]) == {"quant", "kv_fp8", "jump"}
    assert _agreement(result["ids"], target["ids"]) > 0.25
    if run == RUN:
        spec = result["spec"]
        assert spec["spec_decode_num_accepted_tokens"] > spec["spec_decode_num_drafts"], (
            "the dense copies of a trained drafter should be accepted for more than one token per draft")
