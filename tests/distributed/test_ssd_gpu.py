"""Speculative speculative decoding on two GPUs: the target on ``cuda:0`` verifies while the drafter on ``cuda:1``
speculates, and the output stays the target's.

Run with ``CUDA_VISIBLE_DEVICES=0,1 pytest -m "gpu" tests/distributed/test_ssd_gpu.py`` (add ``or slow`` for the
pretrained pair, ``ECONOMIZE_SSD_TARGET`` / ``ECONOMIZE_SSD_DRAFTER``, default Qwen2.5-1.5B and Qwen2.5-0.5B, which
must be in the Hugging Face cache). Both run in float32, where batched verification and one-token decoding agree.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from economize.distributed import AsyncSpeculativeDecoder, SSDConfig
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, perturb, reference_greedy

TARGET = os.environ.get("ECONOMIZE_SSD_TARGET", "Qwen/Qwen2.5-1.5B")
DRAFTER = os.environ.get("ECONOMIZE_SSD_DRAFTER", "Qwen/Qwen2.5-0.5B")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices"),
]


def _cached(model: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(model, "config.json"), str) or Path(model).is_dir()


@pytest.mark.parametrize("mode", ["async", "sync"])
def test_tiny_models_on_two_devices(mode: str) -> None:
    target = make_tiny_model(seed=0).to("cuda:0")
    drafter = make_tiny_model(seed=0)
    perturb(drafter, 0.004, seed=3)
    drafter = drafter.to("cuda:1")
    prompt = torch.randint(0, VOCAB, (PROMPT_LEN,), generator=torch.Generator().manual_seed(1))
    reference = reference_greedy(target, prompt.to("cuda:0"), 40).cpu()
    result = AsyncSpeculativeDecoder(target, drafter, SSDConfig(draft_length=3, fanout=2, mode=mode)).generate(
        prompt, 40)
    assert torch.equal(result.tokens, reference)
    assert result.acceptance_length > 1.3
    if mode == "async":
        assert result.hits > 0


def test_tiny_sampling_on_two_devices() -> None:
    target = make_tiny_model(seed=0).to("cuda:0")
    drafter = make_tiny_model(seed=0).to("cuda:1")
    prompt = torch.randint(0, VOCAB, (PROMPT_LEN,), generator=torch.Generator().manual_seed(1))
    config = SSDConfig(draft_length=3, fanout=2, temperature=0.8, top_p=0.95, seed=3)
    first, second = (AsyncSpeculativeDecoder(target, drafter, config).generate(prompt, 32) for _ in range(2))
    assert torch.equal(first.tokens, second.tokens)
    assert first.acceptance_length == 4  # identical models: speculative sampling keeps every draft


@pytest.mark.slow
@pytest.mark.skipif(not (_cached(TARGET) and _cached(DRAFTER)), reason="the model pair is not in the HF cache")
def test_pretrained_pair_matches_greedy_generate() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TARGET)
    target = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.float32).to("cuda:0").eval()
    drafter = AutoModelForCausalLM.from_pretrained(DRAFTER, dtype=torch.float32).to("cuda:1").eval()
    decoder = AsyncSpeculativeDecoder(target, drafter, SSDConfig(draft_length=4, fanout=2))
    prompts = ["The capital of France is Paris. The capital of Germany is",
               "def quicksort(values):\n    \"\"\"Sort a list of numbers.\"\"\"\n"]
    lengths = []
    for text in prompts:
        ids = tokenizer(text, return_tensors="pt").input_ids
        expected = target.generate(ids.to("cuda:0"), max_new_tokens=48, do_sample=False)[0, ids.shape[1]:].cpu()
        result = decoder.generate(ids, 48)
        assert torch.equal(result.tokens, expected[: result.tokens.numel()]) and result.tokens.numel() == 48
        lengths.append(result.acceptance_length)
        assert result.hits > 0
    assert max(lengths) > 2
