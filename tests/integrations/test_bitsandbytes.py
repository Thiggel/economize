"""Bases loaded with bitsandbytes (4-bit NF4 / FP4 and 8-bit LLM.int8), on a CUDA device.

The exact path of every drafter is the quantized model itself, so lossless decoding must reproduce the quantized
model's own greedy tokens. The quantized parts compute in float32 here (4-bit ``bnb_4bit_compute_dtype``, 8-bit
without outlier decomposition, whose column choice depends on the other rows of a batch) so that batched and
single-token passes agree. ``adaptation="full"`` trains dense copies of the quantized layers, dequantized once.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import torch

import economize
from economize import DecodeConfig, JumpConfig
from economize.techniques.early_exit import EarlyExitConfig
from helpers import VOCAB, make_tiny_model, perturb, reference_greedy

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
    pytest.mark.skipif(importlib.util.find_spec("bitsandbytes") is None, reason="needs bitsandbytes"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning", "ignore::FutureWarning", "ignore::UserWarning"),
]

QUANTS = ("nf4", "fp4", "int8")
DRAFTERS = {
    "jump_frozen": JumpConfig(target_layer=2, source_layer=4),
    "jump_lora": JumpConfig(target_layer=2, source_layer=4, adaptation="lora", lora_rank=4),
    "jump_full": JumpConfig(target_layer=1, source_layer=3, exit_layer=5, adaptation="full"),
    "exit_lora": EarlyExitConfig(exit_layer=3, adaptation="lora"),
    "exit_full": EarlyExitConfig(exit_layer=3, adaptation="full"),
}


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tiny_bnb")
    make_tiny_model().save_pretrained(path)
    return path


def _config(quant: str) -> Any:
    from transformers import BitsAndBytesConfig

    if quant == "int8":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=0.0)
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=quant, bnb_4bit_compute_dtype=torch.float32)


def _load(path: Path, quant: str) -> Any:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(path, quantization_config=_config(quant), dtype=torch.float32,
                                                 device_map="cuda").eval()
    model.requires_grad_(False)
    model.generation_config.eos_token_id = None
    return model


def _prompts() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(4)
    return [torch.randint(1, VOCAB, (n,), generator=generator).cuda() for n in (12, 5, 9)]


@pytest.mark.parametrize("quant", QUANTS)
def test_base_is_quantized_and_described(tiny_dir: Path, quant: str) -> None:
    import bitsandbytes as bnb

    from economize.integrations.quantized import describe_quantization

    base = _load(tiny_dir, quant)
    kind = bnb.nn.Linear8bitLt if quant == "int8" else bnb.nn.Linear4bit
    assert isinstance(base.model.layers[0].self_attn.q_proj, kind)
    info = describe_quantization(base)
    assert info is not None and info["method"] == "bitsandbytes"
    assert info["config"]["load_in_8bit" if quant == "int8" else "load_in_4bit"]


@pytest.mark.parametrize("drafter", list(DRAFTERS))
@pytest.mark.parametrize("quant", QUANTS)
@torch.no_grad()
def test_lossless_decoding_matches_the_quantized_greedy(tiny_dir: Path, quant: str, drafter: str) -> None:
    base = _load(tiny_dir, quant)
    config = DRAFTERS[drafter]
    model = economize.apply(base, config)
    perturb(model.techniques[config.technique], 0.05, seed=1)
    prompts = _prompts()
    greedy = [reference_greedy(base, p, 16).tolist() for p in prompts]
    results = model.decode_batch(prompts, 16, decoding=DecodeConfig(cycle_length=4))
    assert [r.tokens.tolist() for r in results] == greedy
    lossy = model.decode_batch(prompts, 16, decoding=DecodeConfig(cycle_length=4, lossless=False))
    assert all(len(r.tokens) == 16 for r in lossy)


@pytest.mark.parametrize(("quant", "error"), [("nf4", 0.15), ("int8", 0.01)])
def test_full_copies_are_dense_dequantized_layers(tiny_dir: Path, quant: str, error: float) -> None:
    from economize.modules.dtype import trainable_copy

    base = _load(tiny_dir, quant)
    copy = trainable_copy(base.model.layers[2])
    assert all(type(p) is torch.nn.Parameter and p.dtype == torch.float32 and p.requires_grad
               for p in copy.parameters())
    assert not any(type(m).__module__.startswith("bitsandbytes") for m in copy.modules())
    original = dict(make_tiny_model().model.layers[2].named_parameters())
    for name, param in copy.named_parameters():
        reference = original[name].to(param.device)
        assert param.shape == reference.shape, name
        assert (param - reference).norm() <= error * reference.norm(), name
    if quant == "nf4":  # 4-bit layers dequantize and multiply in float32: the copy computes the same outputs
        x = torch.randn(1, 6, base.config.hidden_size, device="cuda")
        with torch.no_grad():
            torch.testing.assert_close(copy.mlp(x), base.model.layers[2].mlp(x), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("quant", ["nf4", "int8"])
def test_lora_trains_on_a_quantized_base(tiny_dir: Path, quant: str) -> None:
    from economize.train import fit

    base = _load(tiny_dir, quant)
    config = DRAFTERS["jump_lora"]
    model = economize.apply(base, config)
    trainable = [p for p in model.techniques["jump"].parameters() if p.requires_grad]
    before = [p.detach().clone() for p in trainable]
    rows = torch.randint(1, VOCAB, (8, 33), generator=torch.Generator().manual_seed(0))
    fit(model, rows, steps=2, batch_size=4, seq_len=32, lr=1e-3, warmup_steps=0, seed=0)
    assert any(not torch.equal(a, p.detach()) for a, p in zip(before, trainable))
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable)


def test_save_and_load_onto_a_quantized_base(tiny_dir: Path, tmp_path: Path) -> None:
    base = _load(tiny_dir, "nf4")
    model = economize.apply(base, DRAFTERS["jump_lora"])
    perturb(model.techniques["jump"], 0.05, seed=2)
    economize.save(model, tmp_path / "ckpt")
    again = economize.load(_load(tiny_dir, "nf4"), tmp_path / "ckpt")
    prompt = _prompts()[0]
    with torch.no_grad():
        lossy = [m.decode(prompt, 12, decoding=DecodeConfig(cycle_length=4, lossless=False)).tokens.tolist()
                 for m in (model, again)]
    assert lossy[0] == lossy[1]
