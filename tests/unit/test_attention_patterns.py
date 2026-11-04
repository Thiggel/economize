"""The ``"economize"`` kernel on every verification pattern the benchmarks use, in float32 on a CUDA device.

Each test decodes with the kernel and compares the tokens with greedy decoding of the same weights under ``sdpa``:
chains (jump, EAGLE-3, an MTP module, early exit, a draft model, FR-Spec), static trees (Medusa, Hydra), ReDrafter
beams, batches of several rows, the sliding and global layers of Gemma, Mistral and OLMo-3, mixture-of-experts
families, and a model on a second device. Each test also checks through :func:`economize.attention.kernel_usage` that
the decoding steps ran on the kernel. Per-call fallbacks are tested on any device.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

import economize
from economize import DecodeConfig, JumpConfig
from economize.attention import (
    decode_attention,
    economize_attention_forward,
    install,
    kernel_usage,
    reference_attention,
    usage_summary,
)
from economize.techniques.draft_model import DraftModelConfig
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.early_exit import EarlyExitConfig
from economize.techniques.frspec import FRSpecConfig
from economize.techniques.hydra import HydraConfig
from economize.techniques.medusa import MedusaConfig
from economize.techniques.redrafter import RedrafterConfig
from economize.techniques.vendor_mtp import VendorMTPConfig
from helpers import VOCAB, family_skip_reason, make_family_model, make_tiny_model, perturb, reference_greedy

NEW = 14
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _prompts(device: str | torch.device) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(4)
    return [torch.randint(3, VOCAB, (n,), generator=generator).to(device) for n in (11, 5, 8)]


def _check(base: torch.nn.Module, config, decoding: DecodeConfig, device: str = "cuda", std: float = 0.05) -> None:
    """Greedy tokens of ``base`` under sdpa == tokens of ``config`` on ``base`` under the kernel; the steps use it."""
    install()
    base = base.to(device).eval()
    prompts = _prompts(device)
    with torch.no_grad():
        greedy = [reference_greedy(base, p, NEW).tolist() for p in prompts]
        base.set_attn_implementation("economize")
        model = economize.apply(base, config)
        for technique in model.techniques.values():
            perturb(technique, std, seed=1)
        kernel_usage(reset=True)
        results = model.decode_batch(prompts, NEW, decoding=decoding)
        single = model.decode(prompts[1], NEW, decoding=decoding)
    assert [r.tokens.tolist()[-NEW:] for r in results] == greedy
    assert single.tokens.tolist()[-NEW:] == greedy[1]
    summary = usage_summary()
    assert summary["calls"].get("kernel", 0) > 0, summary
    assert set(summary["fallback_reasons"]) <= {"long_query", "no_mask"}, summary  # only prefills may fall back
    model.detach()


CHAINS = {
    "jump": lambda: JumpConfig(target_layer=2, source_layer=4),
    "eagle3": lambda: Eagle3Config(head="shared"),
    "mtp": lambda: VendorMTPConfig(),
    "early_exit": lambda: EarlyExitConfig(exit_layer=3),
    "frspec": lambda: FRSpecConfig(drafter=JumpConfig(target_layer=2, source_layer=4), vocab_size=64,
                                   token_ids=list(range(0, VOCAB, 2))),
}


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graphs"])
@pytest.mark.parametrize("drafter", sorted(CHAINS))
def test_chain_verification(drafter: str, graphs: bool) -> None:
    _check(make_tiny_model(), CHAINS[drafter](), DecodeConfig(cycle_length=5, cuda_graphs=graphs))


@gpu
@pytest.mark.gpu
def test_draft_model_chain(tmp_path) -> None:
    make_tiny_model(seed=7, num_layers=2).save_pretrained(tmp_path)
    config = DraftModelConfig(model=str(tmp_path), dtype="float32", attn_implementation="economize")
    _check(make_tiny_model(), config, DecodeConfig(cycle_length=5), std=0.0)


TREES = {
    "medusa_small": lambda: MedusaConfig(num_heads=5, tree="small"),
    "medusa_63": lambda: MedusaConfig(num_heads=5, tree="mc_sim_7b_63"),
    "hydra_small": lambda: HydraConfig(num_heads=5, num_layers=1, tree="small"),
    "redrafter_beams": lambda: RedrafterConfig(num_layers=1, beam_width=8, beam_length=5, top_k=20),
}


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graphs"])
@pytest.mark.parametrize("drafter", sorted(TREES))
def test_tree_verification(drafter: str, graphs: bool) -> None:
    _check(make_tiny_model(), TREES[drafter](), DecodeConfig(cycle_length=6, cuda_graphs=graphs))


FAMILIES = ["llama", "qwen2", "qwen3", "qwen3_moe", "mixtral", "olmo2", "olmo3", "olmoe", "mistral", "gemma2",
            "gemma3_text", "gemma4_text", "gemma4_text_moe", "gpt_oss"]


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize("drafter", ["eagle3", "early_exit"])
@pytest.mark.parametrize("name", FAMILIES)
def test_family_verification(name: str, drafter: str) -> None:
    reason = family_skip_reason(name)
    if reason is not None:
        pytest.skip(reason)
    _check(make_family_model(name), CHAINS[drafter](), DecodeConfig(cycle_length=4))


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_model_on_a_second_device() -> None:
    torch.cuda.set_device(0)
    _check(make_tiny_model(), CHAINS["jump"](), DecodeConfig(cycle_length=5), device="cuda:1")


def _module() -> torch.nn.Module:
    module = torch.nn.Module()
    module.num_key_value_groups, module.is_causal, module.layer_idx = 2, True, 3
    return module


@pytest.mark.parametrize("case", ["short_mask", "kv_head_mask", "rank3_mask", "query_mismatch"])
def test_odd_masks_fall_back_per_call(case: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(2, 4, 3, 16, device=device)
    k, v = torch.randn(2, 2, 50, 16, device=device), torch.randn(2, 2, 50, 16, device=device)
    mask = torch.ones(2, 1, 3, 50, dtype=torch.bool, device=device).tril(diagonal=47)
    reason = {"short_mask": "mask_keys", "kv_head_mask": "mask_shape", "rank3_mask": "mask_rank",
              "query_mismatch": "mask_shape"}[case]
    if case == "short_mask":
        mask, k, v = mask[..., :40], k, v  # fewer mask slots than keys: sdpa raises, the kernel must not guess
    elif case == "kv_head_mask":
        mask = mask.expand(2, 2, 3, 50)  # one mask per KV head: neither 1 nor the 4 query heads
    elif case == "rank3_mask":
        mask = mask[:, 0]
    else:
        mask = mask[:, :, :2]
    kernel_usage(reset=True)
    with contextlib.suppress(RuntimeError):  # sdpa may reject the shape; the kernel must not have been used
        economize_attention_forward(_module(), q, k, v, mask, scaling=0.25)
    usage = kernel_usage()
    if device == "cpu":
        reason = "cpu"
    assert usage == [{"layer": 3, "queries": 3, "path": "sdpa", "reason": reason, "calls": 1}]


@gpu
@pytest.mark.gpu
def test_usage_log_records_the_kernel_by_layer_and_length() -> None:
    q = torch.randn(1, 4, 5, 32, device="cuda")
    k, v = torch.randn(1, 2, 70, 32, device="cuda"), torch.randn(1, 2, 70, 32, device="cuda")
    mask = torch.ones(1, 1, 5, 80, dtype=torch.bool, device="cuda")  # longer than the keys: sliced
    kernel_usage(reset=True)
    out, _ = economize_attention_forward(_module(), q, k, v, mask, scaling=0.2)
    ref = reference_attention(q, k, v, mask[..., :70], 0.2)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=0)
    assert kernel_usage() == [{"layer": 3, "queries": 5, "path": "kernel", "reason": "", "calls": 1}]
    assert usage_summary() == {"calls": {"kernel": 1}, "fallback_reasons": {}, "queries": {"kernel": [5]}}


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_sinks_join_the_denominator(dtype: torch.dtype) -> None:
    q = torch.randn(2, 8, 5, 64, device="cuda").to(dtype)
    k, v = torch.randn(2, 2, 300, 64, device="cuda").to(dtype), torch.randn(2, 2, 300, 64, device="cuda").to(dtype)
    mask = torch.ones(2, 1, 5, 300, dtype=torch.bool, device="cuda").tril(diagonal=290)
    mask[1, :, 2] = False  # a query that sees no key: all mass on the sink, output zero
    sinks = torch.randn(8, device="cuda") * 3
    out = decode_attention(q, k, v, mask, 0.125, sinks=sinks)
    ref = reference_attention(q, k, v, mask, 0.125, sinks=sinks)
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-5 if dtype == torch.float32 else 1.6e-2, rtol=0)
    assert out[1, 2].abs().max() == 0


@gpu
@pytest.mark.gpu
def test_gpt_oss_prefill_falls_back_to_eager_with_sinks() -> None:
    """A long prefill with sinks must run on the model's eager function (sdpa would drop the sinks)."""
    reason = family_skip_reason("gpt_oss")
    if reason is not None:
        pytest.skip(reason)
    install()
    base = make_family_model("gpt_oss").cuda().eval()
    prompt = torch.randint(3, 100, (80,), generator=torch.Generator().manual_seed(2)).cuda()
    with torch.no_grad():
        greedy = reference_greedy(base, prompt, NEW).tolist()
        base.set_attn_implementation("economize")
        model = economize.apply(base, CHAINS["early_exit"]())
        for technique in model.techniques.values():
            perturb(technique, 0.05, seed=1)
        kernel_usage(reset=True)
        result = model.decode(prompt, NEW, decoding=DecodeConfig(cycle_length=4))
    assert result.tokens.tolist()[-NEW:] == greedy[-NEW:]
    paths = {(r["queries"], r["path"]) for r in kernel_usage()}
    assert (80, "eager") in paths and any(path == "kernel" for _, path in paths), paths
    assert "sdpa" not in usage_summary()["calls"]
    model.detach()
