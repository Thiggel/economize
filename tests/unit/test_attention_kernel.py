"""The ``"economize"`` attention implementation (``economize.attention``).

The kernel is compared with :func:`economize.attention.reference_attention` (fp32 tolerance 1e-5, bf16 one rounding
step of the output scale) and, end to end, with ``sdpa``: a tiny model loaded with ``attn_implementation="economize"``
decodes exactly the greedy tokens of the ``sdpa`` model, with and without CUDA graphs, in ordinary and speculative
decoding. The kernel tests need ``-m gpu``; the registry and fallback tests run anywhere.
"""

from __future__ import annotations

import pytest
import torch
from torch.overrides import TorchFunctionMode

import economize
from economize import DecodeConfig, JumpConfig
from economize.arch import adapter_for
from economize.attention import (
    ATTENTION_KERNELS,
    attention_family,
    decode_attention,
    economize_attention_forward,
    install,
    kernel_calls,
    reference_attention,
)
from helpers import VOCAB, make_tiny_model, perturb, reference_greedy


def _mask(batch: int, queries: int, keys: int, filled: int, window: int | None, per_head: int = 1) -> torch.Tensor:
    pos = torch.arange(keys)
    starts = filled - torch.arange(batch) * 7  # rows at different positions
    qp = starts[:, None] + torch.arange(queries)[None, :]
    visible = pos[None, None, :] <= qp[..., None]
    if window is not None:
        visible &= pos[None, None, :] > qp[..., None] - window
    visible = visible[:, None].expand(batch, per_head, queries, keys).clone()
    if per_head > 1:  # hide a few slots for some heads only (evicted slots)
        visible[:, ::2, :, 3:40] = False
    return visible

def test_registry_reports_the_sdpa_family() -> None:
    assert "economize" in ATTENTION_KERNELS
    assert attention_family("economize") == "sdpa"
    assert attention_family("eager") == "eager"
    install()
    model = make_tiny_model("economize")
    assert model.config._attn_implementation == "economize"
    assert adapter_for(model).attn_implementation == "sdpa"

@torch.no_grad()
def test_cpu_model_falls_back_and_decodes_greedy(prompt: torch.Tensor) -> None:
    install()
    reference = reference_greedy(make_tiny_model("sdpa"), prompt, 12)
    model = economize.apply(make_tiny_model("economize"), JumpConfig(target_layer=2, source_layer=4))
    perturb(model.techniques["jump"], 0.05)
    out = model.decode_batch([prompt], 12, decoding=DecodeConfig(cycle_length=4))
    assert out[0].tokens.tolist() == reference.tolist()

class _Tap(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))

@pytest.mark.parametrize("case", ["tap", "long_query", "float_mask"])
def test_fallback_conditions(case: str) -> None:
    module = torch.nn.Module()
    module.num_key_value_groups, module.is_causal = 2, True
    queries = 80 if case == "long_query" else 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(1, 4, queries, 16, device=device)
    k, v = torch.randn(1, 2, 100, 16, device=device), torch.randn(1, 2, 100, 16, device=device)
    mask = _mask(1, queries, 100, 90, None).to(device)
    if case == "float_mask":
        mask = torch.zeros(mask.shape, device=device).masked_fill(~mask, float("-inf"))
    before = kernel_calls()["fallback"]
    if case == "tap":
        with _Tap():
            economize_attention_forward(module, q, k, v, mask, scaling=0.25)
    else:
        economize_attention_forward(module, q, k, v, mask, scaling=0.25)
    assert kernel_calls()["fallback"] == before + 1

CASES = [  # batch, q heads, kv heads, queries, keys, head dim, window, softcap, per-head mask
    (1, 32, 16, 1, 4352, 256, 1024, None, 1),  # Gemma-4-31B sliding layer, one decoding step
    (1, 32, 4, 1, 4352, 512, None, None, 1),  # Gemma-4-31B global layer
    (2, 32, 16, 9, 4352, 256, 1024, None, 1),  # a verification pass over 8 drafts
    (2, 32, 4, 9, 4352, 512, None, None, 1),
    (3, 14, 2, 5, 777, 64, None, 30.0, 1),  # Qwen-like geometry with a soft cap
    (4, 8, 8, 16, 300, 128, None, None, 8),  # per-head mask (evicted slots), no GQA
    (1, 16, 2, 64, 1000, 96, None, None, 1),  # the longest query, a head dim that is no power of two
]

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("case", CASES, ids=[f"b{c[0]}h{c[1]}k{c[2]}q{c[3]}d{c[5]}" for c in CASES])
@pytest.mark.gpu
def test_kernel_matches_reference(case: tuple, dtype: torch.dtype) -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    b, hq, hkv, q_len, keys, d, window, softcap, per_head = case
    gen = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(b, q_len, hq, d, device="cuda", generator=gen).to(dtype).transpose(1, 2)  # HF strides
    k = torch.randn(b, hkv, keys, d, device="cuda", generator=gen).to(dtype)
    v = torch.randn(b, hkv, keys, d, device="cuda", generator=gen).to(dtype)
    mask = _mask(b, q_len, keys, keys - 400, window, per_head).cuda()
    try:
        out = decode_attention(q, k, v, mask, d**-0.5, softcap=softcap)
    except Exception as error:  # Turing has 64 KiB of shared memory, too little for fp32 tiles at head dim 512
        if type(error).__name__ == "OutOfResources":
            pytest.skip(str(error))
        raise
    ref = reference_attention(q, k, v, mask, d**-0.5, softcap=softcap)
    tol = 1e-5 if dtype == torch.float32 else 1.6e-2
    torch.testing.assert_close(out.float(), ref.float(), atol=tol, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_broadcast_batch_mask_and_hidden_rows() -> None:
    q = torch.randn(3, 8, 2, 64, device="cuda")
    k, v = torch.randn(3, 2, 200, 64, device="cuda"), torch.randn(3, 2, 200, 64, device="cuda")
    mask = _mask(1, 2, 200, 150, None).cuda()
    mask[:, :, 1] = False  # the second query sees nothing
    out = decode_attention(q, k, v, mask.expand(3, 1, 2, 200), 0.125)
    ref = reference_attention(q, k, v, mask.expand(3, 1, 2, 200), 0.125)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=0)
    assert out[:, 1].abs().max() == 0

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available() or not hasattr(torch, "float8_e4m3fn"), reason="needs fp8")
@pytest.mark.parametrize("fp8", ["float8_e4m3fn", "float8_e5m2"])
def test_fp8_keys_and_values_are_upcast(fp8: str) -> None:
    dtype = getattr(torch, fp8)
    q = torch.randn(2, 16, 3, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 4, 600, 128, device="cuda").to(dtype)
    v = torch.randn(2, 4, 600, 128, device="cuda").to(dtype)
    mask = _mask(2, 3, 600, 500, None).cuda()
    out = decode_attention(q, k, v, mask, 128**-0.5)
    ref = reference_attention(q, k.to(torch.bfloat16), v.to(torch.bfloat16), mask, 128**-0.5)
    torch.testing.assert_close(out.float(), ref.float(), atol=1.6e-2, rtol=0)

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available() or not hasattr(torch, "float8_e4m3fn"), reason="needs fp8")
@pytest.mark.parametrize("fp8", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fp8_bytes_decode_exactly(fp8: str, dtype: torch.dtype) -> None:
    """With one visible key the output is that key's value, so every finite byte must decode to its exact value."""
    fmt = getattr(torch, fp8)
    raw = torch.arange(256, dtype=torch.uint8, device="cuda")
    finite = torch.isfinite(raw.view(fmt).float())
    raw = torch.where(finite, raw, torch.zeros_like(raw))
    v = raw.view(fmt).view(1, 1, 1, 256).expand(1, 1, 4, 256).contiguous()
    k = torch.zeros(1, 1, 4, 256, device="cuda").to(fmt)
    q = torch.zeros(1, 2, 1, 256, device="cuda", dtype=dtype)
    mask = torch.tensor([True, False, False, False], device="cuda").view(1, 1, 1, 4)
    out = decode_attention(q, k, v, mask, 1.0)
    expected = raw.view(fmt).to(dtype).float()
    assert torch.equal(out[0, 0, 0].float(), expected) and torch.equal(out[0, 0, 1].float(), expected)


def test_fp8_kv_falls_back_with_an_upcast() -> None:
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("needs fp8")
    module = torch.nn.Module()
    module.num_key_value_groups, module.is_causal = 2, True
    q = torch.randn(1, 4, 80, 16)  # 80 queries: a prefill, which runs on sdpa
    k, v = torch.randn(1, 2, 100, 16).to(torch.float8_e4m3fn), torch.randn(1, 2, 100, 16).to(torch.float8_e4m3fn)
    mask = _mask(1, 80, 100, 20, None)
    out, _ = economize_attention_forward(module, q, k, v, mask, scaling=0.25)
    ref = reference_attention(q, k.float(), v.float(), mask, 0.25)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=0)
    assert ATTENTION_KERNELS.get("economize").fp8_kv


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("graphs", [False, True])
@torch.no_grad()
def test_economize_model_decodes_like_sdpa(graphs: bool) -> None:
    install()
    gen = torch.Generator().manual_seed(3)
    prompts = [torch.randint(3, VOCAB, (n,), generator=gen).cuda() for n in (13, 5, 9)]
    base = make_tiny_model("sdpa").cuda()
    greedy = [reference_greedy(base, p, 16).tolist() for p in prompts]
    model = economize.apply(make_tiny_model("economize").cuda(), JumpConfig(target_layer=2, source_layer=4))
    perturb(model.techniques["jump"], 0.05)
    before = kernel_calls()["kernel"]
    for cycle in (1, 4):
        decoding = DecodeConfig(cycle_length=cycle, cuda_graphs=graphs)
        for _ in range(2):  # warm-up / capture, then replay
            results = model.decode_batch(prompts, 16, decoding=decoding)
            assert [r.tokens.tolist() for r in results] == greedy
    assert kernel_calls()["kernel"] > before
