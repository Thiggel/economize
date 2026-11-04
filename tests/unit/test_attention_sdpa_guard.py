"""The ``sdpa`` fallback with a single K/V head and wide heads (``economize.attention.sdpa_guard``).

``transformers``' ``repeat_kv`` turns one K/V head into a stride-0 view across the query heads. PyTorch 2.13.0
(cu130) computes masked attention over such K/V wrongly on CUDA for head dims above 128 when the query length is
one more than a multiple of 32 (33, 65, ..., 673), with errors of order one in fp32. Gemma-4 E2B (one K/V head,
head dims 256 and 512) hits it in every prefill of such a length. The guard hands ``sdpa`` contiguous K/V instead.

The decision runs anywhere; the kernel test needs a GPU (``-m gpu``) and fails on an unguarded PyTorch 2.13.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from economize.attention import interface
from economize.attention.sdpa_guard import needs_dense_kv, sdpa_attention_forward


class _Attention(torch.nn.Module):
    def __init__(self, groups: int) -> None:
        super().__init__()
        self.num_key_value_groups = groups
        self.is_causal = True
        self.layer_idx = 0


def _inputs(queries: int, keys: int, dim: int, heads: int = 8, kv_heads: int = 1, device: str = "cpu"):
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(1, heads, queries, dim, device=device, generator=g)
    k = torch.randn(1, kv_heads, keys, dim, device=device, generator=g)
    v = torch.randn(1, kv_heads, keys, dim, device=device, generator=g)
    position = torch.arange(keys - queries, keys, device=device)
    mask = (torch.arange(keys, device=device)[None] <= position[:, None])[None, None]
    return q, k, v, mask


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    groups = q.shape[1] // k.shape[1]
    k, v = (x.double().repeat_interleave(groups, 1) for x in (k, v))
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q.double(), k, v, attn_mask=mask).transpose(1, 2)


# ---- decision (no kernels) ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("groups", "kv_heads", "queries", "dim", "expected"),
    [
        (8, 1, 673, 256, True),  # Gemma-4 E2B sliding layer, prefill
        (8, 1, 673, 512, True),  # Gemma-4 E2B global layer
        (8, 1, 33, 256, True),
        (8, 1, 32, 256, False),  # short queries (decode steps, verification) never hit the kernel error
        (8, 1, 673, 128, False),  # head dims up to 128 are computed correctly
        (4, 2, 673, 256, False),  # repeat_kv copies when there are several K/V heads
        (1, 8, 673, 256, False),  # no grouping
    ],
)
def test_decision(groups: int, kv_heads: int, queries: int, dim: int, expected: bool) -> None:
    q, k, _, _ = _inputs(queries, queries, dim, heads=groups * kv_heads, kv_heads=kv_heads)
    assert needs_dense_kv(_Attention(groups), q, k, cuda_only=False) is expected
    assert not needs_dense_kv(_Attention(groups), q, k)  # CPU tensors: PyTorch's CPU kernels are correct


def test_guarded_call_matches_reference_on_cpu() -> None:
    q, k, v, mask = _inputs(65, 80, 256)
    out, _ = sdpa_attention_forward(_Attention(8), q, k, v, mask, dense_kv=True)
    assert out.shape == (1, 65, 8, 256)
    assert out.is_contiguous()
    torch.testing.assert_close(out.double(), _reference(q, k, v, mask), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kv_heads", [1, 2, 8])
def test_long_masked_query_attends_in_chunks_with_the_same_result(monkeypatch, kv_heads: int) -> None:
    from economize.attention import sdpa_guard

    q, k, v, mask = _inputs(70, 96, 64, kv_heads=kv_heads)
    calls = []

    def forward(module, query, *args, **kwargs):
        calls.append(query.shape[2])
        return original(module, query, *args, **kwargs)

    import transformers.integrations.sdpa_attention as hf

    original = hf.sdpa_attention_forward
    monkeypatch.setattr(sdpa_guard, "QUERY_CHUNK", 32)
    monkeypatch.setattr(hf, "sdpa_attention_forward", forward)
    out, _ = sdpa_attention_forward(_Attention(8 // kv_heads), q, k, v, mask)
    assert calls == [32, 32, 6]
    assert out.shape == (1, 70, 8, 64)
    torch.testing.assert_close(out.double(), _reference(q, k, v, mask), atol=1e-5, rtol=1e-5)


# ---- kernels --------------------------------------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize(
    ("queries", "keys", "dim"), [(673, 673, 256), (673, 673, 512), (33, 512, 256), (65, 1024, 512)]
)
def test_fallback_is_exact_with_one_kv_head(queries: int, keys: int, dim: int) -> None:
    q, k, v, mask = _inputs(queries, keys, dim, device="cuda")
    expected = _reference(q, k, v, mask)
    out, _ = interface._sdpa(_Attention(8), q, k, v, mask, dropout=0.0, scaling=None, is_causal=None)
    assert float((out.double() - expected).abs().max()) < 1e-4


def test_install_registers_the_guard_under_sdpa() -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from economize.attention import sdpa_guard

    sdpa_guard.install()
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is sdpa_attention_forward
