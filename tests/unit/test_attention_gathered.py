"""The decode kernel over a compacted cache (``economize.attention.gathered``).

Each row keeps a random subset of its prompt entries per KV head (a different number per head and row). The kernel
gets the kept entries in compact buffers and the dense buffers with NaN in every prompt slot and in the padding of
the compact buffers, so any read of an entry it should skip would show. It must agree with
:func:`~economize.attention.reference_attention` and :func:`~economize.attention.decode_attention` over the dense
buffers with the evicted entries masked. Kernel tests need ``-m gpu``.
"""

from __future__ import annotations

import pytest
import torch

from economize.attention import GatheredKV, decode_attention, gathered_decode_attention, reference_attention
from economize.errors import ConfigError

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")]

BATCH, Q_HEADS, KV_HEADS, KEYS = 3, 8, 2, 300
PROMPT = [200, 100, 30]  # prompt end per row (the first dense slot read)
FILLED = [260, 140, 30]  # first slot of the queries per row


def _mask(queries: int, kind: str | None) -> torch.Tensor | None:
    if kind is None:
        return None
    pos = torch.arange(KEYS)
    qp = torch.tensor(FILLED)[:, None] + torch.arange(queries)[None, :]
    visible = pos[None, None, :] <= qp[..., None]
    if kind == "tree":  # a draft tree: query i sees the committed prefix and its ancestors only
        start = torch.tensor(FILLED)[:, None, None]
        drafted = (pos[None, None, :] >= start) & visible
        parent_ok = (pos[None, None, :] - start) % 2 == (torch.arange(queries)[None, :, None] % 2)
        visible &= ~drafted | parent_ok | (pos[None, None, :] == qp[..., None])
    if kind == "window":  # a sliding-window layer of 64 slots, which also cuts into the compacted prompt
        visible &= pos[None, None, :] > qp[..., None] - 64
    heads = Q_HEADS if kind == "per_head" else 1
    visible = visible[:, None].expand(BATCH, heads, queries, KEYS).clone()
    if kind == "per_head":
        visible[:, ::2, :, 10:40] = False
    return visible.cuda()


def _compact(
    key: torch.Tensor, value: torch.Tensor, budget: int, seed: int
) -> tuple[GatheredKV, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A random selection of at most ``budget`` prompt entries per row and KV head, its compact buffers, the keep
    mask over all slots, and the dense buffers with NaN in every prompt slot."""
    generator = torch.Generator().manual_seed(seed)
    slot = torch.arange(KEYS)
    prompt = slot[None, None, :] < torch.tensor(PROMPT)[:, None, None]
    ranked = torch.rand(BATCH, KV_HEADS, KEYS, generator=generator).masked_fill(~prompt, 2.0)
    order = ranked.argsort(-1)
    keep = torch.ones(BATCH, KV_HEADS, KEYS, dtype=torch.bool)
    for b in range(BATCH):
        for h in range(KV_HEADS):
            kept = min(PROMPT[b], budget - 7 * h)  # a different count per head
            keep[b, h, order[b, h, kept : PROMPT[b]]] = False
    kept = keep & prompt
    count = kept.sum(-1).to(torch.int32)
    capacity = int(count.max())
    index = (~kept).to(torch.int8).argsort(dim=-1, stable=True)[..., :capacity].to(torch.int32)
    valid = torch.arange(capacity)[None, None, :] < count[..., None]
    idx = index.long().cuda()
    ck = torch.gather(key, 2, idx[..., None].expand(-1, -1, -1, key.shape[3])).masked_fill(~valid[..., None].cuda(),
                                                                                           float("nan"))
    cv = torch.gather(value, 2, idx[..., None].expand(-1, -1, -1, value.shape[3])).masked_fill(
        ~valid[..., None].cuda(), float("nan"))
    start = torch.tensor(PROMPT, dtype=torch.int32)
    dropped = int((start[:, None] - count).min())
    gathered = GatheredKV(ck.contiguous(), cv.contiguous(), index.cuda(), count.cuda(), start.cuda(), dropped)
    nan = prompt[..., None].cuda()
    return gathered, keep.cuda(), key.masked_fill(nan, float("nan")), value.masked_fill(nan, float("nan"))


CASES = [
    # dtype, queries, mask kind, softcap, sinks, head dim, value head dim, budget
    (torch.float32, 1, None, None, False, 64, 64, 40),
    (torch.bfloat16, 1, "chain", None, False, 128, 128, 40),
    (torch.bfloat16, 5, "chain", None, False, 128, 128, 90),
    (torch.bfloat16, 6, "tree", 30.0, False, 64, 64, 40),
    (torch.float16, 3, "per_head", None, True, 64, 64, 64),
    (torch.float32, 4, "tree", None, True, 128, 128, 17),
    (torch.bfloat16, 2, "window", None, False, 64, 48, 40),
    (torch.float32, 2, "chain", None, False, 64, 64, 1000),  # nothing evicted
]


@pytest.mark.parametrize("case", CASES, ids=[f"{str(c[0])[6:]}-q{c[1]}-{c[2]}-c{c[7]}" for c in CASES])
def test_kernel_reads_the_compact_entries(case: tuple) -> None:
    dtype, queries, mask_kind, softcap, use_sinks, d, dv, budget = case
    torch.manual_seed(0)
    query = torch.randn(BATCH, Q_HEADS, queries, d, device="cuda", dtype=dtype)
    key = (torch.randn(BATCH, KV_HEADS, KEYS, d, device="cuda") * 2).to(dtype)
    value = torch.randn(BATCH, KV_HEADS, KEYS, dv, device="cuda").to(dtype)
    gathered, keep, key_nan, value_nan = _compact(key, value, budget, seed=len(mask_kind or ""))
    base = _mask(queries, mask_kind)
    group = Q_HEADS // KV_HEADS
    visible = keep.repeat_interleave(group, dim=1)[:, :, None, :].expand(BATCH, Q_HEADS, queries, KEYS)
    mask_dense = visible if base is None else base & visible
    sinks = torch.randn(Q_HEADS, device="cuda") if use_sinks else None
    scale = d**-0.5
    # Without a mask of its own the kernel sees every slot (Q = 1); the gathered cache then reads only the kept ones.
    got = gathered_decode_attention(query, key_nan, value_nan, base, scale, gathered, softcap=softcap, sinks=sinks)
    assert torch.isfinite(got).all()
    ref = reference_attention(query, key, value, mask_dense, scale, softcap=softcap, sinks=sinks)
    tol = 1e-5 if dtype == torch.float32 else 1.6e-2
    torch.testing.assert_close(got.float(), ref.float(), atol=tol, rtol=tol)
    dense = decode_attention(query, key, value, mask_dense, scale, softcap=softcap, sinks=sinks)
    torch.testing.assert_close(got.float(), dense.float(), atol=tol, rtol=tol)


def test_rejects_mismatched_compact_buffers() -> None:
    query = torch.randn(1, 4, 1, 64, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 2, 96, 64, device="cuda", dtype=torch.bfloat16)
    count = torch.full((1, 2), 8, dtype=torch.int32, device="cuda")
    start = torch.full((1,), 32, dtype=torch.int32, device="cuda")
    index = torch.zeros(1, 2, 8, dtype=torch.int32, device="cuda")
    wrong = GatheredKV(key[:, :, :8].float(), key[:, :, :8], index, count, start, 24)
    with pytest.raises(ConfigError, match="gathered key"):
        gathered_decode_attention(query, key, key, None, 0.125, wrong)
    wrong = GatheredKV(key[:, :, :8], key[:, :, :8], index, count[:, :1], start, 24)
    with pytest.raises(ConfigError, match="gathered count"):
        gathered_decode_attention(query, key, key, None, 0.125, wrong)
