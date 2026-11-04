"""KIVI's paged kernels for vLLM (:mod:`economize.backends.vllm.kivi.kernels`) against economize's HF definitions.

* :func:`quantize_committed` writes, over vLLM's pages, bitwise the cells of :func:`write_cells` (the torch
  definition, which reuses :func:`economize.techniques.kv_quant.kivi.quantize`), both when a request is quantized in
  one pass and when its tokens arrive a few at a time as in decoding with speculative verification.
* :func:`paged_kivi_attention` over those pages gives the output of
  :func:`economize.attention.kivi.kivi_decode_attention` (economize's ``kivi_fused`` kernel) over the same codes,
  scales and dense entries laid out as the HF runtime holds them, for varlen batches of decoding and verification
  queries, GQA, sinks and a soft cap: bitwise in float32 (IEEE dots), within one unit in the last place in float16
  and bfloat16, where the two kernels' tensor-core dots round their float32 partial sums differently (the softmax
  maxima and sums agree bitwise; the value partials differ at float32 rounding).

Needs a CUDA device and triton; no vLLM.
"""

from __future__ import annotations

import importlib.util
import math

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
    pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="needs triton"),
]

DEVICE = "cuda"


def _layout(bits: int, group: int, residual: int, dim: int, dtype: torch.dtype):
    from economize.backends.vllm.kivi.layout import KIVISettings, cell_layout

    return cell_layout(dim, dtype, KIVISettings(bits=bits, group_size=group, residual_length=residual))


def _pages(seq_lens: list[int], page: int, heads: int, layout, dtype: torch.dtype, seed: int):
    """An empty paged cache and a block table that gives every request a shuffled set of pages."""
    need = [-(-s // page) for s in seq_lens]
    total = sum(need) + 3
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(total, generator=gen)
    table = torch.zeros(len(seq_lens), max(max(need), 1), dtype=torch.int32)
    at = 0
    for b, n in enumerate(need):
        table[b, :n] = order[at:at + n].to(torch.int32)
        at += n
    cache = torch.zeros(total, heads, page, layout.size // layout.element_size, dtype=dtype, device=DEVICE)
    return cache, table.to(DEVICE)


def _slots(table_row: torch.Tensor, n: int, page: int, start: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.arange(start, start + n, device=DEVICE)
    return table_row[t // page].long(), t % page


def _gather(cache: torch.Tensor, table_row: torch.Tensor, n: int) -> torch.Tensor:
    """The cells ``uint8`` ``[H, n, size]`` of a request's first ``n`` tokens."""
    pages, slots = _slots(table_row, n, cache.shape[2])
    return cache.view(torch.uint8)[pages, :, slots].permute(1, 0, 2)


def _scatter(cache: torch.Tensor, table_row: torch.Tensor, cells: torch.Tensor) -> None:
    pages, slots = _slots(table_row, cells.shape[1], cache.shape[2])
    cache.view(torch.uint8)[pages, :, slots] = cells.permute(1, 0, 2)


def _write_dense(cache: torch.Tensor, table_row: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                 start: int = 0) -> None:
    """vLLM's cache write: the dense K/V of a request's tokens from position ``start`` on (flags stay 0)."""
    pages, slots = _slots(table_row, keys.shape[1], cache.shape[2], start)
    dim = keys.shape[-1]
    cache[pages, :, slots, :dim] = keys.permute(1, 0, 2)
    cache[pages, :, slots, dim:2 * dim] = values.permute(1, 0, 2)


def _kv(heads: int, tokens: int, dim: int, dtype: torch.dtype, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    offset = torch.randn(heads, 1, dim, generator=gen, device=DEVICE) * 3  # KIVI's per-channel key outliers
    keys = torch.randn(heads, tokens, dim, generator=gen, device=DEVICE) + offset
    values = torch.randn(heads, tokens, dim, generator=gen, device=DEVICE)
    return keys.to(dtype), values.to(dtype)


def _ulps(got: torch.Tensor, want: torch.Tensor) -> float:
    """The largest distance of ``got`` from ``want`` in units in the last place of ``want``'s dtype."""
    info = torch.finfo(want.dtype)
    ulp = info.eps * torch.exp2(torch.floor(torch.log2(want.float().abs().clamp_min(info.tiny))))
    return float(((got.float() - want.float()).abs() / ulp).max())


def _varlen(seq_lens: list[int], qlens: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    starts = [0]
    for q in qlens:
        starts.append(starts[-1] + q)
    return (torch.tensor(starts, dtype=torch.int32, device=DEVICE),
            torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE))


QUANT_CASES = [
    # bits, dtype, head dim, group, residual, page
    (2, torch.bfloat16, 64, 32, 64, 32),
    (4, torch.bfloat16, 128, 32, 32, 32),
    (8, torch.bfloat16, 64, 16, 48, 16),
    (2, torch.float16, 128, 64, 128, 64),
    (4, torch.float32, 64, 32, 64, 64),
    (8, torch.float32, 32, 64, 64, 64),  # value groups of the whole head (32 % 64 != 0)
]


@pytest.mark.parametrize("case", QUANT_CASES, ids=lambda c: "-".join(str(x).replace("torch.", "") for x in c))
def test_quantize_committed_writes_the_reference_cells(case: tuple) -> None:
    from economize.backends.vllm.kivi.kernels import quantize_committed
    from economize.backends.vllm.kivi.layout import write_cells

    bits, dtype, dim, group, residual, page = case
    layout = _layout(bits, group, residual, dim, dtype)
    heads = 2
    seq_lens, qlens, prefilling = [301, 77, residual + 1, 150, 9], [1, 4, 1, 6, 2], [0, 0, 0, 1, 0]
    cache, table = _pages(seq_lens, page, heads, layout, dtype, seed=bits)
    kv = [_kv(heads, s, dim, dtype, seed=10 * b + bits) for b, s in enumerate(seq_lens)]
    for b, (keys, values) in enumerate(kv):
        _write_dense(cache, table[b], keys, values)
    qsl, seq = _varlen(seq_lens, qlens)
    quantize_committed(cache, layout, residual, table, qsl, seq,
                       torch.tensor(prefilling, dtype=torch.int32, device=DEVICE))
    for b, (keys, values) in enumerate(kv):
        c = 0 if prefilling[b] else seq_lens[b] - qlens[b]
        want = write_cells(keys, values, (c // residual) * residual, max(0, c - residual), layout)
        got = _gather(cache, table[b], seq_lens[b])
        assert torch.equal(got, want), f"request {b}: {(got != want).nonzero()[:8].tolist()}"


@pytest.mark.parametrize("case", QUANT_CASES[:4], ids=lambda c: "-".join(str(x).replace("torch.", "") for x in c))
def test_quantize_committed_is_incremental(case: tuple) -> None:
    """Tokens committed a few at a time (decoding, accepted drafts) end in the cells of a single pass."""
    from economize.backends.vllm.kivi.kernels import quantize_committed
    from economize.backends.vllm.kivi.layout import write_cells

    bits, dtype, dim, group, residual, page = case
    layout = _layout(bits, group, residual, dim, dtype)
    heads, final = 2, [260, 203]
    cache, table = _pages(final, page, heads, layout, dtype, seed=7)
    kv = [_kv(heads, s, dim, dtype, seed=b + 100) for b, s in enumerate(final)]
    gen = torch.Generator().manual_seed(bits)
    seq = [17, 40]  # the prompts
    for b, (keys, values) in enumerate(kv):
        _write_dense(cache, table[b], keys[:, :seq[b]], values[:, :seq[b]])
    while True:
        # a pass that appends 1-6 tokens to each request (finished requests decode a padding query of length 0)
        steps = [0 if s >= f else min(f - s, int(torch.randint(1, 7, (1,), generator=gen))) for s, f in
                 zip(seq, final, strict=True)]
        if not any(steps):
            break
        for b, (keys, values) in enumerate(kv):
            _write_dense(cache, table[b], keys[:, seq[b]:seq[b] + steps[b]], values[:, seq[b]:seq[b] + steps[b]],
                         seq[b])
            seq[b] += steps[b]
        qsl, lens = _varlen(seq, steps)
        quantize_committed(cache, layout, residual, table, qsl, lens, torch.zeros(2, dtype=torch.int32,
                                                                                 device=DEVICE))
    for b, (keys, values) in enumerate(kv):
        c = final[b] - 1  # the last pass committed all tokens but the newest of the request that finished last
        got = _gather(cache, table[b], final[b])
        k_to = (got[:, :, layout.off_flag] != 0).sum(1)[0].item()
        v_to = (got[:, :, layout.off_flag + 1] != 0).sum(1)[0].item()
        assert k_to % group == 0 and k_to <= (c // residual) * residual and v_to <= max(0, c - residual)
        assert torch.equal(got, write_cells(keys, values, k_to, v_to, layout)), f"request {b}"
    # one more pass quantizes everything a single pass at the final length would
    qsl, lens = _varlen(final, [1, 1])
    quantize_committed(cache, layout, residual, table, qsl, lens, torch.zeros(2, dtype=torch.int32, device=DEVICE))
    for b, (keys, values) in enumerate(kv):
        c = final[b] - 1
        want = write_cells(keys, values, (c // residual) * residual, max(0, c - residual), layout)
        assert torch.equal(_gather(cache, table[b], final[b]), want), f"request {b}"


def _hf_operands(cache: torch.Tensor, table: torch.Tensor, layout, seq_lens: list[int], qlens: list[int],
                 residual: int, keys: int, dtype: torch.dtype):
    """The dense K/V and :class:`KIVIPacked` that economize's HF runtime holds for the same cells."""
    from economize.attention.kivi import KIVIPacked

    batch, heads, dim, g = len(seq_lens), cache.shape[1], layout.head_size, layout.group_size
    pb = 8 // layout.bits
    key = torch.zeros(batch, heads, keys, dim, dtype=dtype, device=DEVICE)
    value = torch.zeros_like(key)
    kc = torch.zeros(batch, heads, keys, dim // pb, dtype=torch.uint8, device=DEVICE)
    vc = torch.zeros_like(kc)
    ks = torch.zeros(batch, heads, -(-keys // g), dim, dtype=torch.float16, device=DEVICE)
    kz = torch.zeros_like(ks)
    vs = torch.zeros(batch, heads, keys, layout.value_groups, dtype=torch.float16, device=DEVICE)
    vz = torch.zeros_like(vs)
    kq, vq = [], []
    for b, n in enumerate(seq_lens):
        cells = _gather(cache, table[b], n)
        key[b, :, :n] = cells[:, :, :layout.off_v].contiguous().view(dtype)
        value[b, :, :n] = cells[:, :, layout.off_v:layout.off_vs].contiguous().view(dtype)
        kc[b, :, :n] = cells[:, :, layout.off_kc:layout.off_vc]
        vc[b, :, :n] = cells[:, :, layout.off_vc:layout.off_flag]
        vs[b, :, :n] = cells[:, :, layout.off_vs:layout.off_vz].contiguous().view(torch.float16)
        vz[b, :, :n] = cells[:, :, layout.off_vz:layout.off_ka].contiguous().view(torch.float16)
        groups = n // g
        aux = cells[:, :groups * g, layout.off_ka:layout.off_kc].contiguous().view(torch.float16)
        aux = aux.reshape(heads, groups, g * layout.key_aux)
        ks[b, :, :groups] = aux[..., :dim]
        kz[b, :, :groups] = aux[..., dim:2 * dim]
        c = n - qlens[b]
        kq.append((c // residual) * residual)
        vq.append(max(0, c - residual))
        assert int((cells[0, :, layout.off_flag] == 1).sum()) == kq[-1]
        assert int((cells[0, :, layout.off_flag + 1] == 1).sum()) == vq[-1]
    packed = KIVIPacked(bits=layout.bits, key_group=g, value_group=layout.value_group, key_codes=kc, key_scale=ks,
                        key_zero=kz, value_codes=vc, value_scale=vs, value_zero=vz,
                        keys_quantized=torch.tensor(kq, dtype=torch.int32, device=DEVICE),
                        values_quantized=torch.tensor(vq, dtype=torch.int32, device=DEVICE))
    return key, value, packed


ATTENTION_CASES = [
    # bits, dtype, head dim, query heads, KV heads, queries per request, sinks and soft cap
    (2, torch.bfloat16, 64, 4, 2, [1, 1, 1], False),
    (4, torch.bfloat16, 128, 8, 2, [1, 1, 1], False),
    (8, torch.bfloat16, 64, 4, 4, [1, 1, 1], False),
    (2, torch.bfloat16, 64, 4, 2, [4, 2, 4], False),  # verification of drafts
    (4, torch.float16, 64, 4, 1, [3, 1, 2], True),
    (4, torch.float32, 64, 4, 2, [1, 1, 1], False),
    (2, torch.float32, 128, 2, 2, [5, 5, 1], True),
]


@pytest.mark.parametrize("case", ATTENTION_CASES, ids=lambda c: "-".join(str(x).replace("torch.", "").replace(" ", "")
                                                                        for x in c))
def test_paged_attention_matches_hf_kernel(case: tuple) -> None:
    from economize.attention.kivi import kivi_decode_attention
    from economize.backends.vllm.kivi.kernels import paged_kivi_attention, quantize_committed

    bits, dtype, dim, q_heads, kv_heads, qlens, extras = case
    group, residual, page = 32, 64, 32
    layout = _layout(bits, group, residual, dim, dtype)
    seq_lens = [437, 129, 70]
    # the launch is planned for more keys than any request holds, as under CUDA graphs (max_model_len)
    keys = 512
    cache, table = _pages(seq_lens, page, kv_heads, layout, dtype, seed=bits + dim)
    for b, s in enumerate(seq_lens):
        k, v = _kv(kv_heads, s, dim, dtype, seed=b + 31 * bits)
        _write_dense(cache, table[b], k, v)
    qsl, seq = _varlen(seq_lens, qlens)
    quantize_committed(cache, layout, residual, table, qsl, seq, torch.zeros(3, dtype=torch.int32, device=DEVICE))

    gen = torch.Generator(device=DEVICE).manual_seed(5)
    tokens, queries = sum(qlens), max(qlens)
    query = torch.randn(tokens, q_heads, dim, generator=gen, device=DEVICE).to(dtype)
    scale = 1 / math.sqrt(dim)
    sinks = torch.randn(q_heads, generator=gen, device=DEVICE) if extras else None
    softcap = 20.0 if extras else None
    sentinel = 7.0
    out = torch.full((tokens + 2, q_heads, dim), sentinel, dtype=dtype, device=DEVICE)  # 2 padding rows
    paged_kivi_attention(query, cache, layout, table, qsl, seq, queries, keys, scale, out, softcap=softcap,
                         sinks=sinks)
    assert torch.all(out[tokens:] == sentinel)

    key, value, packed = _hf_operands(cache, table, layout, seq_lens, qlens, residual, keys, dtype)
    hf_query = torch.zeros(len(seq_lens), q_heads, queries, dim, dtype=dtype, device=DEVICE)
    mask = torch.zeros(len(seq_lens), 1, queries, keys, dtype=torch.bool, device=DEVICE)
    for b, (s, q) in enumerate(zip(seq_lens, qlens, strict=True)):
        start = int(qsl[b])
        hf_query[b, :, :q] = query[start:start + q].transpose(0, 1)
        for i in range(q):
            mask[b, 0, i, :s - q + i + 1] = True
    hf = kivi_decode_attention(hf_query, key, value, mask, scale, packed, softcap=softcap, sinks=sinks)
    for b, q in enumerate(qlens):
        start = int(qsl[b])
        got, want = out[start:start + q], hf[b, :q]
        if dtype == torch.float32:
            assert torch.equal(got, want), f"request {b}: max |diff| {(got.float() - want.float()).abs().max()}"
        else:
            assert _ulps(got, want) <= 1, f"request {b}: {_ulps(got, want)} ulp"

    # and both are attention over KIVI's dequantized cache (a loose check of the shared definition)
    groups = q_heads // kv_heads
    for b, q in enumerate(qlens):
        n = seq_lens[b]
        k = key[b, :, :n].float().repeat_interleave(groups, 0)
        v = value[b, :, :n].float().repeat_interleave(groups, 0)
        s = hf_query[b, :, :q].float() @ k.transpose(1, 2) * scale
        if softcap:
            s = softcap * torch.tanh(s / softcap)
        s = s.masked_fill(~mask[b, :, :q, :n], float("-inf"))
        if sinks is not None:
            s = torch.cat([s, sinks[:, None, None].expand(-1, q, 1)], dim=-1)
        p = s.softmax(-1)[..., :n]
        ref = (p @ v).transpose(0, 1)
        tol = 1e-4 if dtype == torch.float32 else 3e-2
        torch.testing.assert_close(out[int(qsl[b]):int(qsl[b]) + q].float(), ref, atol=tol, rtol=tol)
