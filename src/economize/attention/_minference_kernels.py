"""The Triton kernels of :mod:`.minference`. Importing this module imports triton; :mod:`.minference` imports it on
the first launch only."""

from __future__ import annotations

import triton
import triton.language as tl

from ._decode_kernels import _attend_block


@triton.jit
def _vs_convert_kernel(
    V, S, RS, RE, RC, CI, CC,
    nv, ns, blocks, nr, nc,
    BLOCK: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # One program per (b, h, block row m), as one CUDA thread per row in MInference's convert_vertical_slash_indexes.
    # V[bh, :nv]: vertical indices ascending (padding >= every range end); S[bh, :ns]: slash diagonals descending
    # (padding >= every block end, so it is skipped).
    pid = tl.program_id(0)
    m = pid % blocks
    bh = pid // blocks
    end_m = (m + 1) * BLOCK
    zero = tl.program_id(0) * 0  # an int32 scalar tensor, so that the loop-carried state keeps its type
    cnt = zero
    cs = zero
    ce = zero
    started = zero
    for k in range(ns):
        d = tl.load(S + bh * ns + k)
        valid = d < end_m
        e = tl.maximum(end_m - d, BLOCK)
        is_new = valid & (started == 0)
        save = valid & (started != 0) & (e > ce + BLOCK)
        extend = valid & (started != 0) & (e <= ce + BLOCK) & (e > ce)
        tl.store(RS + pid * nr + cnt, cs, mask=save)
        tl.store(RE + pid * nr + cnt, ce, mask=save)
        cnt = cnt + save.to(tl.int32)
        cs = tl.where(is_new | save, e - BLOCK, cs)
        ce = tl.where(is_new | save, e, tl.where(extend, ce + BLOCK, ce))
        started = tl.where(valid, 1, started)
    tl.store(RS + pid * nr + cnt, cs, mask=started != 0)
    tl.store(RE + pid * nr + cnt, ce, mask=started != 0)
    cnt = cnt + started
    tl.store(RC + pid, cnt)
    tl.debug_barrier()  # the column pass reads the ranges back

    out = zero
    for v0 in range(0, nv, BLOCK_V):
        lane = v0 + tl.arange(0, BLOCK_V)
        ok = lane < nv
        v = tl.load(V + bh * nv + lane, mask=ok, other=0)
        covered = lane < 0
        for r in range(cnt):
            lo = tl.load(RS + pid * nr + r)
            hi = tl.load(RE + pid * nr + r)
            covered = covered | ((v >= lo) & (v < hi))
        keep = ok & (covered.to(tl.int32) == 0) & (v < ce)
        keep_i = keep.to(tl.int32)
        pos = out + tl.cumsum(keep_i, 0) - 1
        tl.store(CI + pid * nc + pos, v, mask=keep)
        out = out + tl.sum(keep_i, 0)
    tl.store(CC + pid, out)


@triton.jit
def _sparse_prefill_kernel(
    Q, K, V, Out, RS, RE, RC, CI, CC, WIN, SNK,
    sqb, sqh, sqq, sqd,
    skb, skh, sks, skd,
    svb, svh, svs, svd,
    sob, soq, soh, sod,
    queries, heads, blocks, nr, nc, qk_scale, softcap,
    GROUPS: tl.constexpr,
    HEAD_D: tl.constexpr,
    HEAD_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    INDEX_BLOCK: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    IEEE: tl.constexpr,
    DOT_DTYPE: tl.constexpr,
    KV_FMT: tl.constexpr,
):
    # One program per (row tile, b * Hq + h). BLOCK_M divides INDEX_BLOCK, so the tile lies in one block row of the
    # index: it reads that row's key ranges, then its gathered columns, under the causal and window masks.
    tile = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // heads
    h = bh % heads
    kvh = h // GROUPS
    first_row = tile * BLOCK_M
    r = first_row + tl.arange(0, BLOCK_M)
    r_ok = r < queries
    row = bh * blocks + first_row // INDEX_BLOCK
    d = tl.arange(0, BLOCK_D)
    dv = tl.arange(0, BLOCK_DV)

    q = tl.load(
        Q + b * sqb + h * sqh + r[:, None] * sqq + d[None, :] * sqd,
        mask=r_ok[:, None] & (d[None, :] < HEAD_D),
        other=0.0,
    )
    q = q.to(DOT_DTYPE)
    win = tl.load(WIN + h)
    snk = tl.load(SNK + h)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], tl.float32)
    last = tl.minimum(first_row + BLOCK_M, queries)  # no key of the tile's causal past lies at or beyond it

    n_ranges = tl.load(RC + row)
    for i in range(n_ranges):
        lo = tl.load(RS + row * nr + i)
        hi = tl.minimum(tl.load(RE + row * nr + i), last)
        for n0 in range(lo, hi, BLOCK_N):
            offs = n0 + tl.arange(0, BLOCK_N)
            in_range = offs < hi
            vis = r_ok[:, None] & in_range[None, :] & (offs[None, :] <= r[:, None])
            vis = vis & ((win == 0) | (r[:, None] - offs[None, :] < win) | (offs[None, :] < snk))
            if tl.max(vis.to(tl.int32)) > 0:
                acc, m_i, l_i = _attend_block(
                    q, K, V, b, kvh, offs, in_range, vis, d, dv, acc, m_i, l_i, skb, skh, sks, skd, svb, svh,
                    svs, svd, qk_scale, softcap, HEAD_D, HEAD_DV, HAS_SOFTCAP, IEEE, DOT_DTYPE, KV_FMT,
                )

    n_columns = tl.load(CC + row)
    for c0 in range(0, n_columns, BLOCK_N):
        lane = c0 + tl.arange(0, BLOCK_N)
        ok = lane < n_columns
        offs = tl.load(CI + row * nc + lane, mask=ok, other=0)
        in_range = ok & (offs < last)
        vis = r_ok[:, None] & in_range[None, :] & (offs[None, :] <= r[:, None])
        vis = vis & ((win == 0) | (r[:, None] - offs[None, :] < win) | (offs[None, :] < snk))
        if tl.max(vis.to(tl.int32)) > 0:
            acc, m_i, l_i = _attend_block(
                q, K, V, b, kvh, offs, in_range, vis, d, dv, acc, m_i, l_i, skb, skh, sks, skd, svb, svh,
                svs, svd, qk_scale, softcap, HEAD_D, HEAD_DV, HAS_SOFTCAP, IEEE, DOT_DTYPE, KV_FMT,
            )

    o = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
    tl.store(
        Out + b * sob + r[:, None] * soq + h * soh + dv[None, :] * sod,
        o.to(Out.dtype.element_ty),
        mask=r_ok[:, None] & (dv[None, :] < HEAD_DV),
    )
