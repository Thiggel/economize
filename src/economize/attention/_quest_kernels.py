"""The Triton kernel of :mod:`.quest`. Importing this module imports triton; :mod:`.quest` imports it on the first
launch only."""

from __future__ import annotations

import triton
import triton.language as tl

from ._decode_kernels import _attend_block, _visible


@triton.jit
def _quest_split_kernel(
    Q, K, V, IDX, TAIL, Mask, PartO, PartM, PartL,
    sqb, sqh, sqq, sqd,
    skb, skh, sks, skd,
    svb, svh, svs, svd,
    sib, sih,
    smb, smh, smq, sms,
    keys, sel_len, queries, rows, kv_heads, m_tiles, splits, qk_scale, softcap,
    GROUPS: tl.constexpr,
    HEAD_D: tl.constexpr,
    HEAD_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    SPLIT_LEN: tl.constexpr,
    PAGE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    MASK_PER_HEAD: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    IEEE: tl.constexpr,
    DOT_DTYPE: tl.constexpr,
    KV_FMT: tl.constexpr,
    DV_TILES: tl.constexpr = 1,
    PART_DV: tl.constexpr = 0,
):
    # Virtual key axis of (b, kvh): sel_len = pages * PAGE keys of the selected pages IDX[b, kvh, :] (-1: none),
    # then the dense tail, slots TAIL[b] .. keys. BLOCK_N divides PAGE, so a block lies inside one page.
    pid_dv = tl.program_id(0)  # (row tile, value tile), as in decode._split_kernel
    split = tl.program_id(1)
    dvt = pid_dv % DV_TILES
    pid = pid_dv // DV_TILES
    tile = pid % m_tiles
    kvh = (pid // m_tiles) % kv_heads
    b = pid // (m_tiles * kv_heads)

    r = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    r_ok = r < rows
    g = r // queries
    qi = r % queries
    h = kvh * GROUPS + g
    d = tl.arange(0, BLOCK_D)
    dv = dvt * BLOCK_DV + tl.arange(0, BLOCK_DV)

    q_ptr = Q + b * sqb + h[:, None] * sqh + qi[:, None] * sqq + d[None, :] * sqd
    q = tl.load(q_ptr, mask=r_ok[:, None] & (d[None, :] < HEAD_D), other=0.0)
    q = q.to(DOT_DTYPE)
    tail = tl.load(TAIL + b)
    length = sel_len + keys - tail  # virtual keys of this row

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], tl.float32)

    start = split * SPLIT_LEN
    for n0 in range(0, SPLIT_LEN, BLOCK_N):
        first = start + n0
        if first < length:
            sel = first < sel_len
            page = tl.load(IDX + b * sib + kvh * sih + first // PAGE, mask=sel, other=-1)
            base = tl.where(sel, page * PAGE + first % PAGE, tail + first - sel_len)
            ok = (first >= sel_len) | (page >= 0)
            lane = tl.arange(0, BLOCK_N)
            offs = base + lane
            in_range = (first + lane < length) & (offs < keys) & ok
            if HAS_MASK:
                vis = _visible(Mask, b, h, qi, offs, r_ok, in_range, smb, smh, smq, sms, MASK_PER_HEAD)
            else:
                vis = r_ok[:, None] & in_range[None, :]
            if tl.max(vis.to(tl.int32)) > 0:  # skip unselected slots and blocks no row of the program sees
                acc, m_i, l_i = _attend_block(
                    q, K, V, b, kvh, offs, in_range, vis, d, dv, acc, m_i, l_i, skb, skh, sks, skd, svb, svh,
                    svs, svd, qk_scale, softcap, HEAD_D, HEAD_DV, HAS_SOFTCAP, IEEE, DOT_DTYPE, KV_FMT,
                )

    base_row = (pid * splits + split) * BLOCK_M
    if dvt == 0:  # every value tile computes the same row statistics
        tl.store(PartM + base_row + tl.arange(0, BLOCK_M), m_i, mask=r_ok)
        tl.store(PartL + base_row + tl.arange(0, BLOCK_M), l_i, mask=r_ok)
    o_ptr = (
        PartO + (base_row + tl.arange(0, BLOCK_M))[:, None] * (BLOCK_DV if DV_TILES == 1 else PART_DV) + dv[None, :]
    )
    tl.store(o_ptr, acc, mask=r_ok[:, None] & (l_i[:, None] > 0) & (dv[None, :] < HEAD_DV))
