"""The Triton kernel of :mod:`.gathered`. Importing this module imports triton; :mod:`.gathered` imports it on the
first launch only."""

from __future__ import annotations

import triton
import triton.language as tl

from ._decode_kernels import _accumulate, _online_softmax, _scores


@triton.jit
def _gathered_split_kernel(
    Q, K, V, CK, CV, IDX, CNT, START, Mask, PartO, PartM, PartL,
    sqb, sqh, sqq, sqd,
    skb, skh, sks, skd,
    svb, svh, svs, svd,
    ckb, ckh, cks,
    cvb, cvh, cvs,
    sib, sih,
    smb, smh, smq, sms,
    keys, queries, rows, kv_heads, m_tiles, splits, qk_scale, softcap,
    GROUPS: tl.constexpr,
    HEAD_D: tl.constexpr,
    HEAD_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    SPLIT_LEN: tl.constexpr,
    HAS_MASK: tl.constexpr,
    MASK_PER_HEAD: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    IEEE: tl.constexpr,
    DOT_DTYPE: tl.constexpr,
    DV_TILES: tl.constexpr = 1,
    PART_DV: tl.constexpr = 0,
):
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
    d_ok = d < HEAD_D
    dv_ok = dv < HEAD_DV

    q_ptr = Q + b * sqb + h[:, None] * sqh + qi[:, None] * sqq + d[None, :] * sqd
    q = tl.load(q_ptr, mask=r_ok[:, None] & d_ok[None, :], other=0.0)
    q = q.to(DOT_DTYPE)
    count = tl.load(CNT + b * kv_heads + kvh)
    first_dense = tl.load(START + b)
    length = count + keys - first_dense  # virtual keys of this row

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], tl.float32)

    start = split * SPLIT_LEN
    for n0 in range(0, SPLIT_LEN, BLOCK_N):
        first = start + n0
        if first < length:
            offs = first + tl.arange(0, BLOCK_N)
            compact = offs < count
            in_range = offs < length
            dense = in_range & ~compact
            col = first_dense + offs - count
            if first < count:
                slot = tl.load(IDX + b * sib + kvh * sih + offs, mask=compact, other=0)
                col = tl.where(compact, slot, col)
            if HAS_MASK:
                if MASK_PER_HEAD:
                    m_ptr = Mask + b * smb + h[:, None] * smh + qi[:, None] * smq + col[None, :] * sms
                else:
                    m_ptr = Mask + b * smb + qi[:, None] * smq + col[None, :] * sms
                vis = tl.load(m_ptr, mask=r_ok[:, None] & in_range[None, :], other=0) != 0
            else:
                vis = r_ok[:, None] & in_range[None, :]
            if tl.max(vis.to(tl.int32)) > 0:
                k = tl.load(
                    K + b * skb + kvh * skh + col[:, None] * sks + d[None, :] * skd,
                    mask=dense[:, None] & d_ok[None, :],
                    other=0,
                ).to(DOT_DTYPE)
                if first < count:
                    kc = tl.load(
                        CK + b * ckb + kvh * ckh + offs[:, None] * cks + d[None, :],
                        mask=compact[:, None] & d_ok[None, :],
                        other=0,
                    ).to(DOT_DTYPE)
                    k = tl.where(compact[:, None], kc, k)
                s = _scores(q, k, qk_scale, softcap, HAS_SOFTCAP, IEEE)
                p, alpha, m_new, l_i = _online_softmax(s, vis, m_i, l_i)
                v = tl.load(
                    V + b * svb + kvh * svh + col[:, None] * svs + dv[None, :] * svd,
                    mask=dense[:, None] & dv_ok[None, :],
                    other=0,
                ).to(DOT_DTYPE)
                if first < count:
                    vc = tl.load(
                        CV + b * cvb + kvh * cvh + offs[:, None] * cvs + dv[None, :],
                        mask=compact[:, None] & dv_ok[None, :],
                        other=0,
                    ).to(DOT_DTYPE)
                    v = tl.where(compact[:, None], vc, v)
                acc = _accumulate(acc, alpha, p, v, IEEE, DOT_DTYPE)
                m_i = m_new

    base = (pid * splits + split) * BLOCK_M
    if dvt == 0:
        tl.store(PartM + base + tl.arange(0, BLOCK_M), m_i, mask=r_ok)
        tl.store(PartL + base + tl.arange(0, BLOCK_M), l_i, mask=r_ok)
    o_ptr = PartO + (base + tl.arange(0, BLOCK_M))[:, None] * (BLOCK_DV if DV_TILES == 1 else PART_DV) + dv[None, :]
    tl.store(o_ptr, acc, mask=r_ok[:, None] & (l_i[:, None] > 0) & dv_ok[None, :])
