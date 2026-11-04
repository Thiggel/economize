"""The Triton kernels of :mod:`.decode`. Importing this module imports triton; :mod:`.decode` imports it on the first
launch only, so that ``import economize`` does not need triton."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra.cuda import libdevice as _ld
except ImportError:  # older triton
    from triton.language.extra import libdevice as _ld

#: The ``tl`` dtype the kernels compute ``tl.dot`` in, per query dtype.
_DOT = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


@triton.jit
def _from_fp8(x, KV_FMT: tl.constexpr):
    """Raw fp8 bytes (``uint8``) to their exact float value. e5m2 is the upper byte of an fp16; e4m3fn becomes
    an fp16 with the same sign and mantissa whose exponent is 8 too small (biases 7 and 15), which a factor 256
    corrects for normal and subnormal values alike."""
    b = x.to(tl.int32)
    if KV_FMT == 2:
        bits = b << 8
        factor = 1.0
    else:
        bits = ((b & 0x80) << 8) | ((b & 0x7F) << 7)
        factor = 256.0
    return bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32) * factor


@triton.jit
def _scores(q, k, qk_scale, softcap, HAS_SOFTCAP: tl.constexpr, IEEE: tl.constexpr):
    """The ``[BLOCK_M, BLOCK_N]`` scores of ``q`` against the keys ``k`` in log2 units, soft-capped. Every kernel
    computes them here, each step rounded on its own, so that the decode kernels round them alike."""
    s = tl.dot(q, tl.trans(k), input_precision="ieee") if IEEE else tl.dot(q, tl.trans(k))
    s = _ld.mul_rn(s, qk_scale)
    if HAS_SOFTCAP:
        # softcap * tanh(s / softcap), with tanh from exp (s is in natural-log units here)
        t = _ld.div_rn(s, softcap)
        e = tl.exp(-2.0 * tl.abs(t))
        th = _ld.div_rn(1.0 - e, 1.0 + e)
        s = _ld.mul_rn(softcap, tl.where(t < 0, -th, th))
    return _ld.mul_rn(s, 1.4426950408889634)  # to log2 units for exp2


@triton.jit
def _online_softmax(s, vis, m_i, l_i):
    """One block of the online softmax over the scores ``s`` (hidden where ``vis`` is false): the block's weights
    ``p``, the factor ``alpha`` that rescales the earlier blocks, and the new running maximum and normalizer."""
    s = tl.where(vis, s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, 1))
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    p = tl.math.exp2(_ld.sub_rn(s, m_safe[:, None]))
    alpha = tl.math.exp2(_ld.sub_rn(m_i, m_safe))
    l_i = _ld.add_rn(_ld.mul_rn(l_i, alpha), tl.sum(p, 1))
    return p, alpha, m_new, l_i


@triton.jit
def _accumulate(acc, alpha, p, v, IEEE: tl.constexpr, DOT_DTYPE: tl.constexpr):
    """``acc * alpha + p @ v``, the output update of the online softmax."""
    if IEEE:
        return _ld.mul_rn(acc, alpha[:, None]) + tl.dot(p, v, input_precision="ieee")
    return _ld.mul_rn(acc, alpha[:, None]) + tl.dot(p.to(DOT_DTYPE), v)


@triton.jit
def _visible(Mask, b, h, qi, offs, r_ok, in_range, smb, smh, smq, sms, MASK_PER_HEAD: tl.constexpr):
    """The ``[BLOCK_M, BLOCK_N]`` block of the boolean mask (rows past ``rows`` and keys past ``keys`` hidden)."""
    if MASK_PER_HEAD:
        m_ptr = Mask + b * smb + h[:, None] * smh + qi[:, None] * smq + offs[None, :] * sms
    else:
        m_ptr = Mask + b * smb + qi[:, None] * smq + offs[None, :] * sms
    return tl.load(m_ptr, mask=r_ok[:, None] & in_range[None, :], other=0) != 0


@triton.jit
def _attend_block(
    q, K, V, b, kvh, offs, in_range, vis, d, dv, acc, m_i, l_i,
    skb, skh, sks, skd, svb, svh, svs, svd, qk_scale, softcap,
    HEAD_D: tl.constexpr,
    HEAD_DV: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    IEEE: tl.constexpr,
    DOT_DTYPE: tl.constexpr,
    KV_FMT: tl.constexpr,
):
    """One block of keys of the online softmax: returns the updated ``(acc, m_i, l_i)``."""
    k = tl.load(
        K + b * skb + kvh * skh + offs[:, None] * sks + d[None, :] * skd,
        mask=in_range[:, None] & (d[None, :] < HEAD_D),
        other=0,
    )
    if KV_FMT != 0:
        k = _from_fp8(k, KV_FMT)
    s = _scores(q, k.to(DOT_DTYPE), qk_scale, softcap, HAS_SOFTCAP, IEEE)
    p, alpha, m_new, l_i = _online_softmax(s, vis, m_i, l_i)
    v = tl.load(
        V + b * svb + kvh * svh + offs[:, None] * svs + dv[None, :] * svd,
        mask=in_range[:, None] & (dv[None, :] < HEAD_DV),
        other=0,
    )
    if KV_FMT != 0:
        v = _from_fp8(v, KV_FMT)
    acc = _accumulate(acc, alpha, p, v.to(DOT_DTYPE), IEEE, DOT_DTYPE)
    return acc, m_new, l_i


@triton.jit
def _split_kernel(
    Q, K, V, Mask, PartO, PartM, PartL,
    sqb, sqh, sqq, sqd,
    skb, skh, sks, skd,
    svb, svh, svs, svd,
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
    KV_FMT: tl.constexpr,
    DV_TILES: tl.constexpr = 1,
    PART_DV: tl.constexpr = 0,
):
    # program = (row tile, value tile): with DV_TILES > 1 every program recomputes the scores of its rows and
    # accumulates one BLOCK_DV-wide slice of the value dimension (PART_DV = the partial output's row stride)
    pid_dv = tl.program_id(0)
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

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], tl.float32)

    start = split * SPLIT_LEN
    for n0 in range(0, SPLIT_LEN, BLOCK_N):
        offs = start + n0 + tl.arange(0, BLOCK_N)
        in_range = offs < keys
        if HAS_MASK:
            vis = _visible(Mask, b, h, qi, offs, r_ok, in_range, smb, smh, smq, sms, MASK_PER_HEAD)
        else:
            vis = r_ok[:, None] & in_range[None, :]
        if tl.max(vis.to(tl.int32)) > 0:  # skip blocks no row of the program sees
            acc, m_i, l_i = _attend_block(
                q, K, V, b, kvh, offs, in_range, vis, d, dv, acc, m_i, l_i, skb, skh, sks, skd, svb, svh,
                svs, svd, qk_scale, softcap, HEAD_D, HEAD_DV, HAS_SOFTCAP, IEEE, DOT_DTYPE, KV_FMT,
            )

    base = (pid * splits + split) * BLOCK_M
    if dvt == 0:  # every value tile computes the same row statistics
        tl.store(PartM + base + tl.arange(0, BLOCK_M), m_i, mask=r_ok)
        tl.store(PartL + base + tl.arange(0, BLOCK_M), l_i, mask=r_ok)
    o_ptr = PartO + (base + tl.arange(0, BLOCK_M))[:, None] * (BLOCK_DV if DV_TILES == 1 else PART_DV) + dv[None, :]
    tl.store(o_ptr, acc, mask=r_ok[:, None] & (l_i[:, None] > 0) & (dv[None, :] < HEAD_DV))


@triton.jit
def _merge_kernel(
    PartO, PartM, PartL, Out, Sinks,
    sob, soq, soh, sod,
    queries, rows, kv_heads, m_tiles, splits,
    GROUPS: tl.constexpr,
    HEAD_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    HAS_SINKS: tl.constexpr,
):
    pid = tl.program_id(0)
    row = tl.program_id(1)  # row inside the tile
    tile = pid % m_tiles
    kvh = (pid // m_tiles) % kv_heads
    b = pid // (m_tiles * kv_heads)
    r = tile * BLOCK_M + row
    if r < rows:
        dv = tl.arange(0, BLOCK_DV)
        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros([BLOCK_DV], tl.float32)
        for s in range(splits):
            base = (pid * splits + s) * BLOCK_M + row
            m_s = tl.load(PartM + base)
            l_s = tl.load(PartL + base)
            if l_s > 0:
                o_s = tl.load(PartO + base * BLOCK_DV + dv, mask=dv < HEAD_DV, other=0.0)
                m_new = tl.maximum(m_i, m_s)
                a = tl.math.exp2(m_i - m_new)
                c = tl.math.exp2(m_s - m_new)
                acc = acc * a + o_s * c
                l_i = l_i * a + l_s * c
                m_i = m_new
        g = r // queries
        qi = r % queries
        h = kvh * GROUPS + g
        if HAS_SINKS:  # the sink's logit joins the denominator (log2 units, like m_i)
            sink = tl.load(Sinks + h).to(tl.float32) * 1.4426950408889634
            m_new = tl.maximum(m_i, sink)
            a = tl.math.exp2(m_i - m_new)
            acc = acc * a
            l_i = l_i * a + tl.math.exp2(sink - m_new)
        out = tl.where(l_i > 0, acc / tl.maximum(l_i, 1e-30), 0.0)
        tl.store(Out + b * sob + qi * soq + h * soh + dv * sod, out.to(Out.dtype.element_ty), mask=dv < HEAD_DV)
