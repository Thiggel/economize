"""Triton kernels of KIVI on vLLM's paged cache (layout in :mod:`.layout`).

* :func:`clear_flags` marks the cells vLLM's cache write just filled as full precision.
* :func:`quantize_committed` quantizes, before a target pass attends, the keys up to ``(c // R) * R`` and the values
  up to ``c - R`` of every decoding request, where ``c`` is the number of tokens the cache held before the pass
  (economize's HF frontier). Each (request, KV head) walks back from that bound over the cells whose flag is still 0,
  so only the tokens that crossed the residual since the last pass are touched. The arithmetic is
  :func:`economize.techniques.kv_quant.kivi.quantize` on the GPU: fp32 min and max, the scale times the fp32
  reciprocal of the level count (PyTorch's division by a Python number), rounded to fp16 where fp16 holds it,
  correctly rounded division, round half to even, and ``code * scale + zero`` without fused multiply-adds.
* :func:`paged_kivi_attention` is :func:`economize.attention.kivi.kivi_decode_attention` over pages: the same split
  kernel (packed codes are dequantized in registers, every other key and value is read densely) with vLLM's
  varlen query layout, its block table and a causal bound per query, followed by the same merge, under the same
  launch plan (:func:`economize.attention.kivi.kivi_plan`). Over the same keys it gives bitwise the HF kernel's
  result.

Importing this module imports triton.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from ....attention._decode_kernels import _DOT, _accumulate, _online_softmax, _scores
from ....attention.decode import sinks_operand
from ....attention.kivi import kivi_plan
from ....errors import ConfigError
from .layout import CellLayout

try:
    from triton.language.extra.cuda import libdevice as _ld
except ImportError:  # pragma: no cover - older triton
    from triton.language.extra import libdevice as _ld

__all__ = ["cache_views", "clear_flags", "paged_kivi_attention", "quantize_committed"]

_FP16_MAX = 65504.0
#: Tokens per value tile of the quantization kernel.
_VALUE_TILE = 16


def cache_views(kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The typed, fp16 and byte views of one layer's cache ``[pages, heads, page_size, cell elements]``.

    Raises:
        ConfigError: The cell axis is not contiguous.
    """
    if kv_cache.dim() != 4 or kv_cache.stride(-1) != 1:
        raise ConfigError(f"economize's KIVI backend needs a [pages, heads, tokens, cell] cache with a contiguous "
                          f"cell, got shape {tuple(kv_cache.shape)} strides {kv_cache.stride()}")
    return kv_cache, kv_cache.view(torch.float16), kv_cache.view(torch.uint8)


@triton.jit
def _clear_flags_kernel(U8, Slots, sb, sh, sn, kv_heads, PAGE: tl.constexpr, OFF_F: tl.constexpr,
                        BLOCK_H: tl.constexpr):
    slot = tl.load(Slots + tl.program_id(0)).to(tl.int64)
    if slot >= 0:
        h = tl.arange(0, BLOCK_H)
        cell = (slot // PAGE) * sb + (slot % PAGE) * sn + h.to(tl.int64) * sh + OFF_F
        zero = tl.zeros([BLOCK_H], tl.uint8)
        tl.store(U8 + cell, zero, mask=h < kv_heads)
        tl.store(U8 + cell + 1, zero, mask=h < kv_heads)


def clear_flags(kv_cache: torch.Tensor, slot_mapping: torch.Tensor, layout: CellLayout) -> None:
    """Mark the cells of ``slot_mapping`` (negative slots are padding) as full precision."""
    if slot_mapping.numel() == 0:
        return
    _, _, u8 = cache_views(kv_cache)
    heads = u8.shape[1]
    _clear_flags_kernel[(slot_mapping.shape[0],)](
        u8, slot_mapping, u8.stride(0), u8.stride(1), u8.stride(2), heads,
        PAGE=u8.shape[2], OFF_F=layout.off_flag, BLOCK_H=triton.next_power_of_2(heads),
    )


@triton.jit
def _cells(BT, b, sbt, t, ok, sb, sh, sn, kvh, PAGE: tl.constexpr):
    """Byte offsets of the cells of tokens ``t`` of request ``b`` at KV head ``kvh`` (int64)."""
    page = tl.load(BT + b * sbt + t // PAGE, mask=ok, other=0).to(tl.int64)
    return page * sb + kvh * sh + (t % PAGE).to(tl.int64) * sn


@triton.jit
def _fp16_rounded(x):
    return tl.where(tl.abs(x) <= 65504.0, x.to(tl.float16, fp_downcast_rounding="rtne").to(tl.float32), x)


@triton.jit
def _quantize_kernel(
    T, H16, U8, BT, QSL, Seq, Prefill,
    sbt, sb, sh, sn, inv_levels,
    ES: tl.constexpr, D: tl.constexpr, BITS: tl.constexpr, G: tl.constexpr, VG: tl.constexpr,
    NVG: tl.constexpr, KA: tl.constexpr, R: tl.constexpr, PAGE: tl.constexpr, VT: tl.constexpr,
    OFF_V: tl.constexpr, OFF_VS: tl.constexpr, OFF_VZ: tl.constexpr, OFF_KA: tl.constexpr,
    OFF_KC: tl.constexpr, OFF_VC: tl.constexpr, OFF_F: tl.constexpr,
):
    PB: tl.constexpr = 8 // BITS
    LEVELS: tl.constexpr = (1 << BITS) - 1
    b = tl.program_id(0)
    kvh = tl.program_id(1).to(tl.int64)
    qlen = tl.load(QSL + b + 1) - tl.load(QSL + b)
    c = tl.load(Seq + b) - qlen
    prefill = tl.load(Prefill + b)
    if (prefill == 0) & (qlen > 0) & (c > 0):
        d = tl.arange(0, D)
        shift = (d % PB) * BITS
        ka_tok = (d // KA).to(tl.int64)
        ka_byte = 2 * (d % KA)
        # keys: whole groups below (c // R) * R, newest first, until a quantized group
        s0 = (c // R) * R - G
        last = _cells(BT, b, sbt, s0 + G - 1, s0 >= 0, sb, sh, sn, kvh, PAGE)
        flag = tl.load(U8 + last + OFF_F, mask=s0 >= 0, other=1).to(tl.int32)
        while flag == 0:
            t = s0 + tl.arange(0, G)
            cell = _cells(BT, b, sbt, t, t >= 0, sb, sh, sn, kvh, PAGE)
            ptr = T + (cell // ES)[:, None] + d[None, :]
            work = tl.load(ptr).to(tl.float32)
            low = tl.min(work, 0)
            high = tl.max(work, 0)
            scale = _fp16_rounded(_ld.mul_rn(high - low, inv_levels))
            low = _fp16_rounded(low)
            safe = tl.where(scale > 0, scale, 1.0)
            codes = _ld.rint(_ld.div_rn(work - low[None, :], tl.broadcast_to(safe[None, :], [G, D])))
            codes = tl.minimum(tl.maximum(codes, 0.0), LEVELS * 1.0)
            fake = _ld.add_rn(_ld.mul_rn(codes, tl.broadcast_to(scale[None, :], [G, D])),
                              tl.broadcast_to(low[None, :], [G, D]))
            tl.store(ptr, fake.to(T.dtype.element_ty, fp_downcast_rounding="rtne"))
            packed = tl.sum(tl.reshape(codes.to(tl.int32) << shift[None, :], [G, D // PB, PB]), 2)
            tl.store(U8 + cell[:, None] + OFF_KC + tl.arange(0, D // PB)[None, :], packed.to(tl.uint8))
            first = _cells(BT, b, sbt, s0, s0 >= 0, sb, sh, sn, kvh, PAGE)
            aux = first + ka_tok * sn + OFF_KA + ka_byte
            tl.store(H16 + aux // 2, scale.to(tl.float16, fp_downcast_rounding="rtne"))
            aux_z = first + ((D + d) // KA).to(tl.int64) * sn + OFF_KA + 2 * ((D + d) % KA)
            tl.store(H16 + aux_z // 2, low.to(tl.float16, fp_downcast_rounding="rtne"))
            over = tl.max(((tl.abs(scale) > 65504.0) | (tl.abs(low) > 65504.0)).to(tl.int32), 0)
            tl.store(U8 + cell + OFF_F, tl.full([G], 1, tl.uint8) + over.to(tl.uint8))
            s0 -= G
            last = _cells(BT, b, sbt, s0 + G - 1, s0 >= 0, sb, sh, sn, kvh, PAGE)
            flag = tl.load(U8 + last + OFF_F, mask=s0 >= 0, other=1).to(tl.int32)
        # values: tokens below c - R, a tile of VT at a time, newest first, until a tile was partly quantized
        top = c - R
        more = top > 0
        while more:
            t = top - VT + tl.arange(0, VT)
            ok = t >= 0
            cell = _cells(BT, b, sbt, t, ok, sb, sh, sn, kvh, PAGE)
            todo = ok & (tl.load(U8 + cell + OFF_F + 1, mask=ok, other=1) == 0)
            ptr = T + (cell // ES + OFF_V // ES)[:, None] + d[None, :]
            work = tl.reshape(tl.load(ptr, mask=todo[:, None], other=0).to(tl.float32), [VT, NVG, VG])
            low = tl.min(work, 2)
            high = tl.max(work, 2)
            scale = _fp16_rounded(_ld.mul_rn(high - low, inv_levels))
            low = _fp16_rounded(low)
            safe = tl.where(scale > 0, scale, 1.0)
            codes = _ld.rint(_ld.div_rn(work - low[:, :, None], tl.broadcast_to(safe[:, :, None], [VT, NVG, VG])))
            codes = tl.minimum(tl.maximum(codes, 0.0), LEVELS * 1.0)
            fake = _ld.add_rn(_ld.mul_rn(codes, tl.broadcast_to(scale[:, :, None], [VT, NVG, VG])),
                              tl.broadcast_to(low[:, :, None], [VT, NVG, VG]))
            tl.store(ptr, tl.reshape(fake, [VT, D]).to(T.dtype.element_ty, fp_downcast_rounding="rtne"),
                     mask=todo[:, None])
            flat = tl.reshape(codes, [VT, D]).to(tl.int32) << shift[None, :]
            packed = tl.sum(tl.reshape(flat, [VT, D // PB, PB]), 2)
            tl.store(U8 + cell[:, None] + OFF_VC + tl.arange(0, D // PB)[None, :], packed.to(tl.uint8),
                     mask=todo[:, None])
            j = tl.arange(0, NVG)
            aux = cell[:, None] + 2 * j[None, :]
            tl.store(H16 + (aux + OFF_VS) // 2, scale.to(tl.float16, fp_downcast_rounding="rtne"), mask=todo[:, None])
            tl.store(H16 + (aux + OFF_VZ) // 2, low.to(tl.float16, fp_downcast_rounding="rtne"), mask=todo[:, None])
            over = tl.max(((tl.abs(scale) > 65504.0) | (tl.abs(low) > 65504.0)).to(tl.int32), 1)
            tl.store(U8 + cell + OFF_F + 1, (1 + over).to(tl.uint8), mask=todo)
            more = (top - VT > 0) & (tl.sum(todo.to(tl.int32), 0) == VT)
            top -= VT


def quantize_committed(kv_cache: torch.Tensor, layout: CellLayout, residual_length: int, block_table: torch.Tensor,
                       query_start_loc: torch.Tensor, seq_lens: torch.Tensor, prefilling: torch.Tensor) -> None:
    """Quantize the committed tokens that left the residual of every decoding request (see the module docstring).

    Args:
        kv_cache: One layer's cache (typed, ``[pages, heads, page_size, cell elements]``).
        block_table: ``[requests, pages per request]`` int32.
        query_start_loc: ``[requests + 1]`` int32, vLLM's varlen query offsets.
        seq_lens: ``[requests]`` int32, tokens of each request after the pass.
        prefilling: ``[requests]`` int32, nonzero for a request whose prompt is not done.
    """
    typed, h16, u8 = cache_views(kv_cache)
    page = u8.shape[2]
    if page % layout.group_size:
        raise ConfigError(f"KIVI needs pages of whole key groups: {page} tokens per page, groups of "
                          f"{layout.group_size}")
    requests = seq_lens.shape[0]
    if requests == 0:
        return
    inv_levels = float(np.float32(1.0) / np.float32(2**layout.bits - 1))
    _quantize_kernel[(requests, u8.shape[1])](
        typed, h16, u8, block_table, query_start_loc, seq_lens, prefilling,
        block_table.stride(0), u8.stride(0), u8.stride(1), u8.stride(2), inv_levels,
        ES=layout.element_size, D=layout.head_size, BITS=layout.bits, G=layout.group_size, VG=layout.value_group,
        NVG=layout.value_groups, KA=layout.key_aux, R=residual_length, PAGE=page, VT=_VALUE_TILE,
        OFF_V=layout.off_v, OFF_VS=layout.off_vs, OFF_VZ=layout.off_vz, OFF_KA=layout.off_ka, OFF_KC=layout.off_kc,
        OFF_VC=layout.off_vc, OFF_F=layout.off_flag, num_warps=4,
    )


@triton.jit
def _dequant(code, scale, zero):
    return _ld.add_rn(_ld.mul_rn(code.to(tl.float32), scale.to(tl.float32)), zero.to(tl.float32))


@triton.jit
def _paged_split_kernel(
    Q, T, H16, U8, BT, QSL, Seq, PartO, PartM, PartL,
    sqt, sqh, sqd, sbt, sb, sh, sn,
    keys, queries, rows, kv_heads, m_tiles, splits, qk_scale, softcap,
    GROUPS: tl.constexpr, HEAD_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr, SPLIT_LEN: tl.constexpr, HAS_SOFTCAP: tl.constexpr, IEEE: tl.constexpr,
    DOT_DTYPE: tl.constexpr, BITS: tl.constexpr, KEY_GROUP: tl.constexpr, VALUE_GROUP: tl.constexpr,
    KA: tl.constexpr, PAGE: tl.constexpr, ES: tl.constexpr, OFF_V: tl.constexpr, OFF_VS: tl.constexpr,
    OFF_VZ: tl.constexpr, OFF_KA: tl.constexpr, OFF_KC: tl.constexpr, OFF_VC: tl.constexpr, OFF_F: tl.constexpr,
):
    PER_BYTE: tl.constexpr = 8 // BITS
    LEVELS: tl.constexpr = (1 << BITS) - 1
    pid = tl.program_id(0)
    split = tl.program_id(1)
    tile = pid % m_tiles
    kvh = (pid // m_tiles) % kv_heads
    b = pid // (m_tiles * kv_heads)

    r = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    r_ok = r < rows
    g = r // queries
    qi = r % queries
    h = kvh * GROUPS + g
    d = tl.arange(0, BLOCK_D)
    d_ok = d < HEAD_D
    shift = (d % PER_BYTE) * BITS
    q_start = tl.load(QSL + b)
    qlen = tl.load(QSL + b + 1) - q_start
    seq = tl.load(Seq + b)
    ctx = seq - qlen
    row_ok = r_ok & (qi < qlen)

    q_ptr = Q + (q_start + qi)[:, None].to(tl.int64) * sqt + h[:, None] * sqh + d[None, :] * sqd
    q = tl.load(q_ptr, mask=row_ok[:, None] & d_ok[None, :], other=0.0)
    q = q.to(DOT_DTYPE)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    start = split * SPLIT_LEN
    for n0 in range(0, SPLIT_LEN, BLOCK_N):
        first = start + n0
        offs = first + tl.arange(0, BLOCK_N)
        live = (offs < keys) & (offs < seq)
        vis = row_ok[:, None] & live[None, :] & (offs[None, :] <= ctx + qi[:, None])
        if tl.max(vis.to(tl.int32)) > 0:
            cell = _cells(BT, b, sbt, offs, live, sb, sh, sn, kvh, PAGE)
            # keys: packed codes where the flag is 1, dense entries elsewhere
            k_packed = live & (tl.load(U8 + cell + OFF_F, mask=live, other=0) == 1)
            k = tl.load(
                T + (cell // ES)[:, None] + d[None, :], mask=(live & ~k_packed)[:, None] & d_ok[None, :], other=0,
            ).to(DOT_DTYPE)
            if tl.max(k_packed.to(tl.int32)) > 0:
                packed = k_packed[:, None] & d_ok[None, :]
                byte = tl.load(U8 + cell[:, None] + OFF_KC + (d // PER_BYTE)[None, :], mask=packed, other=0)
                code = (byte.to(tl.int32) >> shift[None, :]) & LEVELS
                # the group's scale of channel d is aux entry d, its zero point entry HEAD_D + d (layout.py)
                lead = cell[:, None] - (offs % KEY_GROUP).to(tl.int64)[:, None] * sn + OFF_KA
                s_byte = lead + (d // KA).to(tl.int64)[None, :] * sn + 2 * (d % KA)[None, :]
                z_byte = lead + ((HEAD_D + d) // KA).to(tl.int64)[None, :] * sn + 2 * ((HEAD_D + d) % KA)[None, :]
                k_scale = tl.load(H16 + s_byte // 2, mask=packed, other=0)
                kd = _dequant(code, k_scale, tl.load(H16 + z_byte // 2, mask=packed, other=0))
                k = tl.where(packed, kd.to(DOT_DTYPE), k)
            s = _scores(q, k, qk_scale, softcap, HAS_SOFTCAP, IEEE)
            p, alpha, m_new, l_i = _online_softmax(s, vis, m_i, l_i)
            # values: likewise with their own flag
            v_packed = live & (tl.load(U8 + cell + OFF_F + 1, mask=live, other=0) == 1)
            v = tl.load(
                T + (cell // ES + OFF_V // ES)[:, None] + d[None, :], mask=(live & ~v_packed)[:, None] & d_ok[None, :],
                other=0,
            ).to(DOT_DTYPE)
            if tl.max(v_packed.to(tl.int32)) > 0:
                packed_v = v_packed[:, None] & d_ok[None, :]
                byte_v = tl.load(U8 + cell[:, None] + OFF_VC + (d // PER_BYTE)[None, :], mask=packed_v, other=0)
                code_v = (byte_v.to(tl.int32) >> shift[None, :]) & LEVELS
                t_byte = cell[:, None] + 2 * (d // VALUE_GROUP)[None, :]
                vd = _dequant(code_v, tl.load(H16 + (t_byte + OFF_VS) // 2, mask=packed_v, other=0),
                              tl.load(H16 + (t_byte + OFF_VZ) // 2, mask=packed_v, other=0))
                v = tl.where(packed_v, vd.to(DOT_DTYPE), v)
            acc = _accumulate(acc, alpha, p, v, IEEE, DOT_DTYPE)
            m_i = m_new

    base = (pid * splits + split) * BLOCK_M
    tl.store(PartM + base + tl.arange(0, BLOCK_M), m_i, mask=r_ok)
    tl.store(PartL + base + tl.arange(0, BLOCK_M), l_i, mask=r_ok)
    o_ptr = PartO + (base + tl.arange(0, BLOCK_M))[:, None] * BLOCK_D + d[None, :]
    tl.store(o_ptr, acc, mask=r_ok[:, None] & (l_i[:, None] > 0) & d_ok[None, :])


@triton.jit
def _varlen_merge_kernel(
    PartO, PartM, PartL, Out, Sinks, QSL,
    sot, soh, sod,
    queries, rows, kv_heads, m_tiles, splits,
    GROUPS: tl.constexpr, HEAD_DV: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_DV: tl.constexpr,
    HAS_SINKS: tl.constexpr,
):
    # economize.attention.decode._merge_kernel with vLLM's varlen output rows
    pid = tl.program_id(0)
    row = tl.program_id(1)
    tile = pid % m_tiles
    kvh = (pid // m_tiles) % kv_heads
    b = pid // (m_tiles * kv_heads)
    r = tile * BLOCK_M + row
    q_start = tl.load(QSL + b)
    qlen = tl.load(QSL + b + 1) - q_start
    g = r // queries
    qi = r % queries
    if (r < rows) & (qi < qlen):
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
        h = kvh * GROUPS + g
        if HAS_SINKS:
            sink = tl.load(Sinks + h).to(tl.float32) * 1.4426950408889634
            m_new = tl.maximum(m_i, sink)
            a = tl.math.exp2(m_i - m_new)
            acc = acc * a
            l_i = l_i * a + tl.math.exp2(sink - m_new)
        out = tl.where(l_i > 0, acc / tl.maximum(l_i, 1e-30), 0.0)
        o_ptr = Out + (q_start + qi).to(tl.int64) * sot + h * soh + dv * sod
        tl.store(o_ptr, out.to(Out.dtype.element_ty), mask=dv < HEAD_DV)


def paged_kivi_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    layout: CellLayout,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    keys: int,
    scale: float,
    out: torch.Tensor,
    softcap: float | None = None,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal attention of vLLM's varlen ``query`` ``[tokens, Hq, D]`` over KIVI's paged cache into ``out``
    ``[tokens, Hq, D]``.

    Request ``b``'s queries are rows ``query_start_loc[b]:query_start_loc[b + 1]``; its cache holds
    ``seq_lens[b]`` tokens, the last of them the queries'. ``keys`` bounds the key extent the launch is planned for
    (fixed under CUDA graphs) and must cover every ``seq_lens[b]``; ``max_query_len`` bounds the queries of a
    request. Rows of ``out`` beyond ``query_start_loc[-1]`` are left as they are.
    """
    typed, h16, u8 = cache_views(kv_cache)
    tokens, q_heads, head_dim = query.shape
    kv_heads, page = u8.shape[1], u8.shape[2]
    if head_dim != layout.head_size or q_heads % kv_heads:
        raise ConfigError(f"query [{tokens}, {q_heads}, {head_dim}] does not fit a cache of {kv_heads} KV heads of "
                          f"{layout.head_size} channels")
    if query.dtype not in _DOT or typed.dtype != query.dtype:
        raise ConfigError(f"paged KIVI attention takes a cache in the query's dtype, got {typed.dtype} / "
                          f"{query.dtype}")
    batch = seq_lens.shape[0]
    groups = q_heads // kv_heads
    queries = max(1, int(max_query_len))
    rows = groups * queries
    p = kivi_plan(batch, kv_heads, rows, keys, head_dim, query.device, query.element_size())
    block_d = max(16, triton.next_power_of_2(head_dim))
    programs = batch * kv_heads * p.m_tiles
    part_o = torch.empty(programs * p.splits * p.block_m * block_d, device=query.device, dtype=torch.float32)
    part_m = torch.empty(programs * p.splits * p.block_m, device=query.device, dtype=torch.float32)
    part_l = torch.empty_like(part_m)
    sink_logits = sinks_operand(sinks, q_heads)
    num_warps = 4 if block_d <= 256 else 8
    with torch.cuda.device(query.device):
        _paged_split_kernel[(programs, p.splits)](
            query, typed, h16, u8, block_table, query_start_loc, seq_lens, part_o, part_m, part_l,
            *query.stride(), block_table.stride(0), u8.stride(0), u8.stride(1), u8.stride(2),
            keys, queries, rows, kv_heads, p.m_tiles, p.splits, float(scale), float(softcap or 1.0),
            GROUPS=groups, HEAD_D=head_dim, BLOCK_M=p.block_m, BLOCK_N=p.block_n, BLOCK_D=block_d,
            SPLIT_LEN=p.split_len, HAS_SOFTCAP=softcap is not None, IEEE=query.dtype == torch.float32,
            DOT_DTYPE=_DOT[query.dtype], BITS=layout.bits, KEY_GROUP=layout.group_size,
            VALUE_GROUP=layout.value_group, KA=layout.key_aux, PAGE=page, ES=layout.element_size,
            OFF_V=layout.off_v, OFF_VS=layout.off_vs, OFF_VZ=layout.off_vz, OFF_KA=layout.off_ka,
            OFF_KC=layout.off_kc, OFF_VC=layout.off_vc, OFF_F=layout.off_flag, num_warps=num_warps, num_stages=1,
        )
        _varlen_merge_kernel[(programs, p.block_m)](
            part_o, part_m, part_l, out, part_m if sink_logits is None else sink_logits, query_start_loc,
            *out.stride(), queries, rows, kv_heads, p.m_tiles, p.splits,
            GROUPS=groups, HEAD_DV=head_dim, BLOCK_M=p.block_m, BLOCK_DV=block_d, HAS_SINKS=sink_logits is not None,
            num_warps=4,
        )
    return out
