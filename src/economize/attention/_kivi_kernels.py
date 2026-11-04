"""The Triton kernels of :mod:`.kivi`. Importing this module imports triton; :mod:`.kivi` imports it on the first
launch only."""

from __future__ import annotations

import triton
import triton.language as tl

from ._decode_kernels import _accumulate, _online_softmax, _scores

try:
    from triton.language.extra.cuda import libdevice as _ld
except ImportError:  # pragma: no cover - older triton
    from triton.language.extra import libdevice as _ld


@triton.jit
def _dequant(code, scale, zero):
    return _ld.add_rn(_ld.mul_rn(code.to(tl.float32), scale.to(tl.float32)), zero.to(tl.float32))


@triton.jit
def _unpack(byte, BITS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    """Codes ``[BLOCK_N, BLOCK_D]`` (int32) of packed bytes ``[BLOCK_N, BLOCK_D * BITS / 8]``: every byte is read
    once and its codes are interleaved in channel order (:func:`pack_codes`)."""
    b = byte.to(tl.int32)
    if BITS == 8:
        codes = b
    elif BITS == 4:
        codes = tl.reshape(tl.join(b & 15, b >> 4), [BLOCK_N, BLOCK_D])
    else:  # 2 bits: join(join(c0, c2), join(c1, c3)) lays the codes out as c0, c1, c2, c3
        codes = tl.reshape(
            tl.join(tl.join(b & 3, (b >> 4) & 3), tl.join((b >> 2) & 3, b >> 6)), [BLOCK_N, BLOCK_D]
        )
    return codes


@triton.jit
def _key_codes(
    KC, KS, KZ, b, kvh, first, offs, d, d_ok, packed, keys, kcb, kch, kcs, ksb, ksh, kss,
    HEAD_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BITS: tl.constexpr,
    KEY_GROUP: tl.constexpr,
    KEY_ROWS: tl.constexpr,
):
    """Dequantized keys ``[BLOCK_N, BLOCK_D]`` (fp32) of the block at ``first``.

    With ``KEY_ROWS > 0`` the block holds whole key groups (or lies inside one): the packed bytes are read once
    each and every group's scale and zero point once, then broadcast over its keys. ``KEY_ROWS == 0`` reads one
    byte, scale and zero point per element (any group size)."""
    PER_BYTE: tl.constexpr = 8 // BITS
    if KEY_ROWS > 0:
        col = tl.arange(0, BLOCK_D // PER_BYTE)
        byte = tl.load(
            KC + b * kcb + kvh * kch + offs[:, None] * kcs + col[None, :],
            mask=packed[:, None] & (col < HEAD_D // PER_BYTE)[None, :],
            other=0,
        )
        code = _unpack(byte, BITS, BLOCK_N, BLOCK_D)
        REP: tl.constexpr = BLOCK_N // KEY_ROWS
        g_first = first + tl.arange(0, KEY_ROWS) * REP  # the first key of every group the block holds
        s_off = b * ksb + kvh * ksh + (g_first // KEY_GROUP)[:, None] * kss + d[None, :]
        s_ok = (g_first < keys)[:, None] & d_ok[None, :]
        k_scale = tl.reshape(
            tl.broadcast_to(tl.reshape(tl.load(KS + s_off, mask=s_ok, other=0), [KEY_ROWS, 1, BLOCK_D]),
                            [KEY_ROWS, REP, BLOCK_D]),
            [BLOCK_N, BLOCK_D],
        )
        k_zero = tl.reshape(
            tl.broadcast_to(tl.reshape(tl.load(KZ + s_off, mask=s_ok, other=0), [KEY_ROWS, 1, BLOCK_D]),
                            [KEY_ROWS, REP, BLOCK_D]),
            [BLOCK_N, BLOCK_D],
        )
    else:
        full = packed[:, None] & d_ok[None, :]
        byte_ptr = KC + b * kcb + kvh * kch + offs[:, None] * kcs + (d // PER_BYTE)[None, :]
        byte = tl.load(byte_ptr, mask=full, other=0)
        code = (byte.to(tl.int32) >> ((d % PER_BYTE) * BITS)[None, :]) & ((1 << BITS) - 1)
        s_off = b * ksb + kvh * ksh + (offs // KEY_GROUP)[:, None] * kss + d[None, :]
        k_scale = tl.load(KS + s_off, mask=full, other=0)
        k_zero = tl.load(KZ + s_off, mask=full, other=0)
    return _dequant(code, k_scale, k_zero)


@triton.jit
def _value_codes(
    VC, VS, VZ, b, kvh, offs, dv, dv_ok, packed, vcb, vch, vcs, vsb, vsh, vss,
    HEAD_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BITS: tl.constexpr,
    VALUE_GROUP: tl.constexpr,
    VALUE_COLS: tl.constexpr,
):
    """Dequantized values ``[BLOCK_N, BLOCK_DV]`` (fp32). With ``VALUE_COLS > 0`` (groups that tile the padded
    value width) the bytes are read once each and every token's ``VALUE_COLS`` scales and zero points once,
    then broadcast over their channels; ``VALUE_COLS == 0`` reads them per element."""
    PER_BYTE: tl.constexpr = 8 // BITS
    if VALUE_COLS > 0:
        col = tl.arange(0, BLOCK_DV // PER_BYTE)
        byte = tl.load(
            VC + b * vcb + kvh * vch + offs[:, None] * vcs + col[None, :],
            mask=packed[:, None] & (col < HEAD_DV // PER_BYTE)[None, :],
            other=0,
        )
        code = _unpack(byte, BITS, BLOCK_N, BLOCK_DV)
        grp = tl.arange(0, VALUE_COLS)
        t_off = b * vsb + kvh * vsh + offs[:, None] * vss + grp[None, :]
        t_ok = packed[:, None] & (grp < HEAD_DV // VALUE_GROUP)[None, :]
        v_scale = tl.reshape(
            tl.broadcast_to(tl.reshape(tl.load(VS + t_off, mask=t_ok, other=0), [BLOCK_N, VALUE_COLS, 1]),
                            [BLOCK_N, VALUE_COLS, VALUE_GROUP]),
            [BLOCK_N, BLOCK_DV],
        )
        v_zero = tl.reshape(
            tl.broadcast_to(tl.reshape(tl.load(VZ + t_off, mask=t_ok, other=0), [BLOCK_N, VALUE_COLS, 1]),
                            [BLOCK_N, VALUE_COLS, VALUE_GROUP]),
            [BLOCK_N, BLOCK_DV],
        )
    else:
        full = packed[:, None] & dv_ok[None, :]
        byte_ptr = VC + b * vcb + kvh * vch + offs[:, None] * vcs + (dv // PER_BYTE)[None, :]
        byte = tl.load(byte_ptr, mask=full, other=0)
        code = (byte.to(tl.int32) >> ((dv % PER_BYTE) * BITS)[None, :]) & ((1 << BITS) - 1)
        t_off = b * vsb + kvh * vsh + offs[:, None] * vss + (dv // VALUE_GROUP)[None, :]
        v_scale = tl.load(VS + t_off, mask=full, other=0)
        v_zero = tl.load(VZ + t_off, mask=full, other=0)
    return _dequant(code, v_scale, v_zero)


# How a block of keys (values) is read: from the dense buffer, from the codes, or from the codes below the
# quantized prefix and the dense buffer above it.
DENSE = tl.constexpr(0)
PACKED = tl.constexpr(1)
MIXED = tl.constexpr(2)


@triton.jit
def _kivi_block(
    q, acc, m_i, l_i, first, h, qi, r_ok, d, dv, d_ok, dv_ok, kq, vq, b, kvh,
    K, V, KC, KS, KZ, VC, VS, VZ, Mask,
    skb, skh, sks, skd,
    svb, svh, svs, svd,
    kcb, kch, kcs,
    ksb, ksh, kss,
    vcb, vch, vcs,
    vsb, vsh, vss,
    smb, smh, smq, sms,
    keys, qk_scale, softcap,
    HEAD_D: tl.constexpr,
    HEAD_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    HAS_MASK: tl.constexpr,
    MASK_PER_HEAD: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    IEEE: tl.constexpr,
    DOT_DTYPE: tl.constexpr,
    BITS: tl.constexpr,
    KEY_GROUP: tl.constexpr,
    VALUE_GROUP: tl.constexpr,
    KEY_ROWS: tl.constexpr,
    VALUE_COLS: tl.constexpr,
    KEY_READ: tl.constexpr,
    VALUE_READ: tl.constexpr,
):
    """The online-softmax update of ``(acc, m_i, l_i)`` by the block of keys at ``first``. ``KEY_READ`` and
    ``VALUE_READ`` say where its keys and values come from (:data:`DENSE`, :data:`PACKED` or :data:`MIXED`); a
    block of values read entirely from one source skips the merge of the two."""
    offs = first + tl.arange(0, BLOCK_N)
    in_range = offs < keys
    if HAS_MASK:
        if MASK_PER_HEAD:
            m_ptr = Mask + b * smb + h[:, None] * smh + qi[:, None] * smq + offs[None, :] * sms
        else:
            m_ptr = Mask + b * smb + qi[:, None] * smq + offs[None, :] * sms
        vis = tl.load(m_ptr, mask=r_ok[:, None] & in_range[None, :], other=0) != 0
    else:
        vis = r_ok[:, None] & in_range[None, :]
    if tl.max(vis.to(tl.int32)) > 0:
        k_ptr = K + b * skb + kvh * skh + offs[:, None] * sks + d[None, :] * skd
        if KEY_READ == DENSE:
            k = tl.load(k_ptr, mask=in_range[:, None] & d_ok[None, :], other=0).to(DOT_DTYPE)
        else:
            # PACKED and MIXED alike: codes below kq, dense slots above (a block of codes only issues no dense
            # load). Building a packed block's keys from the codes alone changes the operand layout of the score
            # dot and rounds the scores one ulp away from the dense kernel's.
            k = tl.zeros([BLOCK_N, BLOCK_D], DOT_DTYPE)
            if first + BLOCK_N > kq:
                k = tl.load(k_ptr, mask=(in_range & (offs >= kq))[:, None] & d_ok[None, :], other=0).to(DOT_DTYPE)
            if first < kq:
                packed = in_range & (offs < kq)
                kd = _key_codes(
                    KC, KS, KZ, b, kvh, first, offs, d, d_ok, packed, keys, kcb, kch, kcs, ksb, ksh, kss,
                    HEAD_D, BLOCK_N, BLOCK_D, BITS, KEY_GROUP, KEY_ROWS,
                )
                k = tl.where(packed[:, None] & d_ok[None, :], kd.to(DOT_DTYPE), k)
        s = _scores(q, k, qk_scale, softcap, HAS_SOFTCAP, IEEE)
        p, alpha, m_new, l_i = _online_softmax(s, vis, m_i, l_i)
        v_ptr = V + b * svb + kvh * svh + offs[:, None] * svs + dv[None, :] * svd
        if VALUE_READ == DENSE:
            v = tl.load(v_ptr, mask=in_range[:, None] & dv_ok[None, :], other=0).to(DOT_DTYPE)
        elif VALUE_READ == PACKED:
            v = _value_codes(
                VC, VS, VZ, b, kvh, offs, dv, dv_ok, in_range, vcb, vch, vcs, vsb, vsh, vss,
                HEAD_DV, BLOCK_N, BLOCK_DV, BITS, VALUE_GROUP, VALUE_COLS,
            ).to(DOT_DTYPE)
        else:
            v = tl.zeros([BLOCK_N, BLOCK_DV], DOT_DTYPE)
            if first + BLOCK_N > vq:
                v = tl.load(v_ptr, mask=(in_range & (offs >= vq))[:, None] & dv_ok[None, :], other=0).to(DOT_DTYPE)
            if first < vq:
                packed_v = in_range & (offs < vq)
                vd = _value_codes(
                    VC, VS, VZ, b, kvh, offs, dv, dv_ok, packed_v, vcb, vch, vcs, vsb, vsh, vss,
                    HEAD_DV, BLOCK_N, BLOCK_DV, BITS, VALUE_GROUP, VALUE_COLS,
                )
                v = tl.where(packed_v[:, None] & dv_ok[None, :], vd.to(DOT_DTYPE), v)
        acc = _accumulate(acc, alpha, p, v, IEEE, DOT_DTYPE)
        m_i = m_new
    return acc, m_i, l_i


@triton.jit
def _kivi_split_kernel(
    Q, K, V, KC, KS, KZ, VC, VS, VZ, KQ, VQ, Mask, PartO, PartM, PartL,
    sqb, sqh, sqq, sqd,
    skb, skh, sks, skd,
    svb, svh, svs, svd,
    kcb, kch, kcs,
    ksb, ksh, kss,
    vcb, vch, vcs,
    vsb, vsh, vss,
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
    BITS: tl.constexpr,
    KEY_GROUP: tl.constexpr,
    VALUE_GROUP: tl.constexpr,
    KEY_ROWS: tl.constexpr,
    VALUE_COLS: tl.constexpr,
):
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
    dv = tl.arange(0, BLOCK_DV)
    d_ok = d < HEAD_D
    dv_ok = dv < HEAD_DV

    q_ptr = Q + b * sqb + h[:, None] * sqh + qi[:, None] * sqq + d[None, :] * sqd
    q = tl.load(q_ptr, mask=r_ok[:, None] & d_ok[None, :], other=0.0)
    q = q.to(DOT_DTYPE)
    kq = tl.load(KQ + b)
    vq = tl.load(VQ + b)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], tl.float32)

    # Blocks below both quantized prefixes read only codes, those above both only the dense buffers, and the few
    # in between merge the two.
    start = split * SPLIT_LEN
    low, high = tl.minimum(kq, vq), tl.maximum(kq, vq)
    for n0 in range(0, SPLIT_LEN, BLOCK_N):
        first = start + n0
        if first + BLOCK_N <= low:
            acc, m_i, l_i = _kivi_block(
                q, acc, m_i, l_i, first, h, qi, r_ok, d, dv, d_ok, dv_ok, kq, vq, b, kvh,
                K, V, KC, KS, KZ, VC, VS, VZ, Mask,
                skb, skh, sks, skd, svb, svh, svs, svd, kcb, kch, kcs, ksb, ksh, kss, vcb, vch, vcs, vsb, vsh, vss,
                smb, smh, smq, sms, keys, qk_scale, softcap,
                HEAD_D, HEAD_DV, BLOCK_M, BLOCK_N, BLOCK_D, BLOCK_DV, HAS_MASK, MASK_PER_HEAD, HAS_SOFTCAP, IEEE,
                DOT_DTYPE, BITS, KEY_GROUP, VALUE_GROUP, KEY_ROWS, VALUE_COLS, PACKED, PACKED,
            )
        elif first >= high:
            acc, m_i, l_i = _kivi_block(
                q, acc, m_i, l_i, first, h, qi, r_ok, d, dv, d_ok, dv_ok, kq, vq, b, kvh,
                K, V, KC, KS, KZ, VC, VS, VZ, Mask,
                skb, skh, sks, skd, svb, svh, svs, svd, kcb, kch, kcs, ksb, ksh, kss, vcb, vch, vcs, vsb, vsh, vss,
                smb, smh, smq, sms, keys, qk_scale, softcap,
                HEAD_D, HEAD_DV, BLOCK_M, BLOCK_N, BLOCK_D, BLOCK_DV, HAS_MASK, MASK_PER_HEAD, HAS_SOFTCAP, IEEE,
                DOT_DTYPE, BITS, KEY_GROUP, VALUE_GROUP, KEY_ROWS, VALUE_COLS, DENSE, DENSE,
            )
        else:
            acc, m_i, l_i = _kivi_block(
                q, acc, m_i, l_i, first, h, qi, r_ok, d, dv, d_ok, dv_ok, kq, vq, b, kvh,
                K, V, KC, KS, KZ, VC, VS, VZ, Mask,
                skb, skh, sks, skd, svb, svh, svs, svd, kcb, kch, kcs, ksb, ksh, kss, vcb, vch, vcs, vsb, vsh, vss,
                smb, smh, smq, sms, keys, qk_scale, softcap,
                HEAD_D, HEAD_DV, BLOCK_M, BLOCK_N, BLOCK_D, BLOCK_DV, HAS_MASK, MASK_PER_HEAD, HAS_SOFTCAP, IEEE,
                DOT_DTYPE, BITS, KEY_GROUP, VALUE_GROUP, KEY_ROWS, VALUE_COLS, MIXED, MIXED,
            )

    base = (pid * splits + split) * BLOCK_M
    tl.store(PartM + base + tl.arange(0, BLOCK_M), m_i, mask=r_ok)
    tl.store(PartL + base + tl.arange(0, BLOCK_M), l_i, mask=r_ok)
    o_ptr = PartO + (base + tl.arange(0, BLOCK_M))[:, None] * BLOCK_DV + dv[None, :]
    tl.store(o_ptr, acc, mask=r_ok[:, None] & (l_i[:, None] > 0) & dv_ok[None, :])
