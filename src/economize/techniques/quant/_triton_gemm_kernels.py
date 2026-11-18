"""The Triton kernel of :mod:`.triton_gemm`. Importing this module imports triton; :mod:`.triton_gemm` imports it on
the first launch only, so that ``import economize`` does not need triton."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _weight_only_gemm(
    x_ptr, w_ptr, s_ptr, z_ptr, o_ptr, M, N, K, G,
    stride_xm, stride_wn, stride_sn, stride_zn, stride_om,
    FP8: tl.constexpr, HAS_ZERO: tl.constexpr, PER_CHANNEL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):  # fmt: skip
    """``o[M, N] = x[M, K] @ dequant(w)[N, K]^T`` for 8-bit codes, decoded in registers.

    Integer codes (``uint8``) become ``code - zero`` (exact in 16-bit floats); fp8 e4m3 codes (passed as ``uint8``)
    become the fp16 number with the same bits shifted into place, which is the e4m3 value times 2^-8 (the scales
    carry the factor 256). Each ``BLOCK_K`` slice lies inside one group (``G % BLOCK_K == 0``), so its partial product
    is scaled once; per-channel scales are applied after the loop. Products accumulate in fp32.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        x = tl.load(x_ptr + rm[:, None] * stride_xm + kk[None, :], mask=(rm[:, None] < M) & (kk[None, :] < K),
                    other=0.0)  # fmt: skip
        codes = tl.load(w_ptr + rn[:, None] * stride_wn + kk[None, :], mask=(rn[:, None] < N) & (kk[None, :] < K),
                        other=0)  # fmt: skip
        if FP8:
            bits = codes.to(tl.int16)
            half = ((bits & 0x7F) << 7) | ((bits & 0x80) << 8)
            w = half.to(tl.float16, bitcast=True).to(x.dtype)
        else:
            w = codes.to(tl.float32)
            if HAS_ZERO:
                zero = tl.load(z_ptr + rn * stride_zn + k0 // G, mask=rn < N, other=0).to(tl.float32)
                w = w - zero[:, None]
            w = w.to(x.dtype)
        part = tl.dot(x, tl.trans(w))
        if PER_CHANNEL:
            acc += part
        else:
            scale = tl.load(s_ptr + rn * stride_sn + k0 // G, mask=rn < N, other=0.0)
            acc += part * scale[None, :]
    if PER_CHANNEL:
        scale = tl.load(s_ptr + rn * stride_sn, mask=rn < N, other=0.0)
        acc = acc * scale[None, :]
    tl.store(o_ptr + rm[:, None] * stride_om + rn[None, :], acc.to(o_ptr.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))  # fmt: skip
