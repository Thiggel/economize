"""The LittleBit binary GEMM in Triton: ``y = sum_p ((((x * g_p) sign(V_p)^T) * l_p) sign(U_p)^T) * h_p + bias``.

Two launches per map, both reading the packed signs of :mod:`.pack` (32 per ``int32`` word) and expanding them to
``±1`` in registers, so a call reads 1 bit per sign instead of the 16 of a dense weight:

* ``down`` computes ``t_p = ((x * g_p) sign(V_p)^T) * l_p`` for every path at once (``t [P, M, r]``, float32). A
  program owns ``BLOCK_N`` latent dimensions of one path and one block of rows, and a slice of the input channels;
  when the column blocks alone do not fill the GPU the input channels are split over ``SPLIT_K`` programs whose
  partial sums meet in the float32 buffer through atomic adds (the scale ``l_p`` is linear, so each part is scaled
  before it is added).
* ``up`` computes ``y = sum_p (t_p sign(U_p)^T) * h_p + bias`` with the paths summed inside one program.

Rows are multiplied with ``tl.dot`` in blocks of 16 to 64 (``t`` is rounded to the activation dtype first, as in
the reference backend); fewer rows are padded to 16. Nothing synchronizes with the host, so the call is
safe to capture in a CUDA graph.
"""

from __future__ import annotations

import functools
import importlib.util
from dataclasses import dataclass

import torch
from torch import Tensor

from .pack import WORD_BITS

__all__ = ["KernelConfig", "default_config", "littlebit_matmul", "triton_available"]

#: Largest row block of one program.
MAX_BLOCK_ROWS = 64


def triton_available() -> bool:
    """Whether Triton is importable and a CUDA device is present."""
    return importlib.util.find_spec("triton") is not None and torch.cuda.is_available()


@dataclass(frozen=True)
class KernelConfig:
    """Launch parameters of one call.

    Attributes:
        down_block_n: Latent dimensions per ``down`` program.
        down_block_w: Packed input words per inner step of ``down`` (32 input channels each).
        split_k: ``down`` programs that share one block of latent dimensions.
        up_block_n: Output features per ``up`` program.
        up_block_w: Packed latent words per inner step of ``up``.
        num_warps: Warps per program.
    """

    down_block_n: int = 64
    down_block_w: int = 4
    split_k: int = 1
    up_block_n: int = 64
    up_block_w: int = 2
    num_warps: int = 4


@functools.cache
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _row_block(m: int) -> int:
    # a single row also takes the tensor-core path: padded to 16 rows it beats a CUDA-core broadcast by 2-3x
    return min(MAX_BLOCK_ROWS, max(16, 1 << (m - 1).bit_length()))


def default_config(m: int, in_features: int, rank: int, paths: int, device: torch.device) -> KernelConfig:
    """The heuristic: split the input channels of ``down`` (a power of two, at least four inner steps per split)
    until its grid has about two programs per SM."""
    cfg = KernelConfig()
    row_blocks = -(-m // _row_block(m))
    blocks = -(-rank // cfg.down_block_n) * paths * row_blocks
    sms = _sm_count(device.index if device.index is not None else torch.cuda.current_device())
    steps = -(-in_features // (WORD_BITS * cfg.down_block_w))
    wanted = max(1, min(-(-2 * sms // blocks), steps // 4))
    return KernelConfig(split_k=1 << (wanted.bit_length() - 1))


@functools.cache
def _kernels():  # type: ignore[no-untyped-def]
    """The two Triton kernels, compiled on first use (Triton is imported lazily)."""
    import triton
    import triton.language as tl

    @triton.jit
    def _unpack(words, BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr):
        shifts = tl.arange(0, 32)
        bits = (words[:, :, None] >> shifts[None, None, :]) & 1
        return tl.reshape(bits * 2 - 1, (BLOCK_N, BLOCK_W * 32))

    @triton.jit
    def down_kernel(
        x_ptr,
        g_ptr,
        w_ptr,
        l_ptr,
        t_ptr,
        M,
        K,
        R,
        KW,
        words_per_split,
        row_blocks,
        stride_xm,
        stride_gp,
        stride_wp,
        stride_wr,
        stride_lp,
        stride_tp,
        stride_tm,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_W: tl.constexpr,
        ATOMIC: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        p = tl.program_id(1) // row_blocks
        pid_m = tl.program_id(1) % row_blocks
        pid_k = tl.program_id(2)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = offs_n < R
        w_start = pid_k * words_per_split
        w_end = tl.minimum(w_start + words_per_split, KW)
        k_end = tl.minimum(w_end * 32, K)
        w_base = w_ptr + p * stride_wp + offs_n[:, None] * stride_wr
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_ok = offs_m < M
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for w0 in range(w_start, w_end, BLOCK_W):
            offs_w = w0 + tl.arange(0, BLOCK_W)
            words = tl.load(w_base + offs_w[None, :], mask=n_ok[:, None] & (offs_w < w_end)[None, :], other=0)
            signs = _unpack(words, BLOCK_N, BLOCK_W)
            offs_k = w0 * 32 + tl.arange(0, BLOCK_W * 32)
            k_ok = offs_k < k_end
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                mask=m_ok[:, None] & k_ok[None, :],
                other=0.0,
            )
            g = tl.load(g_ptr + p * stride_gp + offs_k, mask=k_ok, other=0.0)
            xg = x * g[None, :]
            acc += tl.dot(xg, tl.trans(signs.to(xg.dtype)))
        acc = acc * tl.load(l_ptr + p * stride_lp + offs_n, mask=n_ok, other=0.0).to(tl.float32)[None, :]
        ptrs = t_ptr + p * stride_tp + offs_m[:, None] * stride_tm + offs_n[None, :]
        ok = m_ok[:, None] & n_ok[None, :]
        if ATOMIC:
            tl.atomic_add(ptrs, acc, mask=ok, sem="relaxed")
        else:
            tl.store(ptrs, acc, mask=ok)

    @triton.jit
    def up_kernel(
        t_ptr,
        w_ptr,
        h_ptr,
        b_ptr,
        y_ptr,
        M,
        N,
        R,
        RW,
        stride_tp,
        stride_tm,
        stride_wp,
        stride_wn,
        stride_hp,
        stride_ym,
        PATHS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = offs_n < N
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_ok = offs_m < M
        y = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for p in tl.static_range(PATHS):
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for w0 in range(0, RW, BLOCK_W):
                offs_w = w0 + tl.arange(0, BLOCK_W)
                words = tl.load(
                    w_ptr + p * stride_wp + offs_n[:, None] * stride_wn + offs_w[None, :],
                    mask=n_ok[:, None] & (offs_w < RW)[None, :],
                    other=0,
                )
                signs = _unpack(words, BLOCK_N, BLOCK_W)
                offs_k = w0 * 32 + tl.arange(0, BLOCK_W * 32)
                t = tl.load(
                    t_ptr + p * stride_tp + offs_m[:, None] * stride_tm + offs_k[None, :],
                    mask=m_ok[:, None] & (offs_k < R)[None, :],
                    other=0.0,
                )
                t = t.to(y_ptr.dtype.element_ty)
                acc += tl.dot(t, tl.trans(signs.to(t.dtype)))
            h = tl.load(h_ptr + p * stride_hp + offs_n, mask=n_ok, other=0.0).to(tl.float32)
            y += acc * h[None, :]
        if HAS_BIAS:
            y += tl.load(b_ptr + offs_n, mask=n_ok, other=0.0).to(tl.float32)[None, :]
        ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
        tl.store(ptrs, y.to(y_ptr.dtype.element_ty), mask=m_ok[:, None] & n_ok[None, :])

    return down_kernel, up_kernel


def littlebit_matmul(
    x: Tensor,
    v_bits: Tensor,
    u_bits: Tensor,
    g: Tensor,
    l: Tensor,
    h: Tensor,
    bias: Tensor | None,
    config: KernelConfig | None = None,
) -> Tensor:
    """The LittleBit map of ``x [M, in]`` for packed signs ``v_bits [P, r, in/32]``, ``u_bits [P, out, r/32]`` and
    scales ``g [P, in]``, ``l [P, r]``, ``h [P, out]`` (all contiguous, scales in the dtype of ``x``).

    Returns:
        ``[M, out]`` in the dtype of ``x``.
    """
    import triton

    down, up = _kernels()
    m, k = x.shape
    paths, rank, k_words = v_bits.shape
    n, r_words = u_bits.shape[1], u_bits.shape[2]
    if x.stride(1) != 1:
        x = x.contiguous()
    cfg = config or default_config(m, k, rank, paths, x.device)
    block_m = _row_block(m)
    row_blocks = triton.cdiv(m, block_m)
    words_per_split = triton.cdiv(triton.cdiv(k_words, cfg.split_k), cfg.down_block_w) * cfg.down_block_w
    split = triton.cdiv(k_words, words_per_split)
    atomic = split > 1
    t = (torch.zeros if atomic else torch.empty)((paths, m, rank), dtype=torch.float32, device=x.device)
    down[(triton.cdiv(rank, cfg.down_block_n), paths * row_blocks, split)](
        x,
        g,
        v_bits,
        l,
        t,
        m,
        k,
        rank,
        k_words,
        words_per_split,
        row_blocks,
        x.stride(0),
        g.stride(0),
        v_bits.stride(0),
        v_bits.stride(1),
        l.stride(0),
        t.stride(0),
        t.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=cfg.down_block_n,
        BLOCK_W=cfg.down_block_w,
        ATOMIC=atomic,
        num_warps=cfg.num_warps,
    )
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    up[(triton.cdiv(n, cfg.up_block_n), row_blocks)](
        t,
        u_bits,
        h,
        bias if bias is not None else h,
        y,
        m,
        n,
        rank,
        r_words,
        t.stride(0),
        t.stride(1),
        u_bits.stride(0),
        u_bits.stride(1),
        h.stride(0),
        y.stride(0),
        PATHS=paths,
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=cfg.up_block_n,
        BLOCK_W=cfg.up_block_w,
        num_warps=cfg.num_warps,
    )
    return y
