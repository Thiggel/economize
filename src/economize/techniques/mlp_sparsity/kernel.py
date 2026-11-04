"""A row-masked GEMV in Triton: ``y = (x @ W^T) * mask`` that reads only the weight rows of the outputs it keeps.

CATS and Deja Vu know, before the up (and, for Deja Vu, the gate) map runs, which FFN channels a token needs. The
weight ``W`` ``[N, K]`` of such a map is stored row-major as in ``nn.Linear``, so the weights of one output channel
are one contiguous row. A program owns ``BLOCK_N`` output channels; it loads the weight tile of the channels that
some row keeps and predicates the loads of the others off, so at sparsity ``p`` a memory-bound call reads about
``1 - p`` of the weight bytes (the gate/up kernel of CATS, Lee et al., COLM 2024, Section 4, and the sparse MLP of
Deja Vu, Liu et al., ICML 2023, Section 4.3). The down map of a sparse hidden vector is TEAL's column-sparse GEMV
(:func:`economize.techniques.sparsity.teal_gemv.teal_gemv` with threshold 0).

Rows ``M <= 16`` are one block: ``M = 1`` multiplies with a broadcast reduction, ``M > 1`` with ``tl.dot``, and a
weight row is loaded when any of the ``M`` rows keeps its channel. The mask is read from device memory, so the call
never synchronizes with the host and is safe to capture in a CUDA graph.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ...errors import ConfigError
from ..sparsity.teal_gemv import MAX_ROWS

__all__ = ["RowGemvConfig", "default_row_config", "row_masked_gemv"]


@dataclass(frozen=True)
class RowGemvConfig:
    """Launch parameters of one :func:`row_masked_gemv` call.

    Attributes:
        block_n: Output channels per program.
        block_k: Input channels per inner step.
        num_warps: Warps per program.
        num_stages: Software pipelining depth of the inner loop.
    """

    block_n: int = 32
    block_k: int = 256
    num_warps: int = 4
    num_stages: int = 2


def default_row_config(m: int) -> RowGemvConfig:
    """The launch parameters of the heuristic: 32 channels by 256 inputs for one row, a 64 by 64 ``tl.dot`` tile for
    several (untuned; the GPU sweep is pending)."""
    return RowGemvConfig() if m == 1 else RowGemvConfig(block_n=64, block_k=64)


@functools.cache
def _kernel() -> Any:
    """The Triton kernel, compiled on first use (Triton is imported lazily)."""
    import triton
    import triton.language as tl

    @triton.jit
    def row_masked_gemv_kernel(
        x_ptr,
        w_ptr,
        mask_ptr,
        out_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_wn,
        stride_mm,
        stride_om,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = offs_n < N
        if BLOCK_M == 1:
            keep = (tl.load(mask_ptr + offs_n, mask=n_ok, other=0) != 0) & n_ok
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                k_ok = offs_k < K
                x = tl.load(x_ptr + offs_k, mask=k_ok, other=0.0).to(tl.float32)
                w = tl.load(
                    w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                    mask=keep[:, None] & k_ok[None, :],
                    other=0.0,
                )
                acc += tl.sum(w.to(tl.float32) * x[None, :], axis=1)
            acc = tl.where(keep, acc, 0.0)
            tl.store(out_ptr + offs_n, acc.to(out_ptr.dtype.element_ty), mask=n_ok)
        else:
            offs_m = tl.arange(0, BLOCK_M)
            m_ok = offs_m < M
            ok = m_ok[:, None] & n_ok[None, :]
            keep2 = (tl.load(mask_ptr + offs_m[:, None] * stride_mm + offs_n[None, :], mask=ok, other=0) != 0) & ok
            row = tl.max(keep2.to(tl.int32), axis=0) > 0
            acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                k_ok = offs_k < K
                xs = tl.load(
                    x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                    mask=m_ok[:, None] & k_ok[None, :],
                    other=0.0,
                )
                w = tl.load(
                    w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                    mask=row[:, None] & k_ok[None, :],
                    other=0.0,
                )
                acc2 += tl.dot(xs.to(w.dtype), tl.trans(w))
            acc2 = tl.where(keep2, acc2, 0.0)
            ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
            tl.store(ptrs, acc2.to(out_ptr.dtype.element_ty), mask=ok)

    return row_masked_gemv_kernel


def row_masked_gemv(x: Tensor, weight: Tensor, mask: Tensor, config: RowGemvConfig | None = None) -> Tensor:
    """``(x @ weight.T) * mask`` for ``x`` ``[M, K]`` (``M <= 16``), ``weight`` ``[N, K]`` and ``mask`` ``[M, N]``.

    Args:
        x: Inputs ``[M, K]`` with unit stride along ``K``.
        weight: The weight of an ``nn.Linear``, ``[N, K]`` with unit stride along ``K``, same dtype as ``x``.
        mask: Kept outputs, boolean ``[M, N]``.
        config: Launch parameters (default: :func:`default_row_config`).

    Returns:
        ``[M, N]`` in the dtype of ``x``; the dropped outputs are exactly 0.

    Raises:
        ConfigError: More than 16 rows, or shapes that do not match.
    """
    import triton

    m, k = x.shape
    n = weight.shape[0]
    if m > MAX_ROWS:
        raise ConfigError(f"row_masked_gemv takes at most {MAX_ROWS} rows, got {m}")
    if weight.shape[1] != k or tuple(mask.shape) != (m, n):
        shapes = f"x {tuple(x.shape)}, weight {tuple(weight.shape)}, mask {tuple(mask.shape)}"
        raise ConfigError(f"shapes do not match: {shapes}")
    if x.stride(1) != 1:
        x = x.contiguous()
    if weight.stride(1) != 1:
        weight = weight.contiguous()
    keep = mask.contiguous().view(torch.uint8) if mask.dtype == torch.bool else (mask != 0).to(torch.uint8)
    cfg = config or default_row_config(m)
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(n, cfg.block_n),)
    _kernel()[grid](
        x,
        weight,
        keep,
        out,
        m,
        n,
        k,
        x.stride(0),
        weight.stride(0),
        keep.stride(0),
        out.stride(0),
        BLOCK_M=1 if m == 1 else 16,
        BLOCK_N=cfg.block_n,
        BLOCK_K=cfg.block_k,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return out
