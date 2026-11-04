"""TEAL's sparse GEMV in Triton: ``y = s_t(x) @ W^T`` that reads only the weight rows of the inputs it keeps.

The weight is stored transposed, ``W^T`` ``[in, out]`` row-major, so that the weights of one input channel are one
contiguous row. A program owns ``BLOCK_N`` output columns and a slice of the input channels; it loads ``BLOCK_K``
inputs, thresholds them (``|x| > t``, :func:`.sparsify.sparsify`) and issues the loads of only the weight rows of
the kept channels. Predicated-off loads cost no memory traffic, so at sparsity ``p`` a memory-bound call reads about
``1 - p`` of the weight bytes (the kernel of Liu et al., ICLR 2025, Section 4.4, github.com/FasterDecoding/TEAL).

With ``M > 1`` rows (a verification pass, a batch of sequences) a weight row is needed when any row keeps the
channel, so the saving shrinks with the union of the kept sets; :class:`.triton_backend.TritonBackend` falls back to
the dense matmul above a row cutoff. Rows ``M <= 16`` are one block; the kernel multiplies with ``tl.dot`` for
``M > 1`` and with a broadcast reduction for ``M = 1``.

The input channels are split over ``SPLIT_K`` programs per column block when the column blocks alone do not fill
the GPU; their partial sums meet in a float32 buffer through atomic adds. The threshold is read from a 0-dim device
tensor, so the call never synchronizes with the host and is safe to capture in a CUDA graph.
"""

from __future__ import annotations

import functools
import importlib.util
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ...errors import ConfigError

__all__ = ["GemvConfig", "default_config", "teal_gemv", "triton_available"]

#: Largest row count one kernel call takes (one ``BLOCK_M`` block).
MAX_ROWS = 16


def triton_available() -> bool:
    """Whether Triton is importable and a CUDA device is present."""
    return importlib.util.find_spec("triton") is not None and torch.cuda.is_available()


@dataclass(frozen=True)
class GemvConfig:
    """Launch parameters of one call.

    Attributes:
        block_n: Output columns per program.
        block_k: Input channels per inner step.
        split_k: Programs that share one column block (``1``: no atomics, the output is written directly).
        num_warps: Warps per program.
        num_stages: Software pipelining depth of the inner loop.
    """

    block_n: int = 128
    block_k: int = 64
    split_k: int = 1
    num_warps: int = 4
    num_stages: int = 2


@functools.cache
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def default_config(m: int, n: int, k: int, device: torch.device) -> GemvConfig:
    """The launch parameters of the heuristic.

    One row: 256 columns and 128 input channels per step, the input split (a power of two, at least 256 channels
    per split) until the grid has about two programs per SM; this config was within 10% of the best of a 216-config
    grid on every Gemma-4-31B map (L40, ``scripts/bench_teal_kernel.py --sweep``). Several rows: 128 columns and 32
    channels per step (the ``tl.dot`` tile), about four programs per SM.
    """
    block_n, block_k, per_sm, min_k = (256, 128, 2, 256) if m == 1 else (128, 32, 4, 512)
    blocks = -(-n // block_n)
    sms = _sm_count(device.index if device.index is not None else torch.cuda.current_device())
    wanted = max(1, min(-(-per_sm * sms // blocks), k // min_k))
    split = 1 << (wanted.bit_length() - 1)
    return GemvConfig(block_n=block_n, block_k=block_k, split_k=split, num_warps=4, num_stages=2)


@functools.cache
def _kernel() -> Any:
    """The Triton kernel, compiled on first use (Triton is imported lazily)."""
    import triton
    import triton.language as tl

    @triton.jit
    def teal_gemv_kernel(
        x_ptr,
        w_ptr,
        t_ptr,
        out_ptr,
        M,
        N,
        K,
        k_per_split,
        stride_xm,
        stride_wk,
        stride_om,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ATOMIC: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = offs_n < N
        # rounded to the input dtype, as ``x.abs() > t`` in PyTorch rounds a 0-dim float32 ``t`` (same masks)
        threshold = tl.load(t_ptr).to(x_ptr.dtype.element_ty).to(tl.float32)
        k_start = pid_k * k_per_split
        k_end = tl.minimum(k_start + k_per_split, K)
        if BLOCK_M == 1:
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for k0 in range(k_start, k_end, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                k_ok = offs_k < k_end
                x = tl.load(x_ptr + offs_k, mask=k_ok, other=0.0).to(tl.float32)
                keep = tl.abs(x) > threshold
                w = tl.load(
                    w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :],
                    mask=keep[:, None] & n_ok[None, :],
                    other=0.0,
                )
                acc += tl.sum(tl.where(keep, x, 0.0)[:, None] * w.to(tl.float32), axis=0)
            if ATOMIC:
                tl.atomic_add(out_ptr + offs_n, acc, mask=n_ok, sem="relaxed")
            else:
                tl.store(out_ptr + offs_n, acc.to(out_ptr.dtype.element_ty), mask=n_ok)
        else:
            offs_m = tl.arange(0, BLOCK_M)
            m_ok = offs_m < M
            acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(k_start, k_end, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                k_ok = offs_k < k_end
                xs = tl.load(
                    x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                    mask=m_ok[:, None] & k_ok[None, :],
                    other=0.0,
                )
                keep2 = tl.abs(xs.to(tl.float32)) > threshold
                row = tl.max(keep2.to(tl.int32), axis=0) > 0
                w = tl.load(
                    w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :],
                    mask=row[:, None] & n_ok[None, :],
                    other=0.0,
                )
                acc2 += tl.dot(tl.where(keep2, xs, 0.0).to(w.dtype), w)
            ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
            ok = m_ok[:, None] & n_ok[None, :]
            if ATOMIC:
                tl.atomic_add(ptrs, acc2, mask=ok, sem="relaxed")
            else:
                tl.store(ptrs, acc2.to(out_ptr.dtype.element_ty), mask=ok)

    return teal_gemv_kernel


def teal_gemv(x: Tensor, weight_t: Tensor, threshold: Tensor, config: GemvConfig | None = None) -> Tensor:
    """``sparsify(x, threshold) @ weight_t`` for ``x`` ``[M, K]`` (``M <= 16``) and ``weight_t`` ``[K, N]`` row-major.

    Args:
        x: Inputs, 16-bit, ``[M, K]`` with unit stride along ``K``.
        weight_t: The transposed weight ``[K, N]``, contiguous, same dtype as ``x``.
        threshold: 0-dim float32 tensor on the device (TEAL's ``t``; inputs with ``|x| <= t`` are dropped).
        config: Launch parameters (default: :func:`default_config`).

    Returns:
        ``[M, N]`` in the dtype of ``x``.
    """
    import triton

    m, k = x.shape
    n = weight_t.shape[1]
    if m > MAX_ROWS:
        raise ConfigError(f"teal_gemv takes at most {MAX_ROWS} rows, got {m}")
    if x.stride(1) != 1:
        x = x.contiguous()
    cfg = config or default_config(m, n, k, x.device)
    block_m = 1 if m == 1 else 16
    k_per_split = triton.cdiv(triton.cdiv(k, cfg.split_k), cfg.block_k) * cfg.block_k
    split = triton.cdiv(k, k_per_split)
    atomic = split > 1
    out = (
        torch.zeros((m, n), dtype=torch.float32, device=x.device)
        if atomic
        else torch.empty((m, n), dtype=x.dtype, device=x.device)
    )
    grid = (triton.cdiv(n, cfg.block_n), split)
    _kernel()[grid](
        x,
        weight_t,
        threshold,
        out,
        m,
        n,
        k,
        k_per_split,
        x.stride(0),
        weight_t.stride(0),
        out.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=cfg.block_n,
        BLOCK_K=cfg.block_k,
        ATOMIC=atomic,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return out.to(x.dtype) if atomic else out
