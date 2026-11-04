"""Tree compaction of many slot buffers in one kernel launch (:func:`gather_slots`).

After a draft tree is verified, the K/V of the accepted path move to the slots that follow the root
(:meth:`economize.runtime.KVCache.gather`). Done with advanced indexing, this costs two kernels per buffer: on a
60-layer model with keys and values that is 240 launches, about 3.5 ms per cycle on an L40 when launched eagerly and
still 0.93 ms when replayed from a CUDA graph, since every launch moves only a few slots. :func:`gather_slots` moves
the slots of every eligible buffer in a single Triton launch that reads the buffers' base addresses from a table.

A buffer is eligible when it is a contiguous CUDA tensor ``[B, heads, num_slots, *rest]`` whose slots span a multiple
of four bytes (any K/V buffer, fp8 or 16-bit, and per-slot float32 scales). Other buffers (bool keep masks, CPU
tensors, tensors that require gradients) take the indexing path. ``ECONOMIZE_SLOT_GATHER=0`` sends every buffer
there.
"""

from __future__ import annotations

import functools
import importlib.util
import math
import os
from collections.abc import Sequence
from typing import Any

import torch

__all__ = ["SLOT_GATHER", "gather_slots", "gather_slots_reference"]

#: Whether eligible buffers use the fused kernel (default on). Debug override ``ECONOMIZE_SLOT_GATHER=0`` turns
#: it off.
SLOT_GATHER = os.environ.get("ECONOMIZE_SLOT_GATHER", "1") != "0"

#: Pointer tables kept on the device, keyed by the buffers' addresses and shapes. Static caches keep their buffers,
#: so a decoder reuses one table for its whole life; the dict is cleared when it grows past :data:`_MAX_TABLES`.
_TABLES: dict[tuple[Any, ...], torch.Tensor] = {}
_MAX_TABLES = 64


def gather_slots_reference(buffer: torch.Tensor, rows: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> None:
    """Copy slots ``src[i, j]`` of row ``rows[i]`` of ``buffer`` ``[B, heads, num_slots, ...]`` to slots
    ``dst[i, j]`` (advanced indexing; every source is read before any destination is written)."""
    r = rows.to(buffer.device)[:, None]
    buffer[r, :, dst.to(buffer.device)] = buffer[r, :, src.to(buffer.device)]


def gather_slots(buffers: Sequence[torch.Tensor], rows: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> None:
    """Copy, in every buffer ``[B, heads, num_slots, ...]``, slots ``src[i, j]`` of row ``rows[i]`` to slots
    ``dst[i, j]`` (``rows`` ``[R]``, ``src`` and ``dst`` ``[R, J]``, int64).

    All sources of a row are read before its destinations are written, so the two may overlap. Destinations of a
    row must be distinct except for slots whose content does not matter (the trash slot of padded entries).
    """
    fused: list[torch.Tensor] = []
    for buffer in buffers:
        if _eligible(buffer):
            fused.append(buffer)
        else:
            gather_slots_reference(buffer, rows, src, dst)
    if not fused:
        return
    if rows.numel() == 0 or src.shape[1] == 0:
        return
    device = fused[0].device
    table = _table(fused, device)
    if table is None:  # recording a CUDA graph before the table exists: no host-to-device copy may be issued
        for buffer in fused:
            gather_slots_reference(buffer, rows, src, dst)
        return
    rows, src, dst = (t.to(device, torch.long).contiguous() for t in (rows, src, dst))
    heads = max(b.shape[1] for b in fused)
    words = max(_words(b) for b in fused)
    span = src.shape[1]
    block_j = max(2, 1 << (span - 1).bit_length())
    chunk = max(16, min(1 << (words - 1).bit_length(), 4096 // block_j))
    kernel = _kernel()
    kernel[(len(fused) * heads, rows.shape[0])](
        table, rows, src, dst, span, heads, BLOCK_J=block_j, CHUNK=chunk, num_warps=4
    )


def _words(buffer: torch.Tensor) -> int:
    return math.prod(buffer.shape[3:]) * buffer.element_size() // 4


def _eligible(buffer: torch.Tensor) -> bool:
    return (
        SLOT_GATHER
        and buffer.is_cuda
        and buffer.dim() >= 4
        and buffer.is_contiguous()
        and not buffer.requires_grad
        and (math.prod(buffer.shape[3:]) * buffer.element_size()) % 4 == 0
        and buffer.data_ptr() % 4 == 0
        and _triton_available()
    )


@functools.cache
def _triton_available() -> bool:
    return importlib.util.find_spec("triton") is not None


def _table(buffers: list[torch.Tensor], device: torch.device) -> torch.Tensor | None:
    """``[N, 4]`` int64 on ``device``: base address, heads, slots and 32-bit words per slot of every buffer."""
    key = tuple((b.data_ptr(), tuple(b.shape), b.dtype) for b in buffers)
    table = _TABLES.get(key)
    if table is not None:
        return table
    if torch.cuda.is_current_stream_capturing():
        return None
    if len(_TABLES) >= _MAX_TABLES:
        _TABLES.clear()
    rows = [[b.data_ptr(), b.shape[1], b.shape[2], _words(b)] for b in buffers]
    table = torch.tensor(rows, dtype=torch.long).to(device)  # once per buffer set
    _TABLES[key] = table
    return table


@functools.cache
def _kernel() -> Any:
    """The Triton kernel, compiled on first use (Triton is imported lazily)."""
    import triton
    import triton.language as tl

    @triton.jit
    def slot_gather_kernel(table, rows, src, dst, span, max_heads, BLOCK_J: tl.constexpr, CHUNK: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.program_id(1)
        n = pid // max_heads
        head = pid % max_heads
        heads = tl.load(table + n * 4 + 1)
        if head < heads:
            base = tl.load(table + n * 4).to(tl.pointer_type(tl.int32))
            slots = tl.load(table + n * 4 + 2)
            words = tl.load(table + n * 4 + 3)
            row = tl.load(rows + r)
            plane = base + (row * heads + head) * slots * words
            j = tl.arange(0, BLOCK_J)
            j_ok = j < span
            s = tl.load(src + r * span + j, mask=j_ok, other=0)
            d = tl.load(dst + r * span + j, mask=j_ok, other=0)
            for start in range(0, words, CHUNK):
                c = start + tl.arange(0, CHUNK)
                ok = j_ok[:, None] & (c < words)[None, :]
                tile = tl.load(plane + s[:, None] * words + c[None, :], mask=ok)
                tl.debug_barrier()  # every source of the chunk is read before a destination is written
                tl.store(plane + d[:, None] * words + c[None, :], tile, mask=ok)

    return slot_gather_kernel
