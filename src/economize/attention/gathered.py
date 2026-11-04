"""Decode attention over a physically compacted KV cache (the entries a prompt-time eviction kept).

SnapKV and H2O as KVCache-Factory runs it select, once after the prompt is prefilled, the prompt entries every KV
head keeps; generated entries are never evicted. A cache that only masks the evicted entries still reads every
block that holds one kept entry, which after a selection scattered over the prompt is nearly all of them.
:class:`GatheredKV` holds the kept prompt entries of every row and KV head contiguously, and
:func:`gathered_decode_attention` is the split-K kernel of :mod:`.decode` over a virtual key axis: for row ``b`` and KV
head ``h`` the first ``count[b, h]`` virtual keys are the compact entries (whose original slots ``index`` give their
mask columns, so causal, tree and sliding-window masks apply unchanged), and virtual key ``count[b, h] + i`` is dense
slot ``start[b] + i``. The kernel thus reads ``count[b, h] + keys - start[b]`` entries per row instead of ``keys``.

The kernel visits the same keys as :func:`~.decode.decode_attention` with the evicted entries masked, in a
different order, so the two agree up to the summation order of the softmax (bit-exact only when the blocks happen
to coincide).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..errors import ConfigError
from .decode import (
    _DOT_DTYPES,
    DecodePlan,
    check_value_tiles,
    mask_operand,
    next_power_of_2,
    plan,
    sinks_operand,
    triton_available,
)

__all__ = ["GatheredKV", "gathered_decode_attention"]


@dataclass
class GatheredKV:
    """Compacted prompt entries of one layer (a :class:`~.packed.PackedKV` of format ``"gathered"``).

    Attributes:
        key: ``[B, Hkv, C, D]`` kept prompt keys of every row and KV head, in slot order (``C >= count[b, h]``).
        value: ``[B, Hkv, C, Dv]`` the matching values.
        index: ``int32`` ``[B, Hkv, C]`` original slot of every compact entry (its mask column).
        count: ``int32`` ``[B, Hkv]`` compact entries of every row and KV head.
        start: ``int32`` ``[B]`` first dense slot read for row ``b`` (the prompt's end); slots below it are read
            only through the compact entries.
        dropped: ``min(start[b] - count[b, h])``, on the host: the virtual key axis is ``keys - dropped`` long.
    """

    key: torch.Tensor
    value: torch.Tensor
    index: torch.Tensor
    count: torch.Tensor
    start: torch.Tensor
    dropped: int
    format: str = "gathered"

    def attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
        scale: float,
        softcap: float | None = None,
        sinks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return gathered_decode_attention(query, key, value, mask, scale, self, softcap=softcap, sinks=sinks)


def _check(gathered: GatheredKV, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    batch, kv_heads = key.shape[:2]
    capacity = gathered.key.shape[2]
    expected = {
        "key": ((batch, kv_heads, capacity, key.shape[3]), key.dtype),
        "value": ((batch, kv_heads, capacity, value.shape[3]), value.dtype),
        "index": ((batch, kv_heads, capacity), torch.int32),
        "count": ((batch, kv_heads), torch.int32),
        "start": ((batch,), torch.int32),
    }
    for name, (shape, dtype) in expected.items():
        tensor = getattr(gathered, name)
        if tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.device != query.device:
            raise ConfigError(
                f"gathered {name} is {tuple(tensor.shape)} {tensor.dtype} on {tensor.device}, expected {shape} "
                f"{dtype} on {query.device}"
            )
        if (tensor.stride(-1) != 1) if name in ("key", "value") else not tensor.is_contiguous():
            raise ConfigError(f"gathered {name} is not laid out for the kernel (strides {tensor.stride()})")
    if gathered.dropped < 0 or gathered.dropped > key.shape[2]:
        raise ConfigError(f"gathered cache drops {gathered.dropped} of {key.shape[2]} slots")


def gathered_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
    gathered: GatheredKV,
    softcap: float | None = None,
    sinks: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    launch: DecodePlan | None = None,
) -> torch.Tensor:
    """:func:`~.decode.decode_attention` over a cache whose prompt entries were compacted (see the module docstring).

    Args:
        query: ``[B, Hq, Q, D]``, bf16, fp16 or fp32.
        key: Dense ``[B, Hkv, S, D]`` in the query's dtype; read only at slots ``>= gathered.start[b]``.
        value: Dense ``[B, Hkv, S, Dv]``; likewise.
        mask: Boolean ``[B, 1 or Hq, Q, S]`` over the original slots, or ``None``.
        scale: Multiplier of the scores.
        gathered: The compact entries (see :class:`GatheredKV`).
        softcap: Optional tanh soft cap of the scaled scores.
        sinks: Optional ``[Hq]`` attention-sink logits.
        out: Optional ``[B, Q, Hq, Dv]`` output buffer.
        launch: Launch shape to use instead of :func:`~.decode.plan`'s (tuning and tests).

    Returns:
        ``[B, Q, Hq, Dv]`` in the query's dtype.
    """
    if not triton_available():
        raise RuntimeError("gathered_decode_attention needs triton")
    batch, q_heads, queries, head_dim = query.shape
    _, kv_heads, keys, _ = key.shape
    head_dv = value.shape[-1]
    if q_heads % kv_heads:
        raise ConfigError(f"{q_heads} query heads are not a multiple of {kv_heads} KV heads")
    if query.dtype not in _DOT_DTYPES or key.dtype != query.dtype or value.dtype != query.dtype:
        raise ConfigError(f"gathered_decode_attention takes K/V in the query's dtype, got {key.dtype} / {value.dtype}")
    _check(gathered, query, key, value)

    groups = q_heads // kv_heads
    rows = groups * queries
    virtual = max(1, keys - gathered.dropped)
    p = launch or plan(
        batch, kv_heads, rows, virtual, max(head_dim, head_dv), query.device, element_size=query.element_size(),
        value_dim=head_dv,
    )
    block_dv = max(16, next_power_of_2(head_dv))
    block_d = max(16, next_power_of_2(head_dim))
    check_value_tiles(p, block_dv)
    programs = batch * kv_heads * p.m_tiles
    part_o = torch.empty(programs * p.splits * p.block_m * block_dv, device=query.device, dtype=torch.float32)
    part_m = torch.empty(programs * p.splits * p.block_m, device=query.device, dtype=torch.float32)
    part_l = torch.empty_like(part_m)
    if out is None:
        out = torch.empty(batch, queries, q_heads, head_dv, device=query.device, dtype=query.dtype)
    m, mask_strides, per_head = mask_operand(mask, keys, query)
    sinks = sinks_operand(sinks, q_heads)
    num_warps = p.num_warps or (4 if block_d <= 256 else 8)
    ck, cv, idx = gathered.key, gathered.value, gathered.index
    from ._decode_kernels import _DOT, _merge_kernel
    from ._gathered_kernels import _gathered_split_kernel

    with torch.cuda.device(query.device):
        _gathered_split_kernel[(programs * p.dv_tiles, p.splits)](
            query, key, value, ck, cv, idx, gathered.count, gathered.start, m, part_o, part_m, part_l,
            *query.stride(), *key.stride(), *value.stride(),
            ck.stride(0), ck.stride(1), ck.stride(2),
            cv.stride(0), cv.stride(1), cv.stride(2),
            idx.stride(0), idx.stride(1),
            *mask_strides,
            keys, queries, rows, kv_heads, p.m_tiles, p.splits, float(scale), float(softcap or 1.0),
            GROUPS=groups, HEAD_D=head_dim, HEAD_DV=head_dv, BLOCK_M=p.block_m, BLOCK_N=p.block_n, BLOCK_D=block_d,
            BLOCK_DV=p.block_dv or block_dv, SPLIT_LEN=p.split_len, HAS_MASK=mask is not None, MASK_PER_HEAD=per_head,
            HAS_SOFTCAP=softcap is not None, IEEE=query.dtype == torch.float32, DOT_DTYPE=_DOT[query.dtype],
            DV_TILES=p.dv_tiles, PART_DV=block_dv, num_warps=num_warps, num_stages=p.num_stages,
        )
        _merge_kernel[(programs, p.block_m)](
            part_o, part_m, part_l, out, part_m if sinks is None else sinks, *out.stride(),
            queries, rows, kv_heads, p.m_tiles, p.splits,
            GROUPS=groups, HEAD_DV=head_dv, BLOCK_M=p.block_m, BLOCK_DV=block_dv, HAS_SINKS=sinks is not None,
            num_warps=4,
        )
    return out
