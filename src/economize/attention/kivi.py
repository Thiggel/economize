"""Decode attention that reads KIVI's packed 2/4/8-bit K/V codes (Liu et al., ICML 2024) directly.

KIVI quantizes keys per channel (one fp16 scale and zero point per channel for every group of ``key_group``
consecutive tokens) and values per token (one scale and zero point per group of ``value_group`` channels), and
keeps the most recent tokens in full precision. :func:`kivi_decode_attention` is the split-K kernel of
:mod:`.decode` with a different load: for row ``b`` the first ``keys_quantized[b]`` keys and the first
``values_quantized[b]`` values are read as packed codes and dequantized in registers, every later slot is read
from the dense buffer. A 2-bit key therefore costs a quarter of a byte per element plus its share of the group's
scale and zero point, instead of two bytes.

Codes are packed along the channel axis, ``8 // bits`` codes per byte, channel ``d`` in byte ``d // (8 // bits)``
at bit offset ``(d % (8 // bits)) * bits`` (:func:`pack_codes`). Dequantization computes ``code * scale + zero``
in fp32 with separate roundings (no fused multiply-add), which is what ``torch`` computes for the same tensors, so
the kernel attends over exactly the values that :func:`dequantize` (and a cache that stores them densely) gives.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from ..errors import ConfigError
from .decode import (
    _DOT_DTYPES,
    DecodePlan,
    _sm_count,
    mask_operand,
    next_power_of_2,
    plan,
    sinks_operand,
    triton_available,
)

__all__ = ["KIVIPacked", "dequantize", "kivi_decode_attention", "kivi_plan", "pack_codes", "unpack_codes"]

def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Integer codes ``[..., D]`` in ``[0, 2**bits)`` packed along the last axis into ``uint8`` ``[..., D*bits/8]``."""
    per_byte = 8 // bits
    if codes.shape[-1] % per_byte:
        raise ConfigError(f"{codes.shape[-1]} channels do not pack into whole bytes at {bits} bits")
    grouped = codes.to(torch.int32).reshape(*codes.shape[:-1], codes.shape[-1] // per_byte, per_byte)
    shifts = torch.arange(per_byte, device=codes.device, dtype=torch.int32) * bits
    return (grouped << shifts).sum(-1).to(torch.uint8)


def unpack_codes(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Inverse of :func:`pack_codes` (``int32`` codes)."""
    per_byte = 8 // bits
    shifts = torch.arange(per_byte, device=packed.device, dtype=torch.int32) * bits
    codes = (packed.to(torch.int32)[..., None] >> shifts) & (2**bits - 1)
    return codes.reshape(*packed.shape[:-1], packed.shape[-1] * per_byte)


def dequantize(
    codes: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bits: int, axis: str, group: int, dtype: torch.dtype
) -> torch.Tensor:
    """Dense values of packed ``codes`` ``[B, H, S, D / per_byte]``.

    Args:
        scale: ``[B, H, ceil(S / group), D]`` for ``axis="key"`` (per channel, groups of tokens), ``[B, H, S,
            D / group]`` for ``axis="value"`` (per token, groups of channels); fp16.
        zero: Like ``scale``.
        axis: ``"key"`` or ``"value"``.
        group: Tokens (keys) or channels (values) per scale.
        dtype: Result dtype.
    """
    c = unpack_codes(codes, bits).float()
    if axis == "key":
        s = scale.float().repeat_interleave(group, dim=2)[:, :, : c.shape[2]]
        z = zero.float().repeat_interleave(group, dim=2)[:, :, : c.shape[2]]
    else:
        s = scale.float().repeat_interleave(group, dim=3)
        z = zero.float().repeat_interleave(group, dim=3)
    return (c * s + z).to(dtype)


@dataclass
class KIVIPacked:
    """Packed K/V of one layer (a :class:`~.packed.PackedKV` of format ``"kivi"``).

    Attributes:
        bits: Bits per code.
        key_group: Tokens per key scale (per channel).
        value_group: Channels per value scale (per token).
        key_codes: ``uint8`` ``[B, Hkv, S, D * bits / 8]``.
        key_scale: fp16 ``[B, Hkv, ceil(S / key_group), D]``.
        key_zero: Like ``key_scale``.
        value_codes: ``uint8`` ``[B, Hkv, S, Dv * bits / 8]``.
        value_scale: fp16 ``[B, Hkv, S, Dv / value_group]``.
        value_zero: Like ``value_scale``.
        keys_quantized: ``int32`` ``[B]`` on the device: keys ``< keys_quantized[b]`` of row ``b`` are packed.
        values_quantized: ``int32`` ``[B]``: likewise for values.
    """

    bits: int
    key_group: int
    value_group: int
    key_codes: torch.Tensor
    key_scale: torch.Tensor
    key_zero: torch.Tensor
    value_codes: torch.Tensor
    value_scale: torch.Tensor
    value_zero: torch.Tensor
    keys_quantized: torch.Tensor
    values_quantized: torch.Tensor
    format: str = "kivi"

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
        return kivi_decode_attention(query, key, value, mask, scale, self, softcap=softcap, sinks=sinks)


def _broadcast_layout(block_n: int, block_dv: int, key_group: int, value_group: int) -> tuple[int, int]:
    """``(KEY_ROWS, VALUE_COLS)`` of :func:`_kivi_split_kernel`: key groups per block of ``block_n`` keys when the
    blocks hold whole groups or lie inside one, and value groups per padded value row when the groups tile it
    (powers of two); ``0`` selects the per-element loads."""
    key_rows = 0
    if block_n % key_group == 0:
        key_rows = block_n // key_group
    elif key_group % block_n == 0:
        key_rows = 1
    pow2 = value_group & (value_group - 1) == 0
    value_cols = block_dv // value_group if pow2 and block_dv % value_group == 0 else 0
    return key_rows, value_cols


def _check(name: str, tensor: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype) -> None:
    if tuple(tensor.shape) != shape or tensor.dtype != dtype:
        raise ConfigError(f"KIVI {name} is {tuple(tensor.shape)} {tensor.dtype}, expected {shape} {dtype}")
    if tensor.stride(-1) != 1:
        raise ConfigError(f"KIVI {name} needs a contiguous last axis")


#: Keys per tile of :func:`kivi_plan` for head dims up to 128.
_WIDE_BLOCK_N = 128
#: Keys per tile of :func:`kivi_plan` for head dims above 256.
_WIDE_HEAD_BLOCK_N = 64


def kivi_plan(
    batch: int, kv_heads: int, rows: int, keys: int, head_dim: int, device: torch.device, element_size: int = 2
) -> DecodePlan:
    """The launch shape of :func:`kivi_decode_attention` (the dense kernel under the same plan gives bit-equal
    outputs, since the merge order of the splits is the same).

    It starts from the original split-K rule of :func:`.decode.plan` (one value tile, four programs per SM) and
    changes two things, both measured on an L40 under CUDA graphs (docs/KERNEL_AUDIT.md, KV quantization):

    * Key tiles of 128 keys while such a tile fits in 32 KiB (head dim up to 128 in 16-bit). The loop then runs
      half as many iterations, each loading twice as many codes, which hides more of the load latency of the small
      packed tiles: 1.0-1.47x over 64-key tiles across batch 1-16 and 2K-32K keys, never slower by more than 2%.
    * Two programs per SM while batch and KV heads alone give at most a quarter as many programs as SMs, since a
      split's dequantization amortizes its scale loads and the merge reads half as many partials (1.1-1.4x at
      batch 1, 1.03-1.17x at batch 2-4, unchanged from batch 8 on).

    Heads wider than 256 (the global layers of Gemma 4 have 512) instead take 64-key tiles and one program per SM:
    1.45-1.65x over the 16-key tiles of the base rule at batch 1, 1-5 queries and 2K-32K keys on an A100, within 3%
    of the best of a sweep over tile shapes, warps and waves.
    """
    if head_dim > 256:
        base = plan(batch, kv_heads, rows, keys, head_dim, device, element_size=element_size, tuned=False, waves=1)
        return _with_block_n(base, keys, _WIDE_HEAD_BLOCK_N)
    base = plan(batch, kv_heads, rows, keys, head_dim, device, element_size=element_size, tuned=False)
    if batch * kv_heads * base.m_tiles * 4 <= _sm_count(device):
        base = plan(batch, kv_heads, rows, keys, head_dim, device, element_size=element_size, tuned=False, waves=2)
    if base.block_n >= _WIDE_BLOCK_N or base.block_d * element_size * _WIDE_BLOCK_N > 32768:
        return base
    return _with_block_n(base, keys, _WIDE_BLOCK_N)


def _with_block_n(base: DecodePlan, keys: int, block_n: int) -> DecodePlan:
    """``base`` with key tiles of ``block_n``, its splits rounded up to whole tiles."""
    split_len = -(-base.split_len // block_n) * block_n
    return replace(base, block_n=block_n, split_len=split_len, splits=-(-keys // split_len))


def kivi_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
    packed: KIVIPacked,
    softcap: float | None = None,
    sinks: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """:func:`~.decode.decode_attention` over a cache whose oldest entries are KIVI codes.

    Args:
        query: ``[B, Hq, Q, D]``, bf16, fp16 or fp32.
        key: Dense ``[B, Hkv, S, D]`` in the query's dtype; read only at slots ``>= packed.keys_quantized[b]``.
        value: Dense ``[B, Hkv, S, Dv]``; read only at slots ``>= packed.values_quantized[b]``.
        mask: Boolean ``[B, 1 or Hq, Q, S]`` or ``None``.
        scale: Multiplier of the scores.
        packed: The codes, scales and zero points (see :class:`KIVIPacked`).
        softcap: Optional tanh soft cap of the scaled scores.
        sinks: Optional ``[Hq]`` attention-sink logits.
        out: Optional ``[B, Q, Hq, Dv]`` output buffer.

    Returns:
        ``[B, Q, Hq, Dv]`` in the query's dtype.
    """
    if not triton_available():
        raise RuntimeError("kivi_decode_attention needs triton")
    batch, q_heads, queries, head_dim = query.shape
    _, kv_heads, keys, _ = key.shape
    head_dv = value.shape[-1]
    if q_heads % kv_heads:
        raise ConfigError(f"{q_heads} query heads are not a multiple of {kv_heads} KV heads")
    if query.dtype not in _DOT_DTYPES or key.dtype != query.dtype or value.dtype != query.dtype:
        raise ConfigError(
            f"kivi_decode_attention takes dense K/V in the query's dtype, got {key.dtype} / {value.dtype}"
        )
    bits, kg, vg = packed.bits, packed.key_group, packed.value_group
    if bits not in (2, 4, 8) or head_dim % (8 // bits) or head_dv % (8 // bits) or head_dv % vg:
        raise ConfigError(f"KIVI layout bits={bits}, value_group={vg} does not fit head dims {head_dim} / {head_dv}")
    per_byte = 8 // bits
    shape = (batch, kv_heads)
    _check("key_codes", packed.key_codes, (*shape, keys, head_dim // per_byte), torch.uint8)
    _check("key_scale", packed.key_scale, (*shape, -(-keys // kg), head_dim), torch.float16)
    _check("key_zero", packed.key_zero, (*shape, -(-keys // kg), head_dim), torch.float16)
    _check("value_codes", packed.value_codes, (*shape, keys, head_dv // per_byte), torch.uint8)
    _check("value_scale", packed.value_scale, (*shape, keys, head_dv // vg), torch.float16)
    _check("value_zero", packed.value_zero, (*shape, keys, head_dv // vg), torch.float16)
    key_layout = packed.key_zero.stride() == packed.key_scale.stride()
    if not key_layout or packed.value_zero.stride() != packed.value_scale.stride():
        raise ConfigError("KIVI zero points must share their scales' layout")
    for name in ("keys_quantized", "values_quantized"):
        counts = getattr(packed, name)
        if counts.shape != (batch,) or counts.dtype != torch.int32 or counts.device != query.device:
            raise ConfigError(f"KIVI {name} must be int32 [{batch}] on {query.device}")

    groups = q_heads // kv_heads
    rows = groups * queries
    p = kivi_plan(batch, kv_heads, rows, keys, max(head_dim, head_dv), query.device, query.element_size())
    block_dv = max(16, next_power_of_2(head_dv))
    block_d = max(16, next_power_of_2(head_dim))
    programs = batch * kv_heads * p.m_tiles
    part_o = torch.empty(programs * p.splits * p.block_m * block_dv, device=query.device, dtype=torch.float32)
    part_m = torch.empty(programs * p.splits * p.block_m, device=query.device, dtype=torch.float32)
    part_l = torch.empty_like(part_m)
    if out is None:
        out = torch.empty(batch, queries, q_heads, head_dv, device=query.device, dtype=query.dtype)
    m, mask_strides, per_head = mask_operand(mask, keys, query)
    sinks = sinks_operand(sinks, q_heads)
    num_warps = 4 if block_d <= 256 else 8
    key_rows, value_cols = _broadcast_layout(p.block_n, block_dv, kg, vg)
    kc, ks, vc, vs = packed.key_codes, packed.key_scale, packed.value_codes, packed.value_scale
    from ._decode_kernels import _DOT, _merge_kernel
    from ._kivi_kernels import _kivi_split_kernel

    with torch.cuda.device(query.device):
        _kivi_split_kernel[(programs, p.splits)](
            query, key, value, kc, ks, packed.key_zero, vc, vs, packed.value_zero,
            packed.keys_quantized, packed.values_quantized, m, part_o, part_m, part_l,
            *query.stride(), *key.stride(), *value.stride(),
            kc.stride(0), kc.stride(1), kc.stride(2),
            ks.stride(0), ks.stride(1), ks.stride(2),
            vc.stride(0), vc.stride(1), vc.stride(2),
            vs.stride(0), vs.stride(1), vs.stride(2),
            *mask_strides,
            keys, queries, rows, kv_heads, p.m_tiles, p.splits, float(scale), float(softcap or 1.0),
            GROUPS=groups, HEAD_D=head_dim, HEAD_DV=head_dv, BLOCK_M=p.block_m, BLOCK_N=p.block_n, BLOCK_D=block_d,
            BLOCK_DV=block_dv, SPLIT_LEN=p.split_len, HAS_MASK=mask is not None, MASK_PER_HEAD=per_head,
            HAS_SOFTCAP=softcap is not None, IEEE=query.dtype == torch.float32, DOT_DTYPE=_DOT[query.dtype],
            BITS=bits, KEY_GROUP=kg, VALUE_GROUP=vg, KEY_ROWS=key_rows, VALUE_COLS=value_cols, num_warps=num_warps,
            num_stages=1,
        )
        _merge_kernel[(programs, p.block_m)](
            part_o, part_m, part_l, out, part_m if sinks is None else sinks, *out.stride(),
            queries, rows, kv_heads, p.m_tiles, p.splits,
            GROUPS=groups, HEAD_DV=head_dv, BLOCK_M=p.block_m, BLOCK_DV=block_dv, HAS_SINKS=sinks is not None,
            num_warps=4,
        )
    return out
