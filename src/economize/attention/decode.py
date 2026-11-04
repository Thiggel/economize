"""Split-K decode attention with grouped query heads, in Triton.

:func:`decode_attention` computes ``softmax(q k^T * scale) v`` for a few queries per row (one decoding step, a
verification pass over a draft chain, a draft tree) against a whole KV-cache buffer, the way flash-decoding does:
the key axis is cut into splits, each program reduces one split of one KV head for all query heads of its group,
and a second kernel merges the splits. Compared with ``scaled_dot_product_attention`` behind ``repeat_kv`` (what
``transformers`` runs whenever an explicit mask is passed) it

* reads every key and value of a KV head once for all query heads of its group (no ``repeat_kv`` copy),
* skips every block of keys the boolean mask hides from all queries of the program (slots past the current
  position, slots outside a sliding window, evicted slots), so a sliding layer over a long static buffer reads its
  window only,
* spreads a single query over all SMs through the split axis,
* accepts K/V stored in fp8 (``float8_e4m3fn`` / ``float8_e5m2``) and decodes their bits in registers (exactly,
  with integer shifts, so it needs no fp8 support of the GPU or of ``triton``: it runs on A100s too).

Rows of the mask that see no key give zeros (``scaled_dot_product_attention`` gives NaN there).

The kernel needs ``triton`` and a CUDA device; :func:`reference_attention` is the pure-torch definition the tests
compare against.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import torch

from ..errors import ConfigError

__all__ = ["DecodePlan", "decode_attention", "plan", "reference_attention", "triton_available"]

#: Query dtypes the kernels compute in (``tl.dot`` operands; fp32 in IEEE precision).
_DOT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_FP8 = tuple(getattr(torch, name) for name in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, name))


@functools.cache
def triton_available() -> bool:
    """Whether the Triton kernels can be compiled (``triton`` importable). The kernels live in private modules that
    are imported on their first launch, so ``import economize`` never imports triton; this check imports it."""
    try:
        import triton.language  # noqa: F401
    except ImportError:  # pragma: no cover - exercised only without triton
        return False
    return True


def out_of_resources() -> type[Exception]:
    """Triton's compile-time ``OutOfResources`` error (``RuntimeError`` without triton). A function, so that an
    ``except out_of_resources():`` clause imports triton only once an exception reaches it."""
    try:
        from triton.runtime.errors import OutOfResources
    except ImportError:  # pragma: no cover - without triton the kernels are never called
        return RuntimeError
    return OutOfResources


def next_power_of_2(n: int) -> int:
    """The smallest power of two ``>= n`` (``triton.next_power_of_2`` without importing triton), 1 for ``n <= 1``."""
    return 1 << max(0, n - 1).bit_length()


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
    softcap: float | None = None,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    """The definition :func:`decode_attention` implements, in fp32.

    Args:
        query: ``[B, Hq, Q, D]``.
        key: ``[B, Hkv, S, D]`` (``Hq`` a multiple of ``Hkv``).
        value: ``[B, Hkv, S, Dv]``.
        mask: Boolean ``[B, 1 or Hq, Q, S]`` (``True`` = visible) or ``None`` (every key visible).
        scale: Multiplier of the scores.
        softcap: Optional ``softcap * tanh(score / softcap)`` applied to the scaled scores.
        sinks: Optional ``[Hq]`` attention-sink logits (gpt-oss): one extra logit per head in the softmax's
            denominator, with no value.

    Returns:
        ``[B, Q, Hq, Dv]`` in the query's dtype (zeros for a query that sees no key).
    """
    groups = query.shape[1] // key.shape[1]
    k = key.float().repeat_interleave(groups, dim=1)
    v = value.float().repeat_interleave(groups, dim=1)
    scores = query.float() @ k.transpose(-1, -2) * scale
    if softcap is not None:
        scores = softcap * torch.tanh(scores / softcap)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    if sinks is not None:
        sink = sinks.float().view(1, -1, 1, 1).expand(*scores.shape[:3], 1)
        probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
        return (probs @ v).transpose(1, 2).to(query.dtype)
    probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    return (probs @ v).transpose(1, 2).to(query.dtype)


@dataclass(frozen=True)
class DecodePlan:
    """Launch shape of one call (a function of the tensor shapes only, so a CUDA graph replays it unchanged).

    Attributes:
        block_m: Query rows (head, query pairs of one KV head's group) per program.
        block_n: Keys per inner iteration.
        block_d: Head dimension padded to a power of two.
        m_tiles: Programs along the rows of one KV head's group.
        split_len: Keys per split (a multiple of ``block_n``).
        splits: Number of splits.
        block_dv: Width of the value slice one program accumulates (0 = the whole padded value dimension).
        dv_tiles: Programs along the value dimension (each recomputes the scores of its rows).
        num_warps: Warps per program of the split kernel (0 = the launcher's default for the head size).
        num_stages: Software-pipelining stages of the split kernel's key loop.
    """

    block_m: int
    block_n: int
    block_d: int
    m_tiles: int
    split_len: int
    splits: int
    block_dv: int = 0
    dv_tiles: int = 1
    num_warps: int = 0
    num_stages: int = 1


def _sm_count(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def plan(
    batch: int,
    kv_heads: int,
    rows: int,
    keys: int,
    head_dim: int,
    device: torch.device,
    *,
    element_size: int = 2,
    waves: int | None = None,
    value_dim: int | None = None,
    tuned: bool = True,
    visible: int | None = None,
) -> DecodePlan:
    """The launch shape for ``rows = groups * queries`` query rows per KV head over ``keys`` slots.

    Splits are chosen so that about ``waves`` programs per SM are launched; blocks the mask hides are skipped
    inside a program, so a sliding window over a long buffer leaves most programs with nothing to do.

    Args:
        head_dim: The larger of the key and value head dimensions (it sizes the tiles).
        value_dim: The value head dimension (default ``head_dim``); value tiles cut its padded width.
        waves: Programs per SM to aim for (default: the rule's choice).
        tuned: Use the shape dispatch measured on L40 and A100 GPUs (docs/KERNEL_AUDIT.md, decision 6). ``False``
            gives the original rule (one value tile, key tiles of at most 16 KiB, four waves), which
            :mod:`.kivi`'s kernel uses.
        visible: Keys any row can see when that is fewer than ``keys`` (a sliding window over a longer buffer).
            While the programs do not fill the GPU, the split length is chosen as if the buffer held only these
            keys and the whole buffer is cut into splits of that length, so the programs that hold the window are
            as many as for a short buffer. A larger batch reads the window with one program per row tile.
    """
    block_d = max(16, next_power_of_2(head_dim))
    block_value = max(16, next_power_of_2(value_dim or head_dim))
    sms = _sm_count(device)
    visible = keys if not tuned or visible is None else max(1, min(keys, visible))
    if tuned and element_size <= 2 and block_d == 512:
        return _wide_plan(batch, kv_heads, rows, keys, visible, block_d, block_value, sms, waves)
    block_n = min(64, max(16, 16384 // (block_d * element_size)))  # a K tile of at most 16 KiB
    cap = 64 if block_d <= 64 else 32 if block_d * element_size <= 512 else 16
    block_m = min(max(16, 1 << max(0, rows - 1).bit_length()), cap)
    m_tiles = -(-rows // block_m)
    programs = batch * kv_heads * m_tiles
    if waves is None:
        waves = 4
        if tuned and element_size <= 2:
            # Measured on L40 (docs/KERNEL_AUDIT.md, decision 6): a few rows per KV head stream K/V once and are
            # fastest with one program per SM; more rows re-read K/V per row tile and want more, shorter splits
            # once the batch and heads alone give a few dozen programs.
            waves = 1 if rows <= 32 else 2 if programs < 64 else 4
    splits, split_len = _splits(keys, block_n, waves * sms, programs, visible, sms)
    return DecodePlan(block_m, block_n, block_d, m_tiles, split_len, splits)


def _splits(
    keys: int, block_n: int, target: int, programs: int, visible: int | None = None, sms: int = 0
) -> tuple[int, int]:
    """``(splits, split_len)`` cutting ``keys`` into whole ``block_n`` blocks for about ``target`` programs over the
    ``visible`` keys (default all of them).

    Once the programs without splits fill the ``sms`` SMs, a window is better read by one program per row tile: the
    buffer is cut in two when its hidden part is at least as long as the window (that half's programs skip their
    blocks alongside the window's) and is left whole otherwise."""
    if visible and visible < keys and sms and programs >= sms:
        visible = keys if 2 * visible > keys else -(-keys // 2)
    blocks = max(1, -(-(visible or keys) // block_n))
    splits = min(blocks, max(1, -(-target // programs)))
    split_len = -(-blocks // splits) * block_n
    return -(-keys // split_len), split_len


def _wide_plan(
    batch: int,
    kv_heads: int,
    rows: int,
    keys: int,
    visible: int,
    block_d: int,
    block_value: int,
    sms: int,
    waves: int | None,
) -> DecodePlan:
    """16-bit heads of 257 to 512 (Gemma-4's global layers), measured on L40 over batches 1-64, 1-16 queries per
    sequence and 1k-8k keys (docs/KERNEL_AUDIT.md, decision 6).

    One program holds a ``[block_m, block_dv]`` fp32 accumulator next to a ``[block_n, block_d]`` key tile, so the
    tiles stay small (16 x 32 keys, four warps). Rows that fill at least two 32-row tiles exactly use 32 x 16 tiles,
    which read every key and value tile once for twice the rows. With one 16-row tile per KV head and more such
    tiles than SMs, 16-key tiles leave room for a second program per SM. While the rows of a KV head are few (one or
    a few tokens of a GQA group) and the programs do not fill the GPU, value tiles add parallelism: each program
    recomputes the scores of its rows and accumulates a slice of the value dimension. Rows that span several tiles
    re-read K/V once per tile and gain from shorter splits (two to four programs per SM)."""
    block_m, block_n = (32, 16) if rows % 32 == 0 and rows >= 64 else (16, 32)
    m_tiles = -(-rows // block_m)
    programs = batch * kv_heads * m_tiles
    if rows <= 16 and programs > sms:
        block_n = 16
    dv_tiles = 1
    if rows <= 32 and programs <= sms:
        dv_tiles = 4 if 4 * programs <= (2 * sms if rows <= 16 else 16) else 2
        while dv_tiles > 1 and block_value // dv_tiles < 128:
            dv_tiles //= 2
    if waves is None:
        waves = (2 if programs * dv_tiles < 64 else 1) if rows <= 32 else (2 if block_m == 32 else 4)
    splits, split_len = _splits(keys, block_n, waves * sms, programs * dv_tiles, visible, sms)
    return DecodePlan(
        block_m, block_n, block_d, m_tiles, split_len, splits, block_value // dv_tiles, dv_tiles, num_warps=4
    )


#: K/V storage formats the kernel decodes from raw bytes (``KV_FMT``): 0 = the query's dtype.
_KV_FMT = {name: code for code, name in ((1, "float8_e4m3fn"), (2, "float8_e5m2"))}


def decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
    softcap: float | None = None,
    out: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    launch: DecodePlan | None = None,
    window: int | None = None,
) -> torch.Tensor:
    """Attention of a few queries per row over a KV buffer (see the module docstring).

    Args:
        query: ``[B, Hq, Q, D]`` (any strides), bf16, fp16 or fp32.
        key: ``[B, Hkv, S, D]`` (any strides), the query's dtype or an fp8 dtype.
        value: ``[B, Hkv, S, Dv]``, like ``key``.
        mask: Boolean ``[B, 1 or Hq, Q, S]`` (broadcast strides allowed, ``True`` = visible) or ``None``.
        scale: Multiplier of the scores.
        softcap: Optional tanh soft cap of the scaled scores.
        out: Optional ``[B, Q, Hq, Dv]`` output buffer.
        sinks: Optional ``[Hq]`` attention-sink logits (see :func:`reference_attention`).
        launch: Launch shape to use instead of :func:`plan`'s (tuning and tests).
        window: The layer's sliding window, if the mask restricts every query to it. It only sizes the splits
            (see :func:`plan`); the mask alone decides what each query sees.

    Returns:
        ``[B, Q, Hq, Dv]`` in the query's dtype. fp32 inputs are computed in IEEE fp32 (no TF32).
    """
    if not triton_available():
        raise RuntimeError("decode_attention needs triton; use reference_attention")
    batch, q_heads, queries, head_dim = query.shape
    _, kv_heads, keys, _ = key.shape
    head_dv = value.shape[-1]
    if q_heads % kv_heads:
        raise ConfigError(f"{q_heads} query heads are not a multiple of {kv_heads} KV heads")
    groups = q_heads // kv_heads
    rows = groups * queries
    if query.dtype not in _DOT_DTYPES:
        raise ConfigError(f"decode_attention does not support {query.dtype} queries")
    kv_fmt = 0
    if key.dtype in _FP8:
        if value.dtype != key.dtype:
            raise ConfigError(f"K/V dtypes {key.dtype} / {value.dtype} differ")
        kv_fmt = _KV_FMT[str(key.dtype).removeprefix("torch.")]
        key, value = key.view(torch.uint8), value.view(torch.uint8)
    elif key.dtype != query.dtype or value.dtype != key.dtype:
        raise ConfigError(f"K/V dtype {key.dtype} / {value.dtype} does not match the query's {query.dtype}")
    p = launch or plan(
        batch, kv_heads, rows, keys, max(head_dim, head_dv), query.device, element_size=query.element_size(),
        value_dim=head_dv, visible=window + queries if window else None,
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
    ieee = query.dtype == torch.float32
    num_warps = 4 if block_d <= 256 else 8
    with torch.cuda.device(query.device):  # Triton launches on the current device; a model may sit on another one
        _launch(
            query, key, value, m, part_o, part_m, part_l, out, mask_strides, keys, queries, rows, kv_heads, p, scale,
            softcap, groups, head_dim, head_dv, block_d, block_dv, mask is not None, per_head, ieee, kv_fmt, num_warps,
            sinks,
        )
    return out


def check_value_tiles(p: DecodePlan, block_dv: int) -> None:
    """Value tiles must cover the padded value dimension exactly (overlapping tiles would corrupt the partials)."""
    if (p.block_dv or block_dv) * p.dv_tiles != block_dv:
        raise ConfigError(f"{p.dv_tiles} value tiles of {p.block_dv} do not cover the padded value width {block_dv}")


def mask_operand(mask: torch.Tensor | None, keys: int, placeholder: torch.Tensor) -> tuple[torch.Tensor, tuple, bool]:
    """The mask as the kernels read it: ``uint8`` bytes, ``(batch, head, query, key)`` strides with 0 on broadcast
    axes, and whether it differs per head. ``None`` gives ``placeholder`` (an unused pointer) and zero strides."""
    if mask is None:
        return placeholder, (0, 0, 0, 0), False
    if mask.dtype != torch.bool:
        raise ConfigError("decode_attention takes a boolean mask")
    per_head = mask.shape[1] != 1
    m = mask.view(torch.uint8)
    strides = (m.stride(0), m.stride(1) if per_head else 0, m.stride(2), m.stride(3))
    if m.shape[0] == 1:
        strides = (0, *strides[1:])
    if m.shape[2] == 1:
        strides = (strides[0], strides[1], 0, strides[3])
    if m.shape[3] != keys:
        raise ConfigError(f"mask covers {m.shape[3]} keys, the buffer {keys}")
    return m, strides, per_head


def sinks_operand(sinks: torch.Tensor | None, q_heads: int) -> torch.Tensor | None:
    """Attention-sink logits as one contiguous ``[Hq]`` vector (checked), or ``None``."""
    if sinks is None:
        return None
    if sinks.numel() != q_heads:
        raise ConfigError(f"{sinks.numel()} sink logits for {q_heads} query heads")
    return sinks.reshape(-1).contiguous()


def _launch(
    query, key, value, m, part_o, part_m, part_l, out, mask_strides, keys, queries, rows, kv_heads, p, scale,
    softcap, groups, head_dim, head_dv, block_d, block_dv, has_mask, per_head, ieee, kv_fmt, num_warps, sinks=None,
) -> None:
    from ._decode_kernels import _DOT, _merge_kernel, _split_kernel

    programs = query.shape[0] * kv_heads * p.m_tiles
    _split_kernel[(programs * p.dv_tiles, p.splits)](
        query, key, value, m, part_o, part_m, part_l,
        *query.stride(), *key.stride(), *value.stride(), *mask_strides,
        keys, queries, rows, kv_heads, p.m_tiles, p.splits, float(scale), float(softcap or 1.0),
        GROUPS=groups, HEAD_D=head_dim, HEAD_DV=head_dv, BLOCK_M=p.block_m, BLOCK_N=p.block_n, BLOCK_D=block_d,
        BLOCK_DV=p.block_dv or block_dv, SPLIT_LEN=p.split_len, HAS_MASK=has_mask, MASK_PER_HEAD=per_head,
        HAS_SOFTCAP=softcap is not None, IEEE=ieee, DOT_DTYPE=_DOT[query.dtype], KV_FMT=kv_fmt,
        DV_TILES=p.dv_tiles, PART_DV=block_dv, num_warps=p.num_warps or num_warps, num_stages=p.num_stages,
    )
    _merge_kernel[(programs, p.block_m)](
        part_o, part_m, part_l, out, part_m if sinks is None else sinks, *out.stride(),
        queries, rows, kv_heads, p.m_tiles, p.splits,
        GROUPS=groups, HEAD_DV=head_dv, BLOCK_M=p.block_m, BLOCK_DV=block_dv, HAS_SINKS=sinks is not None,
        num_warps=4,
    )


def default_scale(head_dim: int) -> float:
    """``1 / sqrt(head_dim)``, the scale ``scaled_dot_product_attention`` uses when none is given."""
    return 1.0 / math.sqrt(head_dim)
