"""FlashAttention for long queries (prefill) over a static KV buffer: the prefill path of ``"economize"``.

The split-K decode kernel (:mod:`.decode`) takes at most :data:`~.interface.MAX_QUERIES` queries per row. Longer
queries used to run on ``sdpa`` with economize's explicit boolean mask, which keeps PyTorch off its flash backend.
This module runs them on vLLM's build of FlashAttention instead (FA3 on Hopper, FA2 on Ampere and Ada, FA4 on Hopper
and Blackwell for sliding windows and head dims above 256), when the call is one FlashAttention can express exactly:

* bf16 or fp16 query, key and value of one dtype, head dim at most 256 (512 on FA4), no attention sinks;
* a mask that is, row by row, causal attention over a prefix of the buffer (the query of row ``b`` at index ``i``
  sees the slots ``0 .. o_b + i``), optionally restricted to the last ``sliding_window`` slots, the same for all
  heads. :func:`causal_offsets` checks this against the mask itself, so padding, evicted slots, trees or packed
  sequences fall back to ``sdpa``;
* not during CUDA-graph capture (the mask check reads the mask on the host);
* a sliding window only on FA4 (or FA2 where FA2 is the build's version): vLLM 0.30's FA3 local-window kernel is
  wrong on an H100 (2% of the outputs of a 300-token query at head dim 256 off by up to 0.19, in every K/V layout,
  while FA2, FA4 and ``sdpa`` agree with an fp32 reference), so a window without FA4 on Hopper falls back with
  reason ``"flash_window"``.

The buffer ``[B, Hkv, S, D]`` is handed over as one page per row through a ``[B, S, Hkv, D]`` view (no copy) with
``block_table = [[0], [1], ...]`` and ``seqused_k = o_b + Q``; FlashAttention's causal mask is aligned to the end of
the used keys, which is the mask above.

Short queries (decoding and verification, at most :data:`~.interface.MAX_QUERIES` per row) can run on
FlashAttention too, through :func:`flash_decode`. Checking the mask on the host would synchronize every step and
break CUDA graphs, so this path needs the mask's producer to declare the prefix layout instead
(:func:`declare_prefix`): economize's own chain passes (:meth:`economize.runtime.LayerContext.mask`) attach a
:class:`PrefixHint` to every mask they build, holding ``seqused_k`` as a device tensor computed from the pass's
positions. A mask modified on the way to the attention call (``kv_evict``'s keep masks, a model that slices the mask)
is a new tensor without a hint and runs on the split-K kernel. Whether FlashAttention or the split-K kernel runs a
hinted call is decided by shape (:func:`decode_prefers_flash`): on Hopper FA3 (head dim <= 256) and FA4 (head dim
512) win at larger batches and longer verification windows, while the split-K kernel wins at batch size 1.
"""

from __future__ import annotations

import functools
import os
import threading
import weakref
from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "FLASH_DECODE",
    "FLASH_MIN_BATCH",
    "MAX_HEAD_DIM",
    "PrefixHint",
    "RingPages",
    "causal_offsets",
    "declare_prefix",
    "declare_ring",
    "decode_hints_wanted",
    "decode_prefers_flash",
    "flash_decode",
    "flash_prefill",
    "flash_version",
    "prefix_hint",
    "ring_hints_wanted",
]

#: When hinted short queries run on FlashAttention (:func:`flash_decode`): ``"auto"`` (default) where
#: :func:`decode_prefers_flash` says it is faster, ``"1"`` whenever the call is expressible (tests), ``"0"`` never.
#: Debug override ``ECONOMIZE_FLASH_DECODE``, read once at import.
FLASH_DECODE = os.environ.get("ECONOMIZE_FLASH_DECODE", "auto")

#: Largest head dim of FlashAttention 2 / 3.
MAX_HEAD_DIM = 256
#: Largest head dim of FlashAttention 4 (vLLM's build, Hopper and Blackwell).
MAX_HEAD_DIM_FA4 = 512
#: FA2 reads a paged buffer only in pages of a multiple of this many slots.
_FA2_PAGE = 256


@functools.cache
def _interface() -> Any:
    try:
        from vllm.vllm_flash_attn import flash_attn_interface
    except Exception:  # vLLM is optional; any import failure means "no FlashAttention"
        return None
    return flash_attn_interface


@functools.cache
def flash_version(device_index: int) -> int | None:
    """The FlashAttention version used on a CUDA device (3 on sm90, 2 on sm80+), or ``None`` when vLLM's build has
    none for it."""
    fai = _interface()
    if fai is None:
        return None
    major = torch.cuda.get_device_capability(device_index)[0]
    version = 3 if major == 9 else 2 if major >= 8 else None
    if version is None or not fai.is_fa_version_supported(version):
        return None
    return version


def _prefill_version(index: int, dim: int, window: int | None, softcap: float | None) -> tuple[int | None, str]:
    """The FlashAttention version for a long query on device ``index``, or ``(None, reason)``.

    FA4 (Hopper and Blackwell) takes head dims up to 512 and sliding windows; FA3's local-window kernel is wrong in
    vLLM 0.30 (module docstring), so windows go to FA4, or to FA2 on GPUs whose version is FA2. FA4 has no soft cap.
    """
    version = flash_version(index)
    if version is None:
        return None, "no_flash"
    major = torch.cuda.get_device_capability(index)[0]
    fa4 = major in (9, 10) and dim <= MAX_HEAD_DIM_FA4 and _fa4_supported() and not softcap
    if dim > MAX_HEAD_DIM:
        return (4, "") if fa4 else (None, "flash_head_dim")
    if window is not None:
        return (4, "") if fa4 else (2, "") if version == 2 else (None, "flash_window")
    return version, ""


def causal_offsets(mask: torch.Tensor, keys: int, window: int | None) -> torch.Tensor | None:
    """The first query position ``o_b`` of every row if ``mask[:, 0, :, :keys]`` is causal over a prefix of the
    buffer (with an optional sliding window of ``window`` slots), else ``None``.

    Args:
        mask: Boolean ``[B, 1, Q, S >= keys]``, ``True`` = visible.
        keys: Slots of the KV buffer.
        window: The layer's sliding window (key ``j`` visible to query ``p`` iff ``p - window < j <= p``) or
            ``None``.
    """
    visible = mask[:, 0, :, :keys]
    queries = visible.shape[1]
    slot = torch.arange(keys, device=mask.device)
    first = torch.where(visible[:, 0], slot, -1).amax(-1)  # the last slot the first query sees
    position = first[:, None] + torch.arange(queries, device=mask.device)  # [B, Q]
    if int(first.min()) < 0 or int(position.max()) >= keys:
        return None
    expected = slot <= position[..., None]
    if window is not None:
        expected &= slot > position[..., None] - window
    return first if torch.equal(expected, visible) else None


#: Per thread, the mask :func:`_offsets` saw last, its tag and its offsets. A forward pass runs in one thread, and
#: concurrent decoders (async SSD runs one per GPU in its own thread) each keep their own entry.
_memo = threading.local()


def _offsets(mask: torch.Tensor, batch: int, keys: int, window: int | None) -> torch.Tensor | None:
    """:func:`causal_offsets`, checked once per mask: all layers of a forward pass receive the same mask tensor."""
    tag = (mask._version, batch, keys, window)
    last = getattr(_memo, "last", None)
    if last is None or last[0]() is not mask or last[1] != tag:
        last = (weakref.ref(mask), tag, causal_offsets(mask.expand(batch, -1, -1, -1), keys, window))
        _memo.last = last
    return last[2]


def flash_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
    softcap: float | None = None,
    window: int | None = None,
) -> tuple[torch.Tensor | None, str]:
    """Attention of long queries with FlashAttention, or ``(None, reason)`` when the call is not one it expresses.

    Args:
        query: ``[B, Hq, Q, D]``.
        key: ``[B, Hkv, S, D]``.
        value: ``[B, Hkv, S, D]``.
        mask: Boolean ``[B or 1, 1, Q, >= S]`` or ``None`` (refused: without a mask the positions are unknown).
        scale: Multiplier of the scores.
        softcap: Optional tanh soft cap.
        window: The layer's sliding window (``transformers`` passes it as ``sliding_window``).

    Returns:
        ``([B, Q, Hq, D], path)`` with ``path`` ``"flash<version>"``, or ``(None, reason)``.
    """
    if not query.is_cuda:
        return None, "cpu"
    hint = prefix_hint(mask)
    if hint is not None and hint.ring is not None:  # a ring buffer (window cache): only its hint knows the layout
        return _ring_call(query, key, value, hint, scale, softcap)
    batch, q_heads, queries, dim = query.shape
    keys = key.shape[2]
    index = query.device.index if query.device.index is not None else torch.cuda.current_device()
    version, reason = _prefill_version(index, dim, window, softcap)
    if version is None:
        return None, reason
    if query.dtype not in (torch.bfloat16, torch.float16) or key.dtype != query.dtype or value.dtype != query.dtype:
        return None, "flash_dtype"
    if value.shape[-1] != dim:
        return None, "flash_head_dim"
    if version == 2 and keys % _FA2_PAGE:
        return None, "flash_page"
    if key.stride(-1) != 1 or value.stride(-1) != 1:
        return None, "flash_strides"
    if torch.cuda.is_current_stream_capturing():
        return None, "flash_capture"
    if mask is None or mask.dtype != torch.bool or mask.dim() != 4 or mask.shape[1] != 1:
        return None, "mask_pattern"
    if mask.shape[0] not in (1, batch) or mask.shape[2] != queries or mask.shape[3] < keys:
        return None, "mask_pattern"
    first = _offsets(mask, batch, keys, window)
    if first is None:
        return None, "mask_pattern"
    fai = _interface()
    device = query.device
    q = query.transpose(1, 2).reshape(batch * queries, q_heads, dim)
    cu_q = torch.arange(0, (batch + 1) * queries, queries, device=device, dtype=torch.int32)
    used = (first + queries).to(torch.int32)
    table = torch.arange(batch, device=device, dtype=torch.int32)[:, None]
    out = fai.flash_attn_varlen_func(
        q,
        key.transpose(1, 2),
        value.transpose(1, 2),
        max_seqlen_q=queries,
        cu_seqlens_q=cu_q,
        max_seqlen_k=keys,
        seqused_k=used,
        softmax_scale=scale,
        causal=True,
        window_size=(window - 1, 0) if window is not None else (-1, -1),
        softcap=float(softcap or 0.0),
        block_table=table,
        fa_version=version,
    )
    return out.view(batch, queries, q_heads, dim), f"flash{version}"


# ---- short queries (decoding and verification) -------------------------------------------------------------------

#: Attribute of a mask tensor that holds its :class:`PrefixHint`.
_HINT = "_economize_prefix"


@dataclass(frozen=True, eq=False)
class PrefixHint:
    """A mask's producer declares that the mask is causal over a prefix of the buffer: query ``i`` of row ``b`` sees
    the slots ``j <= used_b - Q + i``, and only ``j > used_b - Q + i - window`` with a sliding window. Padding
    queries may only follow the real queries of their row (their outputs are not used).

    Every field is built when the mask is (inside a CUDA graph's capture when the pass is graphed), so the attention
    calls of all layers reuse them without host work.

    Attributes:
        used: ``seqused_k``, ``[B]`` int32 on the mask's device.
        table: Block table ``[[0], [1], ...]`` (one page per row), ``[B, 1]`` int32.
        cu_queries: ``[0, Q, 2Q, ...]``, ``[B + 1]`` int32.
        queries: Queries per row ``Q``.
        keys: Slots of the buffer (the mask's last dim).
        window: Sliding window of the mask, or ``None``.
        ring: The paged view of a ring buffer (:func:`declare_ring`); ``None`` for a buffer whose slot ``j`` holds
            position ``j``.
    """

    used: torch.Tensor
    table: torch.Tensor
    cu_queries: torch.Tensor
    queries: int
    keys: int
    window: int | None
    ring: RingPages | None = None


def declare_prefix(mask: torch.Tensor, first: torch.Tensor, window: int | None) -> torch.Tensor:
    """Attach a :class:`PrefixHint` to ``mask`` (returned unchanged).

    The caller guarantees the layout: the ``Q`` queries of row ``b`` sit at the consecutive positions
    ``first_b, first_b + 1, ...`` (padding only after the real queries), slot ``j`` of the buffer holds position
    ``j``, ``mask`` is exactly the causal (optionally sliding-window) visibility of these positions, and every row
    ends inside the buffer (``first_b + Q <= S``, padding queries included: FlashAttention aligns the queries to the
    end of ``seqused_k``). economize's decoder sizes its caches for this (the widest pass fits behind the longest
    sequence). ``seqused_k`` is still clipped to the buffer, so a violating row computes wrong outputs but never reads
    outside it.

    Args:
        mask: Boolean ``[B, 1, Q, S]``.
        first: Position of each row's first query, ``[B]`` on the mask's device.
        window: The visibility's sliding window, or ``None``.
    """
    batch, _, queries, keys = mask.shape
    device = mask.device
    used = (first.to(device=device, dtype=torch.int32) + queries).clamp_(max=keys)
    table = torch.arange(batch, device=device, dtype=torch.int32)[:, None]
    cu = torch.arange(0, (batch + 1) * queries, queries, device=device, dtype=torch.int32)
    setattr(mask, _HINT, PrefixHint(used, table, cu, queries, keys, window))
    return mask


def prefix_hint(mask: torch.Tensor | None) -> PrefixHint | None:
    """The :class:`PrefixHint` of ``mask``, if its producer declared one."""
    return None if mask is None else getattr(mask, _HINT, None)


FLASH_MIN_BATCH = 4
"""Smallest batch size whose decode calls may run on FlashAttention (:func:`decode_prefers_flash`)."""


@functools.cache
def _hints_wanted(device_index: int) -> bool:
    fai = _interface()
    if fai is None:
        return False
    major = torch.cuda.get_device_capability(device_index)[0]
    return (major == 9 and flash_version(device_index) == 3) or (major in (9, 10) and _fa4_supported())


def decode_hints_wanted(device: torch.device, batch: int) -> bool:
    """Whether masks of ``batch`` rows built for ``device`` should carry a :class:`PrefixHint`: a CUDA device with a
    FlashAttention decode path (Hopper or Blackwell, see :func:`flash_decode`), ``ECONOMIZE_FLASH_DECODE`` not
    ``"0"``, and at least :data:`FLASH_MIN_BATCH` rows (or ``ECONOMIZE_FLASH_DECODE=1``). Smaller batches never
    run on FlashAttention, so their passes skip building the hint and stay identical to the split-K-only path."""
    if FLASH_DECODE == "0" or device.type != "cuda":
        return False
    if batch < FLASH_MIN_BATCH and FLASH_DECODE != "1":
        return False
    return _hints_wanted(device.index if device.index is not None else torch.cuda.current_device())


def _decode_version(device: torch.device, dim: int, window: int | None = None) -> int | None:
    """The FlashAttention version for short queries at head dim ``dim`` on ``device``: FA3 up to 256 and FA4 up to 512
    on Hopper (sm90) and FA4 up to 512 on Blackwell (sm100); ``None`` elsewhere. A sliding window always runs on FA4
    (FA3's local-window kernel is wrong, see the module docstring). FA2 reads paged buffers only in pages of a
    multiple of 256 slots, which economize's caches (capacity plus one trash slot) never have."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    fai = _interface()
    if fai is None:
        return None
    major = torch.cuda.get_device_capability(index)[0]
    if major == 9 and dim <= MAX_HEAD_DIM and window is None and flash_version(index) == 3:
        return 3
    if major in (9, 10) and dim <= MAX_HEAD_DIM_FA4 and _fa4_supported():
        return 4
    return None


@functools.cache
def _fa4_supported() -> bool:
    fai = _interface()
    try:
        return bool(fai is not None and fai.is_fa_version_supported(4))
    except Exception:  # an older build without FA4
        return False


def decode_prefers_flash(version: int, dim: int, window: int | None, batch: int, queries: int, keys: int) -> bool:
    """Whether FlashAttention ``version`` beats the split-K kernel for one decode call of this shape (H100, fitted
    on the 288-point grid of ``scripts/bench_flash_decode.py``, docs/KERNEL_AUDIT.md, Hopper section; it picks the
    slower kernel at 2 of the 192 points with batch >= 4, both within 9%). FA3 wins every batch of 4 or more at head
    dims 128 and 256 and at head dim 64 from batch 8 or 4096 slots; FA4 (head dim 512) from batch 16 or 9 queries.
    Below :data:`FLASH_MIN_BATCH` rows of the cache the split-K kernel always runs: there the two
    are within about 1% of a Gemma-4-31B verify step, so batch-1 decoding keeps one kernel on every GPU."""
    if batch < FLASH_MIN_BATCH:
        return False
    if version == 4 and dim > MAX_HEAD_DIM:  # head dim 512 (Gemma-4 global layers): FA4 needs many query rows
        return batch >= 16 or queries >= 9
    if dim <= 64 and batch < 8:  # small heads: the split-K kernel stays ahead over short buffers
        return keys >= 4096
    return True


def flash_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    hint: PrefixHint,
    scale: float,
    softcap: float | None = None,
    *,
    force: bool = False,
) -> tuple[torch.Tensor | None, str]:
    """Attention of short queries with FlashAttention over a hinted mask (:func:`declare_prefix`), or
    ``(None, reason)`` when the call is not one it expresses or the split-K kernel is faster for its shape.

    Graph-safe: no host synchronization, every index tensor comes from the hint.

    Args:
        query: ``[B, Hq, Q, D]``.
        key: ``[B, Hkv, S, D]``.
        value: ``[B, Hkv, S, D]``.
        hint: The mask's :class:`PrefixHint`.
        scale: Multiplier of the scores.
        softcap: Optional tanh soft cap.
        force: Skip the shape policy (:func:`decode_prefers_flash`); also set by ``ECONOMIZE_FLASH_DECODE=1``.

    Returns:
        ``([B, Q, Hq, D], "flash<version>")`` or ``(None, reason)``.
    """
    if FLASH_DECODE == "0":
        return None, "flash_off"
    if not query.is_cuda:
        return None, "cpu"
    if hint.ring is not None:
        return _ring_call(query, key, value, hint, scale, softcap, decode=not (force or FLASH_DECODE == "1"))
    batch, q_heads, queries, dim = query.shape
    keys = key.shape[2]
    version = _decode_version(query.device, dim, hint.window)
    if version is None:
        return None, "no_flash"
    if query.dtype not in (torch.bfloat16, torch.float16) or key.dtype != query.dtype or value.dtype != query.dtype:
        return None, "flash_dtype"
    if value.shape[-1] != dim or key.stride(-1) != 1 or value.stride(-1) != 1:
        return None, "flash_layout"
    if hint.queries != queries or hint.keys != keys or hint.used.shape[0] != batch or key.shape[0] != batch:
        return None, "flash_hint"
    if version == 4 and softcap:
        return None, "flash_softcap"
    if not (force or FLASH_DECODE == "1") and not decode_prefers_flash(version, dim, hint.window, batch, queries, keys):
        return None, "flash_shape"
    fai = _interface()
    q = query.transpose(1, 2).reshape(batch * queries, q_heads, dim)
    out = fai.flash_attn_varlen_func(
        q,
        key.transpose(1, 2),
        value.transpose(1, 2),
        max_seqlen_q=queries,
        cu_seqlens_q=hint.cu_queries,
        max_seqlen_k=keys,
        seqused_k=hint.used,
        softmax_scale=scale,
        causal=True,
        window_size=(hint.window - 1, 0) if hint.window is not None else (-1, -1),
        softcap=float(softcap or 0.0),
        block_table=hint.table,
        fa_version=version,
    )
    return out.view(batch, queries, q_heads, dim), f"flash{version}"


# ---- ring buffers (the window cache of sliding-window layers) ----------------------------------------------------


@dataclass(frozen=True, eq=False)
class RingPages:
    """How a sliding-window layer's ring buffer reads as a paged KV cache.

    A ring of ``R = pages * page`` slots holds position ``p`` at slot ``p mod R``; the buffer has ``R + page``
    slots (the trash slot ``R`` and never-written padding up to a whole page). Row ``b`` of a pass whose last query
    sits at position ``P_b`` reads the logical keys ``L_b .. P_b`` with ``L_b = page * max(0, P_b // page - pages +
    1)``: logical page ``i`` is the ring page ``(L_b // page + i) mod pages``. FlashAttention aligns the queries to the
    end of ``seqused_k = P_b - L_b + 1`` and applies the window, so it reads exactly the slots the dense ring mask
    shows. The ring must hold every key a query of the pass sees (``R - page >= Q + window - 2``,
    :class:`~economize.runtime.RingSpec` sizes it so).

    Attributes:
        table: Block table ``[B, pages]`` int32 into the page view of the buffer (:func:`_ring_view`).
        used: ``seqused_k``, ``[B]`` int32.
        page: Slots per page.
        pages: Pages of the ring.
        window: The layers' sliding window.
        version: The FlashAttention version that runs the ring (4, else 2), or ``None`` without one.
    """

    table: torch.Tensor
    used: torch.Tensor
    page: int
    pages: int
    window: int
    version: int | None


@functools.cache
def _ring_version(device_index: int) -> int | None:
    """FA4 on Hopper and Blackwell, FA2 where FA2 is the build's version (FA3's window kernel is wrong), else
    ``None``."""
    major = torch.cuda.get_device_capability(device_index)[0]
    if major in (9, 10) and _fa4_supported():
        return 4
    return 2 if flash_version(device_index) == 2 else None


def ring_hints_wanted(device: torch.device) -> bool:
    """Whether masks over a ring buffer built for ``device`` should carry a ring :class:`PrefixHint`: a CUDA device
    whose FlashAttention can read the ring (FA4 or FA2). Without the hint, long queries over a ring run on ``sdpa``
    with the dense mask and short ones on the split-K kernel, which reads the dense mask too."""
    if device.type != "cuda" or _interface() is None:
        return False
    return _ring_version(device.index if device.index is not None else torch.cuda.current_device()) is not None


def declare_ring(mask: torch.Tensor, first: torch.Tensor, window: int, slots: int, page: int) -> torch.Tensor:
    """Attach a ring :class:`PrefixHint` to ``mask`` (returned unchanged).

    The caller guarantees what :func:`declare_prefix` asks of a chain pass (consecutive queries from ``first_b``,
    padding only after them), with the buffer a ring of ``slots`` slots (a multiple of ``page``) and ``mask`` its
    sliding-window visibility. Graph-safe: every tensor is computed on the device.

    Args:
        mask: Boolean ``[B, 1, Q, slots + page]``.
        first: Position of each row's first query, ``[B]`` on the mask's device.
        window: The layers' sliding window.
        slots: Slots ``R`` of the ring.
        page: Slots per page.
    """
    batch, _, queries, keys = mask.shape
    device = mask.device
    pages = slots // page
    last = first.to(device=device, dtype=torch.long) + (queries - 1)
    start = (last // page - (pages - 1)).clamp_min(0)  # first logical page
    used = (last - start * page + 1).to(torch.int32)
    # ring page of each logical page; _ring_call adds the row offset into the page view
    table = ((start[:, None] + torch.arange(pages, device=device)) % pages).to(torch.int32)
    cu = torch.arange(0, (batch + 1) * queries, queries, device=device, dtype=torch.int32)
    version = _ring_version(device.index if device.index is not None else torch.cuda.current_device())
    ring = RingPages(table, used, page, pages, window, version)
    setattr(mask, _HINT, PrefixHint(used, table, cu, queries, keys, window, ring))
    return mask


def _ring_call(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    hint: PrefixHint,
    scale: float,
    softcap: float | None,
    *,
    decode: bool = False,
) -> tuple[torch.Tensor | None, str]:
    """Attention over a ring buffer on FlashAttention, or ``(None, reason)`` (the dense ring mask then runs on
    ``sdpa`` or the split-K kernel). ``decode`` applies the shape policy of short queries
    (:func:`decode_prefers_flash`)."""
    ring = hint.ring
    assert ring is not None
    batch, q_heads, queries, dim = query.shape
    _, kv_heads, keys, _ = key.shape
    version = ring.version
    if version is None:
        return None, "no_flash"
    if version == 4 and softcap:
        version = 2 if flash_version(query.device.index or 0) == 2 else None
        if version is None:
            return None, "flash_softcap"
    if dim > (MAX_HEAD_DIM_FA4 if version == 4 else MAX_HEAD_DIM) or value.shape[-1] != dim:
        return None, "flash_head_dim"
    if query.dtype not in (torch.bfloat16, torch.float16) or key.dtype != query.dtype or value.dtype != query.dtype:
        return None, "flash_dtype"
    if not (key.is_contiguous() and value.is_contiguous()) or value.shape != key.shape:
        return None, "flash_strides"
    if keys % ring.page or keys < (ring.pages + 1) * ring.page or hint.queries != queries:
        return None, "flash_hint"
    if hint.used.shape[0] != batch or key.shape[0] != batch:
        return None, "flash_hint"
    if decode and not decode_prefers_flash(version, dim, ring.window, batch, queries, keys):
        return None, "flash_shape"
    fai = _interface()
    k_pages, v_pages = _ring_view(key, ring.page), _ring_view(value, ring.page)
    rows = torch.arange(batch, device=query.device, dtype=torch.int32)[:, None] * (kv_heads * keys // ring.page)
    q = query.transpose(1, 2).reshape(batch * queries, q_heads, dim)
    out = fai.flash_attn_varlen_func(
        q,
        k_pages,
        v_pages,
        max_seqlen_q=queries,
        cu_seqlens_q=hint.cu_queries,
        max_seqlen_k=ring.pages * ring.page,
        seqused_k=ring.used,
        softmax_scale=scale,
        causal=True,
        window_size=(ring.window - 1, 0),
        softcap=float(softcap or 0.0),
        block_table=ring.table + rows,
        fa_version=version,
    )
    return out.view(batch, queries, q_heads, dim), f"flash{version}_ring"


def _ring_view(buffer: torch.Tensor, page: int) -> torch.Tensor:
    """A contiguous ``[B, H, S, D]`` buffer as FlashAttention pages ``[pages, page, H, D]`` (no copy): page
    ``b * H * S / page + j`` holds the slots ``j * page ..`` of row ``b`` for every head (the head stride is
    ``S * D``). Only the pages of offsets below ``S / page`` within a row are addressed."""
    batch, heads, slots, dim = buffer.shape
    pages = (batch - 1) * heads * slots // page + slots // page
    return torch.as_strided(buffer, (pages, page, heads, dim), (page * dim, dim, slots * dim, 1))
