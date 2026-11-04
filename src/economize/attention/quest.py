"""Query-aware page selection at decode time (Quest, Tang et al., ICML 2024).

Quest cuts the committed cache of every KV head into pages of ``page`` consecutive slots and keeps, per page and
channel, the smallest and the largest key. A decoding query ``q`` bounds its score against any key of page ``p`` by
``sum_d max(q_d * min_pd, q_d * max_pd)``; the ``pages`` pages with the largest bounds, together with the page that
holds the running token, are the only ones attention reads. The reference selects per query head of a multi-head
model. Under grouped-query attention all query heads of a KV head share its pages here, which are chosen by the sum
of the heads' bounds over the group (for multi-head attention the two coincide):

``score[b, h, p] = sum_d qsum_d * (min_pd + max_pd) / 2 + qabs_d * (max_pd - min_pd) / 2``

with ``qsum`` and ``qabs`` the sums of ``q`` and ``|q|`` over the group's heads and the pass's queries.

Row ``b``'s committed pages are those below ``done[b]``; slots from ``done[b] * page`` on (the page of the running
token, draft tokens and the slots the pass writes) are read densely. A row whose committed pages number at most
``pages`` reads every one of them, so that short rows are attended exactly, as in the reference.

:class:`QuestPages` is the :class:`~.packed.PackedKV` of format ``"quest"`` that the ``quest`` technique attaches to
the keys of a decoding step. :func:`quest_decode_attention` is the split-K kernel of :mod:`.decode` over a virtual
key axis: the first ``pages * page`` virtual keys are the selected pages, in slot order, and the rest is the dense
tail, so the kernel reads ``pages * page + keys - done[b] * page`` slots per row instead of ``keys``. Attention
functions without the kernel apply :meth:`QuestPages.restrict_mask`, which hides the unselected pages in the mask:
both compute the same attention up to the summation order of the softmax.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from ..errors import ConfigError
from .decode import (
    _DOT_DTYPES,
    _FP8,
    _KV_FMT,
    DecodePlan,
    check_value_tiles,
    decode_attention,
    mask_operand,
    next_power_of_2,
    plan,
    reference_attention,
    sinks_operand,
    triton_available,
)

__all__ = ["QuestPages", "page_bounds", "quest_decode_attention", "quest_reference", "select_pages"]


def page_bounds(keys: torch.Tensor, first: int, last: int, page: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Channel-wise smallest and largest key of pages ``first .. last - 1`` of ``keys`` ``[B, H, S, D]``.

    Returns:
        ``(kmin, kmax)``, each ``[B, H, last - first, D]``: fp8 keys in bf16 (which holds them exactly), fp32 keys
        in fp32 and 16-bit keys in their own dtype.
    """
    block = keys[:, :, first * page : last * page]
    if block.dtype in _FP8:
        block = block.to(torch.bfloat16)
    block = block.unflatten(2, (last - first, page))
    return block.amin(dim=3), block.amax(dim=3)


def bounds_dtype(key_dtype: torch.dtype) -> torch.dtype:
    """The dtype :func:`page_bounds` returns for keys of ``key_dtype``."""
    return torch.bfloat16 if key_dtype in _FP8 else key_dtype


def select_pages(
    query: torch.Tensor, kmin: torch.Tensor, kmax: torch.Tensor, done: torch.Tensor, max_done: int, pages: int
) -> torch.Tensor:
    """The committed pages every row and KV head reads (see the module docstring).

    Args:
        query: ``[B, Hq, Q, D]``.
        kmin: ``[B, Hkv, P, D]`` per-page smallest keys (``P >= max_done``).
        kmax: ``[B, Hkv, P, D]`` per-page largest keys.
        done: ``int`` ``[B]`` committed pages of every row.
        max_done: ``max(done)``, on the host.
        pages: Pages to select per row and KV head.

    Returns:
        ``int32`` ``[B, Hkv, pages]``: the selected page indices in increasing order, then ``-1``. A row with at most
        ``pages`` committed pages selects all of them.
    """
    batch, kv_heads = kmin.shape[:2]
    if max_done <= 0 or pages <= 0:
        return torch.full((batch, kv_heads, max(pages, 0)), -1, dtype=torch.int32, device=kmin.device)
    q = query.float().unflatten(1, (kv_heads, -1))  # [B, Hkv, G, Q, D]
    qsum = q.sum(dim=(2, 3))
    qabs = q.abs().sum(dim=(2, 3))
    lo = kmin[:, :, :max_done].float()
    hi = kmax[:, :, :max_done].float()
    score = torch.einsum("bhd,bhpd->bhp", qsum, (lo + hi) * 0.5) + torch.einsum("bhd,bhpd->bhp", qabs, (hi - lo) * 0.5)
    done = done.to(device=kmin.device, dtype=torch.long).view(batch, 1, 1)
    committed = torch.arange(max_done, device=kmin.device).view(1, 1, -1) < done
    score = score.masked_fill(~committed, float("-inf"))
    count = min(pages, max_done)
    index = score.topk(count, dim=-1).indices
    index = torch.where(index < done, index, max_done).sort(dim=-1).values  # uncommitted picks sort last
    index = torch.where(index == max_done, -1, index).to(torch.int32)
    if count < pages:
        index = torch.cat([index, index.new_full((batch, kv_heads, pages - count), -1)], dim=-1)
    return index.contiguous()


@dataclass
class QuestPages:
    """Page bounds of one layer's committed cache (a :class:`~.packed.PackedKV` of format ``"quest"``).

    Attributes:
        kmin: ``[B, Hkv, P, D]`` channel-wise smallest key of every committed page (rows below ``done[b]`` valid).
        kmax: ``[B, Hkv, P, D]`` the largest keys.
        done: ``int32`` ``[B]`` committed pages of every row; slots from ``done[b] * page`` on are read densely.
        min_done: ``min(done)``, on the host.
        max_done: ``max(done)``, on the host.
        page: Slots per page (a power of two, at least 16).
        pages: Committed pages every row and KV head reads (the reference's page budget minus the running page).
        max_queries: Longest query (tokens per row) that is restricted to the pages; longer ones see every key.
        on_use: Called with ``layer`` when an attention function reads the pages (the cache checks that every
            attached layer was read).
        layer: The cache layer.
    """

    kmin: torch.Tensor
    kmax: torch.Tensor
    done: torch.Tensor
    min_done: int
    max_done: int
    page: int
    pages: int
    max_queries: int = 1
    on_use: Callable[[int], None] | None = None
    layer: int = 0
    format: str = "quest"

    def select(self, query: torch.Tensor) -> torch.Tensor:
        """``int32`` ``[B, Hkv, pages]`` selected pages for ``query`` (see :func:`select_pages`)."""
        return select_pages(query, self.kmin, self.kmax, self.done, self.max_done, self.pages)

    def keep(self, query: torch.Tensor, slots: int) -> torch.Tensor:
        """Boolean ``[B, Hkv, slots]``: the slots ``query`` reads (selected pages and the dense tail)."""
        index = self.select(query).long()
        batch, kv_heads, _ = index.shape
        total = -(-slots // self.page)
        chosen = torch.zeros(batch, kv_heads, total + 1, dtype=torch.bool, device=index.device)
        chosen.scatter_(-1, torch.where(index < 0, total, index), True)
        keep = chosen[..., :total].repeat_interleave(self.page, dim=-1)[..., :slots]
        tail = (self.done.to(device=index.device, dtype=torch.long) * self.page).view(batch, 1, 1)
        return keep | (torch.arange(slots, device=index.device).view(1, 1, -1) >= tail)

    def _used(self) -> None:
        if self.on_use is not None:
            self.on_use(self.layer)

    def restrict_mask(
        self, query: torch.Tensor, key: torch.Tensor, mask: torch.Tensor | None, is_causal: bool = False
    ) -> torch.Tensor | None:
        """``mask`` with the unselected pages hidden, for attention functions that read the dense buffers.

        Args:
            query: ``[B, Hq, Q, D]``.
            key: The dense key buffer ``[B, Hkv, S, D]``.
            mask: Boolean (``True`` = visible) or additive float mask broadcastable to ``[B, Hq, Q, S]``, or ``None``.
            is_causal: Whether a ``None`` mask stands for a causal one (aligned top-left, as in ``sdpa``).

        Returns:
            A ``[B, Hq, Q, S]`` mask of ``mask``'s kind (boolean for ``None``), or ``mask`` itself when the query is
            longer than :attr:`max_queries`.
        """
        batch, q_heads, queries, _ = query.shape
        slots = key.shape[2]
        if queries > self.max_queries:
            return mask
        self._used()
        keep = self.keep(query, slots).repeat_interleave(q_heads // key.shape[1], dim=1)[:, :, None, :]
        if mask is None:
            mask = torch.ones(queries, slots, dtype=torch.bool, device=query.device)
            if is_causal:
                mask = mask.tril()
            return mask.expand(batch, q_heads, queries, slots) & keep
        mask = mask[..., :slots].expand(batch, q_heads, queries, slots)
        if mask.dtype == torch.bool:
            return mask & keep
        return mask.masked_fill(~keep, torch.finfo(mask.dtype).min)

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
        if query.shape[2] > self.max_queries:
            return decode_attention(query, key, value, mask, scale, softcap=softcap, sinks=sinks)
        self._used()
        return quest_decode_attention(query, key, value, mask, scale, self, softcap=softcap, sinks=sinks)


def quest_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
    pages: QuestPages,
    softcap: float | None = None,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    """What :func:`quest_decode_attention` computes, in fp32 over the dense buffers (``[B, Q, Hq, Dv]``)."""
    restricted = pages.restrict_mask(query, key, mask)
    return reference_attention(query, key, value, restricted, scale, softcap=softcap, sinks=sinks)


def _check(pages: QuestPages, query: torch.Tensor, key: torch.Tensor) -> None:
    batch, kv_heads, keys, head_dim = key.shape
    if pages.page < 16 or pages.page & (pages.page - 1):
        raise ConfigError(f"quest pages must hold a power of two of at least 16 slots, got {pages.page}")
    shape = (batch, kv_heads, pages.kmin.shape[2], head_dim)
    for name in ("kmin", "kmax"):
        tensor = getattr(pages, name)
        if tuple(tensor.shape) != shape or tensor.device != query.device:
            raise ConfigError(f"quest {name} is {tuple(tensor.shape)} on {tensor.device}, expected {shape}")
    if pages.kmin.shape[2] < pages.max_done or pages.max_done * pages.page > keys:
        raise ConfigError(f"quest bounds of {pages.kmin.shape[2]} pages do not cover {pages.max_done} committed ones")
    if tuple(pages.done.shape) != (batch,) or pages.done.dtype != torch.int32 or pages.done.device != query.device:
        raise ConfigError(f"quest done is {tuple(pages.done.shape)} {pages.done.dtype}, expected ({batch},) int32")
    if not 0 <= pages.min_done <= pages.max_done:
        raise ConfigError(f"quest done range {pages.min_done}..{pages.max_done} is empty")


def quest_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
    pages: QuestPages,
    softcap: float | None = None,
    sinks: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    launch: DecodePlan | None = None,
    index: torch.Tensor | None = None,
) -> torch.Tensor:
    """:func:`~.decode.decode_attention` over the selected pages and the dense tail (see the module docstring).

    Args:
        query: ``[B, Hq, Q, D]``, bf16, fp16 or fp32.
        key: Dense ``[B, Hkv, S, D]`` in the query's dtype or an fp8 dtype.
        value: Dense ``[B, Hkv, S, Dv]``, like ``key``.
        mask: Boolean ``[B, 1 or Hq, Q, S]`` over the slots, or ``None``.
        scale: Multiplier of the scores.
        pages: The page bounds (see :class:`QuestPages`).
        softcap: Optional tanh soft cap of the scaled scores.
        sinks: Optional ``[Hq]`` attention-sink logits.
        out: Optional ``[B, Q, Hq, Dv]`` output buffer.
        launch: Launch shape to use instead of :func:`~.decode.plan`'s (tuning and tests); its key tile must divide
            the page and its split length.
        index: Selected pages (``int32`` ``[B, Hkv, pages]``) to use instead of :meth:`QuestPages.select`'s.

    Returns:
        ``[B, Q, Hq, Dv]`` in the query's dtype.
    """
    if not triton_available():
        raise RuntimeError("quest_decode_attention needs triton; use quest_reference")
    batch, q_heads, queries, head_dim = query.shape
    _, kv_heads, keys, _ = key.shape
    head_dv = value.shape[-1]
    if q_heads % kv_heads:
        raise ConfigError(f"{q_heads} query heads are not a multiple of {kv_heads} KV heads")
    if query.dtype not in _DOT_DTYPES:
        raise ConfigError(f"quest_decode_attention does not support {query.dtype} queries")
    _check(pages, query, key)
    if index is None:
        index = pages.select(query)
    if tuple(index.shape) != (batch, kv_heads, pages.pages) or index.dtype != torch.int32:
        raise ConfigError(f"quest index is {tuple(index.shape)} {index.dtype}, expected int32 {pages.pages} pages")
    index = index.contiguous()
    kv_fmt = 0
    if key.dtype in _FP8:
        if value.dtype != key.dtype:
            raise ConfigError(f"K/V dtypes {key.dtype} / {value.dtype} differ")
        kv_fmt = _KV_FMT[str(key.dtype).removeprefix("torch.")]
        key, value = key.view(torch.uint8), value.view(torch.uint8)
    elif key.dtype != query.dtype or value.dtype != key.dtype:
        raise ConfigError(f"K/V dtype {key.dtype} / {value.dtype} does not match the query's {query.dtype}")

    groups = q_heads // kv_heads
    rows = groups * queries
    sel_len = pages.pages * pages.page
    virtual = max(1, sel_len + keys - pages.min_done * pages.page)
    if launch is None:
        p = plan(
            batch, kv_heads, rows, virtual, max(head_dim, head_dv), query.device, element_size=query.element_size(),
            value_dim=head_dv,
        )
        p = dataclasses.replace(p, block_n=min(p.block_n, pages.page))  # powers of two: it divides page and split
    else:
        p = launch
    if pages.page % p.block_n or p.split_len % p.block_n:
        raise ConfigError(f"key tile {p.block_n} must divide the page {pages.page} and the split {p.split_len}")
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
    tail = (pages.done * pages.page).to(torch.int32)
    num_warps = p.num_warps or (4 if block_d <= 256 else 8)
    from ._decode_kernels import _DOT, _merge_kernel
    from ._quest_kernels import _quest_split_kernel

    with torch.cuda.device(query.device) if query.is_cuda else nullcontext():
        _quest_split_kernel[(programs * p.dv_tiles, p.splits)](
            query, key, value, index, tail, m, part_o, part_m, part_l,
            *query.stride(), *key.stride(), *value.stride(),
            index.stride(0), index.stride(1),
            *mask_strides,
            keys, sel_len, queries, rows, kv_heads, p.m_tiles, p.splits, float(scale), float(softcap or 1.0),
            GROUPS=groups, HEAD_D=head_dim, HEAD_DV=head_dv, BLOCK_M=p.block_m, BLOCK_N=p.block_n, BLOCK_D=block_d,
            BLOCK_DV=p.block_dv or block_dv, SPLIT_LEN=p.split_len, PAGE=pages.page, HAS_MASK=mask is not None,
            MASK_PER_HEAD=per_head, HAS_SOFTCAP=softcap is not None, IEEE=query.dtype == torch.float32,
            DOT_DTYPE=_DOT[query.dtype], KV_FMT=kv_fmt, DV_TILES=p.dv_tiles, PART_DV=block_dv,
            num_warps=num_warps, num_stages=p.num_stages,
        )
        _merge_kernel[(programs, p.block_m)](
            part_o, part_m, part_l, out, part_m if sinks is None else sinks, *out.stride(),
            queries, rows, kv_heads, p.m_tiles, p.splits,
            GROUPS=groups, HEAD_DV=head_dv, BLOCK_M=p.block_m, BLOCK_DV=block_dv, HAS_SINKS=sinks is not None,
            num_warps=4,
        )
    return out
