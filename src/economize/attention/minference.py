"""Sparse attention over a prompt (MInference, Jiang et al., NeurIPS 2024).

MInference gives every attention head of a long-context model one of three static patterns and fits the pattern's
indices to each prompt at prefill time, from the prompt's own queries and keys:

``"a_shape"`` (the paper's A-shape, the reference's ``stream_llm``)
    the first ``vertical`` tokens (attention sinks) and a local window of ``slash`` tokens; nothing is estimated.
``"vertical_slash"``
    the ``vertical`` key columns and the ``slash`` diagonals (offsets ``i - j``) that receive the most attention
    from the last 64 queries of the prompt, with the first 30 columns and the 100 diagonals nearest the main one
    always kept.
``"block_sparse"``
    per block of 64 queries, the ``top_k`` blocks of 64 keys whose mean-pooled key has the largest product with the
    block's mean-pooled query.

The estimate follows the reference implementation (``minference`` 0.1.6.0, MIT): its sizes ``min(Q, max(v, 30))``
and ``min(Q, max(s, 50))``, its forced columns and diagonals, and its conversion of vertical and slash lines into
key ranges and columns per block of 64 queries (``convert_vertical_slash_indexes``), which every pattern shares here
(:class:`SparseIndex`). Scores are estimated in fp32 with the layer's own score multiplier (the reference uses the
model dtype and ``1 / sqrt(head_dim)``, which differ only for models with a custom scale) and without soft-capping.

:func:`sparse_prefill_attention` is a Triton kernel that reads only the indexed keys: per tile of queries it walks
the block row's key ranges and then gathers its columns, under the causal mask and the per-head window of the
A-shape pattern. :func:`mask_from_index` expands an index into the boolean mask of the same attention, which the
fallbacks (CPU, a score tap) use; both compute the same attention up to the summation order of the softmax.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from ..errors import ConfigError
from .decode import _DOT_DTYPES, _FP8, _KV_FMT, next_power_of_2, reference_attention, triton_available

__all__ = [
    "BLOCK",
    "PATTERNS",
    "HeadPattern",
    "MInferencePrefill",
    "SparseIndex",
    "a_shape_index",
    "block_sparse_index",
    "convert_vertical_slash",
    "load_patterns",
    "mask_from_index",
    "minference_index",
    "sparse_prefill_attention",
    "sparse_prefill_reference",
    "vertical_slash_index",
    "vertical_slash_lines",
    "vertical_slash_scores",
]

#: Queries per block row of the index and keys per block of the block-sparse pattern (64 in the reference).
BLOCK = 64
#: Last queries of the prompt whose attention estimates the vertical and slash lines.
LAST_Q = 64
#: Key columns the vertical-slash pattern always keeps (the first ones: attention sinks).
FORCED_COLUMNS = 30
#: Diagonals nearest the main one (offsets ``0 .. 99``) the vertical-slash pattern always keeps.
FORCED_DIAGONALS = 100
#: Smallest numbers of vertical and slash lines (the reference's floors).
MIN_VERTICAL = 30
MIN_SLASH = 50
#: The pattern names of :class:`HeadPattern` and the reference's names they read from a pattern file.
PATTERNS = {"vertical_slash": "vertical_and_slash", "block_sparse": "block_sparse", "a_shape": "stream_llm"}
_BIG = 2**30  # padding of index lists: beyond every key and block end


@dataclass(frozen=True)
class HeadPattern:
    """The pattern of one query head.

    Attributes:
        kind: ``"vertical_slash"``, ``"block_sparse"`` or ``"a_shape"`` (see the module docstring).
        vertical: Vertical lines (``"vertical_slash"``) or attention sinks (``"a_shape"``).
        slash: Slash lines (``"vertical_slash"``) or the local window (``"a_shape"``).
    """

    kind: str = "vertical_slash"
    vertical: int = 1000
    slash: int = 6096

    def __post_init__(self) -> None:
        if self.kind not in PATTERNS:
            raise ConfigError(f"MInference pattern must be one of {sorted(PATTERNS)}, got {self.kind!r}")
        if self.vertical < 0 or self.slash < 1:
            raise ConfigError(f"MInference pattern {self.kind} needs vertical >= 0 and slash >= 1")


@dataclass(frozen=True)
class SparseIndex:
    """The keys every query of a prompt of ``queries`` tokens attends to, per row and query head.

    Query ``i`` (in block row ``m = i // BLOCK``) of row ``b`` and head ``h`` sees key ``j`` iff ``j <= i``, ``j`` lies
    in one of the row's ``range_count[b, h, m]`` key ranges ``[range_start, range_end)`` or among its
    ``column_count[b, h, m]`` columns ``column_index``, and, when ``window[h] > 0``, ``i - j < window[h]`` or
    ``j < sinks[h]``.

    Attributes:
        range_start: ``int32`` ``[B, H, M, R]``.
        range_end: ``int32`` ``[B, H, M, R]``.
        range_count: ``int32`` ``[B, H, M]``.
        column_index: ``int32`` ``[B, H, M, C]``.
        column_count: ``int32`` ``[B, H, M]``.
        window: ``int32`` ``[H]``, 0 for no window.
        sinks: ``int32`` ``[H]``, the keys a windowed head also sees.
        queries: Prompt length ``Q``.
    """

    range_start: torch.Tensor
    range_end: torch.Tensor
    range_count: torch.Tensor
    column_index: torch.Tensor
    column_count: torch.Tensor
    window: torch.Tensor
    sinks: torch.Tensor
    queries: int

    @property
    def blocks(self) -> int:
        """Block rows ``M = ceil(Q / BLOCK)``."""
        return self.range_start.shape[2]

    def density(self) -> float:
        """Fraction of the causal (query, key) pairs the index keeps, over all rows and heads (builds the mask)."""
        q = self.queries
        return float(mask_from_index(self).sum()) / (self.range_start.shape[0] * self.window.numel() * q * (q + 1) / 2)


# ---- vertical and slash lines -----------------------------------------------------------------------------------
def _kv_heads(key: torch.Tensor, heads: torch.Tensor, groups: int) -> torch.Tensor:
    return key.index_select(1, torch.div(heads, groups, rounding_mode="floor"))


def vertical_slash_scores(
    query: torch.Tensor, key: torch.Tensor, scale: float, last_q: int = LAST_Q
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention mass of every key column and every diagonal under the prompt's last ``last_q`` queries.

    Args:
        query: ``[B, Hq, Q, D]``.
        key: ``[B, Hkv, Q, D]``.
        scale: Score multiplier.
        last_q: Queries of the estimate (the last ones of the prompt).

    Returns:
        ``(vertical, slash)``, each fp32 ``[B, Hq, Q]``: ``vertical[..., j]`` sums the softmax weights of key ``j``,
        ``slash[..., t]`` those of the diagonal at offset ``i - j = t``.
    """
    batch, heads, queries, _ = query.shape
    groups = heads // key.shape[1]
    rows = min(last_q, queries)
    pos = torch.arange(queries - rows, queries, device=query.device)
    col = torch.arange(queries, device=query.device)
    causal = col[None, :] <= pos[:, None]  # [L, Q]
    offset = (pos[:, None] - col[None, :]).clamp(min=0)
    vertical = torch.empty(batch, heads, queries, device=query.device, dtype=torch.float32)
    slash = torch.empty_like(vertical)
    step = max(1, (1 << 28) // max(1, batch * rows * queries * 4))  # heads per chunk: about 256 MB of scores
    for h0 in range(0, heads, step):
        h = torch.arange(h0, min(heads, h0 + step), device=query.device)
        q = query[:, h0 : h0 + len(h), queries - rows :].float()
        k = _kv_heads(key, h, groups).float()
        weights = (q @ k.transpose(-1, -2) * scale).masked_fill(~causal, float("-inf")).softmax(-1)
        vertical[:, h0 : h0 + len(h)] = weights.sum(-2)
        flat = weights.flatten(-2)
        index = offset.flatten().expand_as(flat)
        slash[:, h0 : h0 + len(h)] = torch.zeros_like(weights[..., 0, :]).scatter_add_(-1, index, flat)
    return vertical, slash


def vertical_slash_lines(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: float,
    vertical: torch.Tensor | int,
    slash: torch.Tensor | int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The vertical and slash lines of every head (the reference's estimate).

    Args:
        query: ``[B, Hq, Q, D]``.
        key: ``[B, Hkv, Q, D]``.
        scale: Score multiplier.
        vertical: Vertical lines per head (``[Hq]`` or one number), floored at :data:`MIN_VERTICAL`.
        slash: Slash lines per head, floored at :data:`MIN_SLASH`.

    Returns:
        ``(columns, diagonals)``: ``int32`` ``[B, Hq, V]`` key columns in ascending order and ``[B, Hq, S]``
        diagonal offsets in descending order; a head with fewer lines than the longest list is padded with a value
        beyond every key (at the end of ``columns``, at the start of ``diagonals``).
    """
    heads, queries = query.shape[1], query.shape[2]
    device = query.device
    v_size = torch.as_tensor(vertical, device=device).expand(heads).clamp(min=MIN_VERTICAL).clamp(max=queries)
    s_size = torch.as_tensor(slash, device=device).expand(heads).clamp(min=MIN_SLASH).clamp(max=queries)
    v_score, s_score = vertical_slash_scores(query, key, scale)
    v_score[..., :FORCED_COLUMNS] = float("inf")
    s_score[..., :FORCED_DIAGONALS] = float("inf")

    def top(score: torch.Tensor, size: torch.Tensor, descending: bool) -> torch.Tensor:
        width = int(size.max())
        index = score.topk(width, dim=-1).indices
        keep = torch.arange(width, device=device)[None, :] < size[:, None]
        return torch.where(keep, index, _BIG).sort(-1, descending=descending).values.to(torch.int32)

    return top(v_score, v_size, False), top(s_score, s_size, True)


def convert_vertical_slash(
    columns: torch.Tensor, diagonals: torch.Tensor, queries: int, kernel: bool | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Key ranges and columns per block row from vertical and slash lines (``convert_vertical_slash_indexes``).

    Block row ``m`` covers queries ``64 m .. 64 m + 63``. Each diagonal ``t < 64 (m + 1)`` becomes the 64 keys up to
    ``e = max(64 (m + 1) - t, 64)``; in descending order of ``t`` they merge into ranges (a diagonal whose block
    starts within 64 keys of the current range end extends that range by 64 keys). The columns are the vertical
    lines below the last range end that no range covers.

    Args:
        columns: ``int32`` ``[B, H, V]``, ascending (see :func:`vertical_slash_lines`).
        diagonals: ``int32`` ``[B, H, S]``, descending.
        queries: Prompt length.
        kernel: Use the Triton kernel (default: on CUDA when triton is available).

    Returns:
        ``(range_start, range_end, range_count, column_index, column_count)`` as in :class:`SparseIndex`.
    """
    if kernel is None:
        kernel = columns.is_cuda and triton_available()
    if kernel:
        return _convert_kernel(columns, diagonals, queries)
    batch, heads, n_diag = diagonals.shape
    device = columns.device
    blocks = math.ceil(queries / BLOCK)
    width = max(1, min(n_diag, blocks))
    end = (torch.arange(blocks, device=device) + 1) * BLOCK  # [M]
    starts = torch.zeros(batch, heads, blocks, width, dtype=torch.long, device=device)
    ends = torch.zeros_like(starts)
    count = torch.zeros(batch, heads, blocks, dtype=torch.long, device=device)
    cur_s = torch.zeros(batch, heads, blocks, dtype=torch.long, device=device)
    cur_e = torch.zeros_like(cur_s)
    started = torch.zeros(batch, heads, blocks, dtype=torch.bool, device=device)

    def store(where: torch.Tensor) -> None:
        slot = count.clamp(max=width - 1).unsqueeze(-1)
        starts.scatter_(-1, slot, torch.where(where, cur_s, starts.gather(-1, slot).squeeze(-1)).unsqueeze(-1))
        ends.scatter_(-1, slot, torch.where(where, cur_e, ends.gather(-1, slot).squeeze(-1)).unsqueeze(-1))
        count.add_(where.long())

    for k in range(n_diag):
        d = diagonals[:, :, k : k + 1].long()
        valid = d < end
        e = (end - d).clamp(min=BLOCK)
        new = valid & ~started
        save = valid & started & (e > cur_e + BLOCK)
        extend = valid & started & ~save & (e > cur_e)
        store(save)
        reset = new | save
        cur_s = torch.where(reset, e - BLOCK, cur_s)
        cur_e = torch.where(reset, e, torch.where(extend, cur_e + BLOCK, cur_e))
        started |= valid
    store(started)
    last_end = torch.where(started, cur_e, 0)

    # columns: below the last range end and in no range (ranges ascend, so a binary search finds the candidate)
    padded = torch.where(torch.arange(width, device=device) < count.unsqueeze(-1), starts, _BIG)
    v = columns.long().unsqueeze(2).expand(batch, heads, blocks, columns.shape[-1]).contiguous()
    slot = torch.searchsorted(padded, v, right=True) - 1
    inside = (slot >= 0) & (v < ends.gather(-1, slot.clamp(min=0)))
    keep = ~inside & (v < last_end.unsqueeze(-1))
    column_index = torch.where(keep, v, _BIG).sort(-1).values
    return (
        starts.to(torch.int32),
        ends.to(torch.int32),
        count.to(torch.int32),
        column_index.to(torch.int32),
        keep.sum(-1).to(torch.int32),
    )


def _convert_kernel(
    columns: torch.Tensor, diagonals: torch.Tensor, queries: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from ._minference_kernels import _vs_convert_kernel

    batch, heads, n_diag = diagonals.shape
    n_col = columns.shape[-1]
    blocks = math.ceil(queries / BLOCK)
    width = max(1, min(n_diag, blocks))
    device = columns.device
    starts = torch.zeros(batch, heads, blocks, width, dtype=torch.int32, device=device)
    ends = torch.zeros_like(starts)
    count = torch.zeros(batch, heads, blocks, dtype=torch.int32, device=device)
    column_index = torch.full((batch, heads, blocks, max(1, n_col)), _BIG, dtype=torch.int32, device=device)
    column_count = torch.zeros_like(count)
    with torch.cuda.device(device) if columns.is_cuda else nullcontext():
        _vs_convert_kernel[(batch * heads * blocks,)](
            columns.to(torch.int32).contiguous(), diagonals.to(torch.int32).contiguous(), starts, ends, count,
            column_index, column_count, n_col, n_diag, blocks, width, max(1, n_col),
            BLOCK=BLOCK, BLOCK_V=min(128, max(16, next_power_of_2(max(1, n_col)))), num_warps=1,
        )
    return starts, ends, count, column_index, column_count


def vertical_slash_index(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: float,
    vertical: torch.Tensor | int,
    slash: torch.Tensor | int,
    kernel: bool | None = None,
) -> SparseIndex:
    """The :class:`SparseIndex` of the vertical-slash pattern on every head (see :func:`vertical_slash_lines`)."""
    columns, diagonals = vertical_slash_lines(query, key, scale, vertical, slash)
    parts = convert_vertical_slash(columns, diagonals, query.shape[2], kernel=kernel)
    zeros = torch.zeros(query.shape[1], dtype=torch.int32, device=query.device)
    return SparseIndex(*parts, window=zeros, sinks=zeros, queries=query.shape[2])


def block_sparse_index(query: torch.Tensor, key: torch.Tensor, top_k: int) -> SparseIndex:
    """The :class:`SparseIndex` of the block-sparse pattern on every head.

    As in the reference, queries and keys are padded with zeros by ``64 - Q % 64`` positions (a whole block when
    ``Q`` is a multiple of 64), mean-pooled per block of 64 (the padding counts in the mean) and scored without the
    score multiplier; block row ``m`` reads the ``min(m + 1, top_k)`` best key blocks at or before it.
    """
    batch, heads, queries, dim = query.shape
    groups = heads // key.shape[1]
    pad = BLOCK - queries % BLOCK
    pooled = (queries + pad) // BLOCK
    k = _kv_heads(key, torch.arange(heads, device=key.device), groups)
    pq = F.pad(query.float(), (0, 0, 0, pad)).view(batch, heads, pooled, BLOCK, dim).mean(-2)
    pk = F.pad(k.float(), (0, 0, 0, pad)).view(batch, heads, pooled, BLOCK, dim).mean(-2)
    score = pq @ pk.transpose(-1, -2)
    future = torch.ones(pooled, pooled, dtype=torch.bool, device=query.device).triu(1)
    score = score.masked_fill(future, float("-inf"))
    width = min(top_k, pooled)
    chosen = score.topk(width, dim=-1).indices.sort(-1).values  # [B, H, P, k]
    blocks = math.ceil(queries / BLOCK)
    chosen = chosen[:, :, :blocks].to(torch.int32)
    rows = torch.arange(blocks, device=query.device, dtype=torch.int32)
    count = torch.clamp(rows + 1, max=width).expand(batch, heads, blocks).contiguous()
    zeros = torch.zeros(heads, dtype=torch.int32, device=query.device)
    no_columns = torch.zeros(batch, heads, blocks, 1, dtype=torch.int32, device=query.device)
    return SparseIndex(
        chosen * BLOCK, chosen * BLOCK + BLOCK, count, no_columns, no_columns[..., 0].clone(), zeros, zeros, queries
    )


def a_shape_index(
    batch: int, queries: int, sinks: torch.Tensor | int, local: torch.Tensor | int, heads: int, device: torch.device
) -> SparseIndex:
    """The :class:`SparseIndex` of the A-shape pattern: query ``i`` sees key ``j <= i`` iff ``i - j < local`` or
    ``j < sinks`` (the reference's ``streaming_forward`` with ``n_init = sinks``, ``n_local = local``)."""
    sinks_t = torch.as_tensor(sinks, device=device).expand(heads).to(torch.int32)
    local_t = torch.as_tensor(local, device=device).expand(heads).to(torch.int32)
    blocks = math.ceil(queries / BLOCK)
    start = torch.arange(blocks, device=device, dtype=torch.int32) * BLOCK
    lo = (start[None, :] - local_t[:, None] + 1).clamp(min=0)  # [H, M]: the window of the block's first query
    hi = torch.clamp(start + BLOCK, max=queries).expand(heads, blocks)
    width = max(1, int(sinks_t.max()))
    j = torch.arange(width, device=device, dtype=torch.int32)
    n_cols = torch.minimum(sinks_t[:, None], lo)  # [H, M]: sinks below the window
    cols = torch.where(j < n_cols[..., None], j, _BIG)
    expand = (batch, heads, blocks)
    return SparseIndex(
        lo.unsqueeze(-1).expand(*expand, 1).contiguous(),
        hi.unsqueeze(-1).expand(*expand, 1).contiguous(),
        torch.ones(expand, dtype=torch.int32, device=device),
        cols.expand(*expand, width).contiguous(),
        n_cols.expand(expand).contiguous(),
        local_t.clone(),
        sinks_t.clone(),
        queries,
    )


def _merge(
    parts: Sequence[tuple[torch.Tensor, SparseIndex]], batch: int, heads: int, device: torch.device
) -> SparseIndex:
    """One :class:`SparseIndex` of all heads from indices of disjoint head subsets."""
    if len(parts) == 1 and parts[0][0].numel() == heads:
        return parts[0][1]
    first = parts[0][1]
    blocks = first.blocks
    width = max(p.range_start.shape[-1] for _, p in parts)
    n_col = max(p.column_index.shape[-1] for _, p in parts)
    rs = torch.zeros(batch, heads, blocks, width, dtype=torch.int32, device=device)
    re = torch.zeros_like(rs)
    rc = torch.zeros(batch, heads, blocks, dtype=torch.int32, device=device)
    ci = torch.full((batch, heads, blocks, n_col), _BIG, dtype=torch.int32, device=device)
    cc = torch.zeros_like(rc)
    window = torch.zeros(heads, dtype=torch.int32, device=device)
    sinks = torch.zeros_like(window)
    for h, p in parts:
        rs[:, h, :, : p.range_start.shape[-1]] = p.range_start
        re[:, h, :, : p.range_end.shape[-1]] = p.range_end
        rc[:, h] = p.range_count
        ci[:, h, :, : p.column_index.shape[-1]] = p.column_index
        cc[:, h] = p.column_count
        window[h] = p.window
        sinks[h] = p.sinks
    return SparseIndex(rs, re, rc, ci, cc, window, sinks, first.queries)


def minference_index(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: float,
    patterns: Sequence[HeadPattern],
    top_k: int = 100,
    kernel: bool | None = None,
) -> SparseIndex:
    """The :class:`SparseIndex` of one layer whose query head ``h`` follows ``patterns[h]``.

    Args:
        query: ``[B, Hq, Q, D]``.
        key: ``[B, Hkv, Q, D]``.
        scale: The layer's score multiplier.
        patterns: One :class:`HeadPattern` per query head.
        top_k: Key blocks per query block of the block-sparse heads.
        kernel: Convert vertical and slash lines with the Triton kernel (default: on CUDA with triton).
    """
    batch, heads, queries, _ = query.shape
    if len(patterns) != heads:
        raise ConfigError(f"MInference got {len(patterns)} head patterns for {heads} query heads")
    device = query.device
    groups = heads // key.shape[1]
    parts: list[tuple[torch.Tensor, SparseIndex]] = []
    for kind in PATTERNS:
        chosen = [h for h, p in enumerate(patterns) if p.kind == kind]
        if not chosen:
            continue
        h = torch.tensor(chosen, device=device)
        vertical = torch.tensor([patterns[i].vertical for i in chosen], device=device)
        slash = torch.tensor([patterns[i].slash for i in chosen], device=device)
        if kind == "a_shape":
            index = a_shape_index(batch, queries, vertical, slash, len(chosen), device)
        else:
            q = query.index_select(1, h)
            k = _kv_heads(key, h, groups)
            if kind == "vertical_slash":
                index = vertical_slash_index(q, k, scale, vertical, slash, kernel=kernel)
            else:
                index = block_sparse_index(q, k, top_k)
        parts.append((h, index))
    return _merge(parts, batch, heads, device)


def mask_from_index(index: SparseIndex, keys: int | None = None) -> torch.Tensor:
    """The boolean mask ``[B, H, Q, keys]`` (``True`` = visible) of ``index``; keys from ``Q`` on are hidden."""
    q = index.queries
    keys = q if keys is None else keys
    device = index.range_start.device
    batch, heads, blocks, width = index.range_start.shape
    valid = torch.arange(width, device=device) < index.range_count.unsqueeze(-1)
    lo = torch.where(valid, index.range_start.long().clamp(0, q), q)
    hi = torch.where(valid, index.range_end.long().clamp(0, q), q)
    lo, hi = torch.minimum(lo, hi), hi
    cover = torch.zeros(batch, heads, blocks, q + 1, dtype=torch.int32, device=device)
    cover.scatter_add_(-1, lo, valid.int()).scatter_add_(-1, hi, -valid.int())
    rows = cover.cumsum(-1)[..., :q] > 0
    c_valid = torch.arange(index.column_index.shape[-1], device=device) < index.column_count.unsqueeze(-1)
    cols = torch.zeros(batch, heads, blocks, q + 1, dtype=torch.bool, device=device)
    cols.scatter_(-1, torch.where(c_valid, index.column_index.long().clamp(0, q), q), True)
    rows |= cols[..., :q]
    pos = torch.arange(q, device=device)
    mask = rows[:, :, pos // BLOCK]  # [B, H, Q, Q]
    mask &= pos[None, :] <= pos[:, None]
    window = index.window.long()[:, None, None]
    offset = pos[:, None] - pos[None, :]
    mask &= (window == 0) | (offset < window) | (pos[None, :] < index.sinks.long()[:, None, None])
    if keys > q:
        mask = F.pad(mask, (0, keys - q), value=False)
    return mask[..., :keys]


def sparse_prefill_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index: SparseIndex,
    scale: float,
    softcap: float | None = None,
) -> torch.Tensor:
    """The definition :func:`sparse_prefill_attention` implements, in fp32 (``[B, Q, Hq, Dv]`` in the query's
    dtype)."""
    q = index.queries
    return reference_attention(query, key[:, :, :q], value[:, :, :q], mask_from_index(index), scale, softcap)


def _tiles(block_d: int) -> tuple[int, int, int]:
    """(rows, keys, warps) per program at a padded head dim, from a sweep on an A100 at 32k queries (Gemma-4-31B's
    global layers have head dim 512: 184 ms against 344 ms for 16 x 32 tiles; head dim 256: 95 ms against 251 ms for
    32 x 32 tiles). Four warps spill registers at head dim 512, sixteen leave too few programs per SM."""
    if block_d <= 256:
        return 64, 64, 4
    return 64, 64, 8


def sparse_prefill_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index: SparseIndex,
    scale: float,
    softcap: float | None = None,
    out: torch.Tensor | None = None,
    tiles: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Causal attention of a prompt over the keys of ``index`` with the Triton kernel.

    Args:
        query: ``[B, Hq, Q, D]``, bf16, fp16 or fp32.
        key: ``[B, Hkv, S >= Q, D]`` in the query's dtype or an fp8 dtype; slots from ``Q`` on are ignored.
        value: ``[B, Hkv, S, Dv]``, like ``key``.
        index: The keys each query sees (``index.queries == Q``).
        scale: Score multiplier.
        softcap: Optional tanh soft cap of the scaled scores.
        out: Optional ``[B, Q, Hq, Dv]`` output buffer.
        tiles: ``(rows, keys)`` per program instead of the default for the head dim (powers of two of at least 16;
            the rows divide 64).

    Returns:
        ``[B, Q, Hq, Dv]`` in the query's dtype.
    """
    if not triton_available():
        raise RuntimeError("sparse_prefill_attention needs triton; use sparse_prefill_reference")
    batch, heads, queries, head_dim = query.shape
    kv_heads = key.shape[1]
    head_dv = value.shape[-1]
    if heads % kv_heads:
        raise ConfigError(f"{heads} query heads are not a multiple of {kv_heads} KV heads")
    if query.dtype not in _DOT_DTYPES:
        raise ConfigError(f"sparse_prefill_attention does not support {query.dtype} queries")
    if index.queries != queries or key.shape[2] < queries or index.window.numel() != heads:
        raise ConfigError(
            f"sparse index for {index.queries} queries and {index.window.numel()} heads does not fit a prompt of "
            f"{queries} queries, {heads} heads and {key.shape[2]} keys"
        )
    kv_fmt = 0
    if key.dtype in _FP8:
        if value.dtype != key.dtype:
            raise ConfigError(f"K/V dtypes {key.dtype} / {value.dtype} differ")
        kv_fmt = _KV_FMT[str(key.dtype).removeprefix("torch.")]
        key, value = key.view(torch.uint8), value.view(torch.uint8)
    elif key.dtype != query.dtype or value.dtype != key.dtype:
        raise ConfigError(f"K/V dtype {key.dtype} / {value.dtype} does not match the query's {query.dtype}")
    block_d = max(16, next_power_of_2(head_dim))
    block_dv = max(16, next_power_of_2(head_dv))
    block_m, block_n, warps = _tiles(max(block_d, block_dv))
    if tiles is not None:
        block_m, block_n = tiles
    if BLOCK % block_m or block_m < 16 or block_n < 16 or block_n & (block_n - 1):
        raise ConfigError(f"sparse prefill tiles {(block_m, block_n)} must be powers of two >= 16, rows dividing 64")
    if out is None:
        out = torch.empty(batch, queries, heads, head_dv, device=query.device, dtype=query.dtype)
    from ._decode_kernels import _DOT
    from ._minference_kernels import _sparse_prefill_kernel

    def flat(t: torch.Tensor) -> torch.Tensor:
        return t.to(torch.int32).contiguous()

    rs, re, rc = flat(index.range_start), flat(index.range_end), flat(index.range_count)
    ci, cc = flat(index.column_index), flat(index.column_count)
    with torch.cuda.device(query.device) if query.is_cuda else nullcontext():
        _sparse_prefill_kernel[(math.ceil(queries / block_m), batch * heads)](
            query, key, value, out, rs, re, rc, ci, cc, flat(index.window), flat(index.sinks),
            *query.stride(), *key.stride(), *value.stride(), *out.stride(),
            queries, heads, index.blocks, rs.shape[-1], ci.shape[-1], float(scale), float(softcap or 1.0),
            GROUPS=heads // kv_heads, HEAD_D=head_dim, HEAD_DV=head_dv, BLOCK_M=block_m, BLOCK_N=block_n,
            BLOCK_D=block_d, BLOCK_DV=block_dv, INDEX_BLOCK=BLOCK, HAS_SOFTCAP=softcap is not None,
            IEEE=query.dtype == torch.float32, DOT_DTYPE=_DOT[query.dtype], KV_FMT=kv_fmt,
            num_warps=warps, num_stages=2,
        )
    return out


# ---- the attachment of the minference technique ---------------------------------------------------------------
@dataclass(frozen=True)
class MInferencePrefill:
    """The :class:`~.prefill.SparsePrefill` of one layer under MInference.

    Attributes:
        heads: The pattern of every query head.
        top_k: Key blocks per query block of the block-sparse heads.
        min_queries: Shortest prompt the patterns apply to.
        name: The path name in :func:`~.interface.kernel_usage`.
    """

    heads: tuple[HeadPattern, ...]
    top_k: int = 100
    min_queries: int = 65
    name: str = "minference"

    def index(self, query: torch.Tensor, key: torch.Tensor, scale: float) -> SparseIndex:
        return minference_index(query, key, scale, self.heads, self.top_k)


def load_patterns(path: str | Path, ratio: float = 1.0) -> tuple[tuple[HeadPattern, ...], ...]:
    """Per layer, the head patterns of a pattern file of the reference (``minference/configs/*.json``).

    The file holds one object per layer mapping each query head (as a string) to ``[type, v, s, score]`` with type
    ``"vertical_and_slash"``, ``"block_sparse"`` or ``"stream_llm"``; ``ratio`` scales ``v`` and ``s`` as the
    reference's ``minference_ratio`` does (``int(v * ratio)``).
    """
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot read MInference pattern file {str(path)!r}: {error}") from error
    names = {official: ours for ours, official in PATTERNS.items()}
    if not isinstance(data, list) or not all(isinstance(layer, dict) for layer in data):
        raise ConfigError(f"MInference pattern file {str(path)!r} must hold a list of per-layer objects")
    layers = []
    for number, layer in enumerate(data):
        heads = []
        for h in range(len(layer)):
            entry = layer.get(str(h))
            if not isinstance(entry, list) or len(entry) < 3 or entry[0] not in names:
                raise ConfigError(f"MInference pattern file {str(path)!r}: bad entry for layer {number} head {h}")
            heads.append(HeadPattern(names[entry[0]], int(int(entry[1]) * ratio), max(1, int(int(entry[2]) * ratio))))
        layers.append(tuple(heads))
    return tuple(layers)
