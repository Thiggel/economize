"""TriForce's retrieval view of the cache: chunks of the prompt chosen once per generation by the query's attention.

:func:`retrieval_keep` is the selection rule as a mask over the cache, :func:`chunk_slots` the same selection as cache
slots. :class:`RetrievalPath` runs it for every full-attention layer in the first draft step of a generation (which
runs on the whole cache and records each layer's query and keys) and keeps the result in persistent per-batch-size
buffers, which the later draft steps (and their CUDA graphs) read.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from .path import SparseKVPath
from .views import DraftBudget, GatheredView

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...runtime import KVCache, LayerCall, LayerContext

__all__ = ["RetrievalPath", "chunk_slots", "retrieval_keep", "tail_width"]

_SDPA = F.scaled_dot_product_attention


def _choose(query: torch.Tensor, key: torch.Tensor, positions: torch.Tensor, *, chunk_size: int, select_sets: int,
            groups: int) -> tuple[torch.Tensor | None, torch.Tensor]:
    """TriForce's chunk choice (see :func:`retrieval_keep`): the chosen chunks ``[B, H_k, C]`` (boolean, ``None`` if
    the keys hold no whole chunk) and the number of whole prompt chunks per row ``[B]``."""
    heads = query.shape[1]
    n_chunks = key.shape[-2] // chunk_size
    device = query.device
    region = torch.div(positions.to(device), chunk_size, rounding_mode="floor").clamp(max=n_chunks)  # whole chunks
    if n_chunks == 0:
        return None, region
    k = key[..., : n_chunks * chunk_size, :].float()
    means = k.unflatten(-2, (n_chunks, chunk_size)).mean(-2)  # [B, H_k', C, D]
    if means.shape[1] != heads:
        means = means.repeat_interleave(heads // means.shape[1], dim=1)
    scores = torch.einsum("bhd,bhcd->bhc", query[:, :, 0].float(), means)
    scores = scores.unflatten(1, (heads // groups, groups)).sum(2)  # [B, H_k, C]
    inside = torch.arange(n_chunks, device=device)[None, :] < region[:, None]  # [B, C]
    scores = scores.masked_fill(~inside[:, None], float("-inf"))
    scores[..., 0] = float("inf")
    picked = scores.topk(min(select_sets, n_chunks), dim=-1).indices
    chosen = torch.zeros_like(scores, dtype=torch.bool).scatter(-1, picked, True) & inside[:, None]
    return chosen, region


def _keep_mask(chosen: torch.Tensor | None, region: torch.Tensor, valid: torch.Tensor | None, *, heads: int,
               chunk_size: int, groups: int, width: int) -> torch.Tensor:
    """The mask of :func:`retrieval_keep` from the output of :func:`_choose`."""
    device = region.device
    if chosen is None:
        keep = torch.ones(region.shape[0], heads // groups, width, dtype=torch.bool, device=device)
        return keep.repeat_interleave(groups, dim=1)[:, :, None]
    slots = torch.arange(width, device=device)
    tail = slots[None, :] >= (region * chunk_size)[:, None]  # [B, width]
    selected = F.pad(chosen.repeat_interleave(chunk_size, dim=-1), (0, width - chosen.shape[-1] * chunk_size))
    keep = selected | tail[:, None]
    if valid is not None:
        keep = keep | ~valid.to(device)[:, None, None]
    return keep.repeat_interleave(groups, dim=1)[:, :, None]


def retrieval_keep(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    valid: torch.Tensor | None,
    *,
    chunk_size: int,
    select_sets: int,
    groups: int,
    width: int,
) -> torch.Tensor:
    """The slots each query head of a retrieval draft path reads, ``[B, H_q, 1, width]`` boolean.

    The slots below ``positions`` (the prompt cached before the query) are cut into whole chunks of ``chunk_size``.
    Every chunk is scored by the query's dot product with the chunk's mean key, summed over the ``groups`` query heads
    of a KV head, so that all heads of a group read the same slots. The first chunk and the ``select_sets - 1`` best
    other chunks are kept (TriForce's rule); every slot from the end of the last whole chunk on is kept, and rows that
    are not ``valid`` keep every slot.

    Args:
        query: The query of one token per row, ``[B, H_q, 1, D]``.
        key: The keys of the layer's cache, ``[B, H_k, K, D]`` with ``H_k`` dividing ``H_q`` (repeated keys allowed).
        positions: The query's position per row, ``[B]`` (the number of cached prompt slots).
        valid: ``[B]``, ``None`` for all rows.
        chunk_size: Slots per chunk.
        select_sets: Chunks kept per KV head.
        groups: Query heads per KV head of the model.
        width: Last dimension of the returned mask (the attention mask's key count, at least ``K``).
    """
    chosen, region = _choose(query, key, positions, chunk_size=chunk_size, select_sets=select_sets, groups=groups)
    return _keep_mask(chosen, region, valid, heads=query.shape[1], chunk_size=chunk_size, groups=groups, width=width)


def chunk_slots(chosen: torch.Tensor, chunk_size: int, select_sets: int, trash: int) -> torch.Tensor:
    """The cache slots of the chosen chunks ``[B, H_k, C]``, ``[B, H_k, select_sets * chunk_size]`` in slot order.

    A row with fewer than ``select_sets`` chosen chunks (a short prompt) fills the rest with the ``trash`` slot,
    which the attention mask hides.
    """
    n_chunks = chosen.shape[-1]
    index = torch.arange(n_chunks, device=chosen.device)
    order = torch.where(chosen, index, n_chunks).sort(-1).values[..., :select_sets]
    if order.shape[-1] < select_sets:
        order = F.pad(order, (0, select_sets - order.shape[-1]), value=n_chunks)
    slots = order[..., None] * chunk_size + torch.arange(chunk_size, device=chosen.device)
    slots = torch.where(order[..., None] == n_chunks, trash, slots)
    return slots.flatten(-2)


def tail_width(need: int) -> int:
    """``need`` rounded up to a multiple of ``max(64, 2 ** (floor(log2(need)) - 2))`` (fewer than 64 or a quarter of
    ``need`` extra slots), so that generations of similar lengths share a tail width and its CUDA graphs."""
    if need <= 0:
        return 0
    grain = max(64, 1 << max(need.bit_length() - 3, 0))
    return -(-need // grain) * grain


class _AttentionInputs(TorchFunctionMode):
    """Records the query and keys of the last ``scaled_dot_product_attention`` call inside the block."""

    def __init__(self) -> None:
        super().__init__()
        self.query: torch.Tensor | None = None
        self.key: torch.Tensor | None = None

    def __torch_function__(
        self,
        func: Callable[..., object],
        types: Collection[type],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        kwargs = kwargs or {}
        if func is _SDPA:
            self.query = cast(torch.Tensor, args[0] if args else kwargs["query"])
            self.key = cast(torch.Tensor, args[1] if len(args) > 1 else kwargs["key"])
        return func(*args, **kwargs)


class _CacheHeads:
    """The layer's cache, recording the number of heads of the key buffer that ``update`` returns."""

    def __init__(self, cache: KVCache) -> None:
        self._cache, self.heads = cache, 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args: object,
               **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        keys, values = self._cache.update(key_states, value_states, layer_idx, *args, **kwargs)
        self.heads = keys.shape[1]
        return keys, values

    def __getattr__(self, name: str) -> object:
        return getattr(self._cache, name)


class RetrievalPath(SparseKVPath):
    """The draft path of TriForce's retrieval drafter (one per drafter; it holds the selection of its generation).

    :meth:`select_next` makes the next :meth:`~SparseKVPath.run` a selecting step: every full-attention layer runs
    on its whole cache, and the path records the layer's query and keys and stores the selection in buffers of the
    batch size: the keep mask of :func:`retrieval_keep` (read by the masked view) and the slots of the chosen chunks
    (:func:`chunk_slots`, read by the gathered view). Until a selection exists :meth:`keep` and :meth:`budget`
    return ``None`` (dense layers). Buffers are allocated once per batch size (and key count) and overwritten in
    place, so CUDA graphs recorded over a path stay valid across generations.

    :meth:`finish_selection` then fixes the generation's *tail*: the slots from the end of each row's last whole
    prompt chunk to the end of the cache, which hold the rest of the prompt and every position the generation
    writes. A gathered draft layer reads the chosen chunks and the tail, ``select_sets * chunk_size`` plus the tail
    width entries per KV head. The tail does not grow with the prompt: the cache ends at the generation's capacity
    (its longest prompt plus the new tokens and one cycle, rounded up to the granularity of a graphed decoder, which
    may also reuse a larger cache), so its width is the number of new tokens plus that slack and the spread of the
    batch's prompt lengths. The drafter hands the tail to every later step as a tensor (:meth:`using_tail`), so its
    width keys the step's CUDA graphs; :func:`tail_width` buckets it. A selection whose rows are not all valid, or
    whose gathered entries would be no fewer than the cache's, gets an empty tail and runs on the masked view; so
    does a model whose cache stores fewer heads than the selection has (a latent cache shared by all heads, as in
    DeepSeek's multi-head latent attention), since its heads cannot each read their own chunks.

    Args:
        adapter: The model's adapter.
        chunk_size: Slots per retrieval chunk.
        select_sets: Chunks kept per KV head.
        view: Name of the :class:`~.views.DraftView` (see :class:`~.path.SparseKVPath`).
    """

    def __init__(self, adapter: ArchAdapter, chunk_size: int, select_sets: int, view: str = "gathered") -> None:
        super().__init__(adapter, view)
        self.chunk_size, self.select_sets = chunk_size, select_sets
        self._buffers: dict[tuple[int, int], dict[int, torch.Tensor]] = {}
        self._chunks: dict[int, dict[int, torch.Tensor]] = {}
        self._stored: set[int] = set()
        self._active: tuple[int, int] | None = None
        self._selecting: tuple[torch.Tensor, torch.Tensor | None] | None = None
        self._region: torch.Tensor | None = None
        self._trash = 0
        self._tail: torch.Tensor | None = None
        self._step_tail: torch.Tensor | None = None
        #: Number of selections made (one per generation), for tests and diagnostics.
        self.selections = 0

    @property
    def selected(self) -> bool:
        """Whether a selection exists for the current generation."""
        return self._active is not None

    @property
    def tail(self) -> torch.Tensor | None:
        """The tail slots of the current selection ``[B, W]`` (``W = 0``: the draft layers run on the masked view),
        ``None`` without a selection."""
        return self._tail if self._active is not None else None

    def reset(self) -> None:
        """Forget the current selection (the next generation selects anew)."""
        self._active = self._region = self._tail = None

    def select_next(self, positions: torch.Tensor, valid: torch.Tensor | None) -> None:
        """Make the next run select, for queries at ``positions`` ``[B]`` (``valid`` ``[B]`` or ``None``)."""
        self.reset()
        self._selecting = (positions, valid)
        self._stored = set()

    def finish_selection(self) -> bool:
        """End a selecting run and fix the generation's tail; whether it selected (``False`` if no layer called
        ``scaled_dot_product_attention``). Synchronizes with the device once when the view gathers."""
        selecting, self._selecting = self._selecting, None
        if self._active is None:
            return False
        self.selections += 1
        region = self._region
        device = next(iter(self._buffers[self._active].values())).device
        self._tail = torch.zeros(self._active[0], 0, dtype=torch.long, device=device)
        if not isinstance(self.view, GatheredView) or region is None or not self._stored >= set(self.layers):
            return True
        start = region * self.chunk_size
        valid = None if selecting is None else selecting[1]
        every = torch.ones((), dtype=torch.bool, device=start.device) if valid is None else valid.all()
        first, all_valid = torch.stack([start.min(), every.to(start.device, start.dtype)]).tolist()
        width = tail_width(self._trash - first)
        if all_valid and self.select_sets * self.chunk_size + width <= self._trash:
            self._tail = (start[:, None] + torch.arange(width, device=start.device)).clamp(max=self._trash)
        return True

    @contextmanager
    def using_tail(self, tail: torch.Tensor | None) -> Iterator[None]:
        """Run the steps inside on ``tail`` (a step's input in place of :attr:`tail`, so CUDA graphs replay it).

        ``tail`` is ``[B, W]``, or ``[B, N, W]`` for the per-node copies of a tree step (equal for all nodes)."""
        previous, self._step_tail = self._step_tail, tail
        try:
            yield
        finally:
            self._step_tail = previous

    def buffer(self, batch: int, width: int) -> dict[int, torch.Tensor] | None:
        """The keep buffers of ``batch`` rows and ``width`` keys by layer, ``None`` if not allocated."""
        return self._buffers.get((batch, width))

    def chunks(self, batch: int) -> dict[int, torch.Tensor] | None:
        """The chunk-slot buffers of ``batch`` rows by layer, ``None`` if not allocated."""
        return self._chunks.get(batch)

    def keep(self, index: int, mask: torch.Tensor, ctx: LayerContext) -> torch.Tensor | None:
        if self._selecting is not None or self._active is None:
            return None
        buffers = self._buffers.get((ctx.batch_size, mask.shape[-1]))
        if buffers is None or self._active != (ctx.batch_size, mask.shape[-1]):
            return None
        return buffers[index]

    def budget(self, index: int, mask: torch.Tensor, ctx: LayerContext) -> DraftBudget | None:
        """The chosen chunks of layer ``index`` and the step's tail; ``None`` (the masked view) while selecting,
        without a selection for the step's batch, or with an empty tail."""
        if self._selecting is not None or self._active is None or self._active[0] != ctx.batch_size:
            return None
        tail = self._step_tail if self._step_tail is not None else self._tail
        chunks = self._chunks.get(ctx.batch_size, {}).get(index)
        if tail is None or tail.shape[-1] == 0 or chunks is None:
            return None
        tail = tail.to(chunks.device)
        if tail.ndim == 3:
            tail = tail[:, 0]
        return DraftBudget(torch.cat([chunks, tail[:, None].expand(-1, chunks.shape[1], -1)], dim=-1))

    def call_dense(self, index: int, layer: LayerCall, hidden: torch.Tensor, args: tuple[object, ...],
                   kwargs: dict[str, object], ctx: LayerContext) -> torch.Tensor:
        if self._selecting is None:
            return layer(hidden, *args, **kwargs)
        layer_cache = cast("KVCache", kwargs.get("past_key_values", ctx.cache))  # the step's cache or a wrapper of it
        tap, cache = _AttentionInputs(), _CacheHeads(layer_cache)
        kwargs["past_key_values"] = cache
        with tap:
            out = layer(hidden, *args, **kwargs)
        if tap.query is None or tap.key is None:
            return out
        positions, valid = self._selecting
        width = cast(torch.Tensor, kwargs["attention_mask"]).shape[-1]
        query, key, groups = tap.query.detach(), tap.key.detach(), self.group_size(index)
        chosen, region = _choose(query, key, positions, chunk_size=self.chunk_size, select_sets=self.select_sets,
                                 groups=groups)
        keep = _keep_mask(chosen, region, valid, heads=query.shape[1], chunk_size=self.chunk_size, groups=groups,
                          width=max(width, key.shape[-2]))[..., :width]
        self._store(index, keep)
        if chosen is not None and key.shape[-2] == ctx.cache.num_slots and _gatherable(chosen.shape[1], cache.heads):
            self._trash, self._region = ctx.cache.num_slots - 1, region
            self._store_chunks(index, chunk_slots(chosen, self.chunk_size, self.select_sets, self._trash))
            self._stored.add(index)
        return out

    def _store(self, index: int, keep: torch.Tensor) -> None:
        key = (keep.shape[0], keep.shape[-1])
        buffers = self._buffers.setdefault(key, {})
        buffers[index] = _overwrite(buffers.get(index), keep)
        self._active = key

    def _store_chunks(self, index: int, slots: torch.Tensor) -> None:
        buffers = self._chunks.setdefault(slots.shape[0], {})
        buffers[index] = _overwrite(buffers.get(index), slots)

    def reorder(self, rows: torch.Tensor) -> None:
        """Keep the batch rows ``rows`` of the selection (in the buffers of the new batch size)."""
        if self._active is None:
            return
        batch, width = self._active
        for index, keep in list(self._buffers[(batch, width)].items()):
            self._store(index, keep.index_select(0, rows.to(keep.device)))
        for index, slots in list(self._chunks.get(batch, {}).items()):
            self._store_chunks(index, slots.index_select(0, rows.to(slots.device)))
        if self._region is not None:
            self._region = self._region.index_select(0, rows.to(self._region.device))
        if self._tail is not None:
            self._tail = self._tail.index_select(0, rows.to(self._tail.device))
        self._active = (rows.numel(), width)


def _gatherable(selection_heads: int, cache_heads: int) -> bool:
    """Whether per-KV-head chunks (``selection_heads`` sets) can be gathered from a cache buffer of ``cache_heads``
    heads: each buffer head must belong to one selection head. A latent cache that stores one entry per position for
    every head (DeepSeek's multi-head latent attention) cannot give each head its own chunks."""
    return selection_heads == 1 or (cache_heads >= selection_heads and cache_heads % selection_heads == 0)


def _overwrite(target: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    """``value`` copied into ``target`` in place when the shapes and devices match, else a new buffer."""
    if target is None or target.shape != value.shape or target.device != value.device:
        return value.clone()
    target.copy_(value)
    return target
