"""TriForce's retrieval view of the cache: chunks of the prompt chosen once per generation by the query's attention.

:func:`retrieval_keep` is the selection rule. :class:`RetrievalPath` runs it for every full-attention layer in the
first draft step of a generation (which runs on the whole cache and records each layer's query and keys) and keeps the
result in persistent per-batch-size buffers, which the later draft steps (and their CUDA graphs) read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from .path import SparseKVPath

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...runtime import LayerContext

__all__ = ["RetrievalPath", "retrieval_keep"]

_SDPA = F.scaled_dot_product_attention


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
    batch, heads = query.shape[0], query.shape[1]
    n_chunks = key.shape[-2] // chunk_size
    device = query.device
    keep = torch.ones(batch, heads // groups, width, dtype=torch.bool, device=device)
    if n_chunks == 0:
        return keep.repeat_interleave(groups, dim=1)[:, :, None]
    k = key[..., : n_chunks * chunk_size, :].float()
    means = k.unflatten(-2, (n_chunks, chunk_size)).mean(-2)  # [B, H_k', C, D]
    if means.shape[1] != heads:
        means = means.repeat_interleave(heads // means.shape[1], dim=1)
    scores = torch.einsum("bhd,bhcd->bhc", query[:, :, 0].float(), means)
    scores = scores.unflatten(1, (heads // groups, groups)).sum(2)  # [B, H_k, C]
    region = torch.div(positions.to(device), chunk_size, rounding_mode="floor").clamp(max=n_chunks)  # whole chunks
    chunk = torch.arange(n_chunks, device=device)
    inside = chunk[None, :] < region[:, None]  # [B, C]
    scores = scores.masked_fill(~inside[:, None], float("-inf"))
    scores[..., 0] = float("inf")
    picked = scores.topk(min(select_sets, n_chunks), dim=-1).indices
    chosen = torch.zeros_like(scores, dtype=torch.bool).scatter(-1, picked, True) & inside[:, None]
    slots = torch.arange(width, device=device)
    tail = slots[None, :] >= (region * chunk_size)[:, None]  # [B, width]
    selected = F.pad(chosen.repeat_interleave(chunk_size, dim=-1), (0, width - n_chunks * chunk_size))
    keep = selected | tail[:, None]
    if valid is not None:
        keep = keep | ~valid.to(device)[:, None, None]
    return keep.repeat_interleave(groups, dim=1)[:, :, None]


class _AttentionInputs(TorchFunctionMode):
    """Records the query and keys of the last ``scaled_dot_product_attention`` call inside the block."""

    def __init__(self) -> None:
        super().__init__()
        self.query: torch.Tensor | None = None
        self.key: torch.Tensor | None = None

    def __torch_function__(self, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None) -> Any:
        kwargs = kwargs or {}
        if func is _SDPA:
            self.query = args[0] if args else kwargs["query"]
            self.key = args[1] if len(args) > 1 else kwargs["key"]
        return func(*args, **kwargs)


class RetrievalPath(SparseKVPath):
    """The draft path of TriForce's retrieval drafter (one per drafter; it holds the selection of its generation).

    :meth:`select_next` makes the next :meth:`~SparseKVPath.run` a selecting step: every full-attention layer runs
    on its whole cache, the path records the layer's query and keys and stores :func:`retrieval_keep` in the
    buffer of the batch size and key count. Until a selection exists :meth:`keep` returns ``None`` (dense layers),
    afterwards the buffers. Buffers are allocated once per ``(batch size, key count)`` and overwritten in place, so
    CUDA graphs recorded over a path stay valid across generations.

    Args:
        adapter: The model's adapter.
        chunk_size: Slots per retrieval chunk.
        select_sets: Chunks kept per KV head.
    """

    def __init__(self, adapter: ArchAdapter, chunk_size: int, select_sets: int) -> None:
        super().__init__(adapter)
        self.chunk_size, self.select_sets = chunk_size, select_sets
        self._groups = {i: self._group_size(i) for i in self.layers}
        self._buffers: dict[tuple[int, int], dict[int, torch.Tensor]] = {}
        self._active: tuple[int, int] | None = None
        self._selecting: tuple[torch.Tensor, torch.Tensor | None] | None = None
        #: Number of selections made (one per generation), for tests and diagnostics.
        self.selections = 0

    def _group_size(self, index: int) -> int:
        module = self.adapter.attention_module(index)
        groups = getattr(module, "num_key_value_groups", 1)
        return groups if isinstance(groups, int) and groups > 0 else 1

    @property
    def selected(self) -> bool:
        """Whether a selection exists for the current generation."""
        return self._active is not None

    def reset(self) -> None:
        """Forget the current selection (the next generation selects anew)."""
        self._active = None

    def select_next(self, positions: torch.Tensor, valid: torch.Tensor | None) -> None:
        """Make the next run select, for queries at ``positions`` ``[B]`` (``valid`` ``[B]`` or ``None``)."""
        self._selecting = (positions, valid)

    def finish_selection(self) -> bool:
        """End a selecting run; whether it selected (``False`` if no layer called ``scaled_dot_product_attention``)."""
        self._selecting = None
        if self._active is None:
            return False
        self.selections += 1
        return True

    def buffer(self, batch: int, width: int) -> dict[int, torch.Tensor] | None:
        """The keep buffers of ``batch`` rows and ``width`` keys by layer, ``None`` if not allocated."""
        return self._buffers.get((batch, width))

    def keep(self, index: int, mask: torch.Tensor, ctx: LayerContext) -> torch.Tensor | None:
        if self._selecting is not None or self._active is None:
            return None
        buffers = self._buffers.get((ctx.batch_size, mask.shape[-1]))
        if buffers is None or self._active != (ctx.batch_size, mask.shape[-1]):
            return None
        return buffers[index]

    def call_dense(self, index: int, layer: Any, hidden: torch.Tensor, args: tuple[Any, ...],
                   kwargs: dict[str, Any], ctx: LayerContext) -> Any:
        if self._selecting is None:
            return layer(hidden, *args, **kwargs)
        tap = _AttentionInputs()
        with tap:
            out = layer(hidden, *args, **kwargs)
        if tap.query is None or tap.key is None:
            return out
        positions, valid = self._selecting
        width = kwargs["attention_mask"].shape[-1]
        keep = retrieval_keep(
            tap.query.detach(), tap.key.detach(), positions, valid,
            chunk_size=self.chunk_size, select_sets=self.select_sets, groups=self._groups[index],
            width=max(width, tap.key.shape[-2]),
        )[..., :width]
        self._store(index, keep)
        return out

    def _store(self, index: int, keep: torch.Tensor) -> None:
        key = (keep.shape[0], keep.shape[-1])
        buffers = self._buffers.setdefault(key, {})
        target = buffers.get(index)
        if target is None or target.shape != keep.shape or target.device != keep.device:
            buffers[index] = keep.clone()
        else:
            target.copy_(keep)
        self._active = key

    def reorder(self, rows: torch.Tensor) -> None:
        """Keep the batch rows ``rows`` of the selection (in the buffers of the new batch size)."""
        if self._active is None:
            return
        batch, width = self._active
        source = self._buffers[(batch, width)]
        for index, keep in list(source.items()):
            self._store(index, keep.index_select(0, rows.to(keep.device)))
        self._active = (rows.numel(), width)
