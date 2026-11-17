"""How a draft step's full-attention layers read the budget of a sparse draft path.

A :class:`~.path.SparseKVPath` decides which cache entries a draft query may read (its budget). A
:class:`DraftView` decides how the layer is made to read only those entries:

``"gathered"`` (:class:`GatheredView`, the default)
    Gathers the budget's K/V entries and the matching columns of the attention mask into tensors of the budget's
    width, so the layer's attention, whatever kernel runs it, computes over the budget only. Its FLOPs and the KV
    bytes it reads scale with the budget instead of the context length.
``"masked"`` (:class:`MaskedView`, the numerical reference)
    Hides the entries outside the budget in the attention mask. Only a kernel that skips hidden keys (economize's
    decode kernel, which skips every key block the mask hides) saves the reads; the ``sdpa`` and ``eager`` paths
    read the whole cache and only simulate the saving.

Both views give the attention over the same entries, so their drafts agree up to the summation order. New views
subclass :class:`DraftView` and register in :data:`DRAFT_VIEWS`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from ..._identity import IdentityMap
from ...core.metadata import Saving, Simulation
from ...errors import UnsupportedModelError
from ...registry import Registry

if TYPE_CHECKING:
    from ...runtime import LayerContext
    from .path import SparseKVPath

__all__ = [
    "DRAFT_VIEWS",
    "MASKED_READS",
    "DraftBudget",
    "DraftView",
    "GatheredView",
    "MaskedView",
    "gather_entries",
    "restrict",
]

#: How the masked view simulates a sparse draft on a kernel that reads hidden keys.
MASKED_READS = Simulation(Saving.LATENCY, "masks the KV entries outside the draft budget, which the attention "
                          "kernel still reads")


def restrict(mask: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """``mask`` (a boolean ``sdpa`` mask or an additive ``eager`` mask, ``[B, 1, Q, K]``) hiding every key that
    ``keep`` (boolean, broadcastable to ``[B, H, Q, K]``) drops; per-head ``keep`` gives a per-head mask."""
    if mask.dtype == torch.bool:
        return mask & keep
    return mask.masked_fill(~keep, torch.finfo(mask.dtype).min)


@dataclass(frozen=True)
class DraftBudget:
    """The cache entries one full-attention layer of a draft step reads.

    Attributes:
        slots: Cache slots read by every row, ``[B, H, W]`` with ``H`` 1 (all heads read the same slots) or the
            layer's number of KV heads. Entries may repeat the trash slot, which the layer's mask hides.
        keep: Which of the gathered slots each query keeps besides what the layer's mask allows, boolean and
            broadcastable to ``[B, 1, Q, W]``; ``None`` keeps all of them.
    """

    slots: torch.Tensor
    keep: torch.Tensor | None = None

    @property
    def width(self) -> int:
        """Entries per row and head (``W``)."""
        return self.slots.shape[-1]


def gather_entries(buffer: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """The entries ``slots`` ``[B, H', W]`` of a cache buffer ``[B, H, N, D]``, ``[B, H, W, D]``.

    ``H'`` is 1 (the same slots for every head), ``H`` or a divisor of ``H`` (the buffer repeats each KV head).
    One-byte floating-point buffers (fp8 caches) are gathered through their bytes.
    """
    batch, heads, _, dim = buffer.shape
    index = slots.to(buffer.device)
    if index.shape[1] not in (1, heads):
        index = index.repeat_interleave(heads // index.shape[1], dim=1)
    index = index[..., None].expand(batch, heads, index.shape[-1], dim)
    if buffer.is_floating_point() and buffer.element_size() == 1:
        return buffer.view(torch.uint8).gather(2, index).view(buffer.dtype)
    return buffer.gather(2, index)


class _GatheringCache:
    """The layer's cache seen through a budget: ``update`` writes as usual and returns the budget's entries."""

    def __init__(self, cache: Any, slots: torch.Tensor) -> None:
        self._cache, self._slots = cache, slots

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args: Any,
               **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        keys, values = self._cache.update(key_states, value_states, layer_idx, *args, **kwargs)
        return gather_entries(keys, self._slots), gather_entries(values, self._slots)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cache, name)


def _mask_memo(shared: dict[Any, Any], kind: str) -> IdentityMap[torch.Tensor, Any]:
    """The step's ``kind`` entries keyed by the mask object: a layer whose mask was built after an earlier layer's
    mask was freed may get that mask's ``id``, never its entry."""
    memo = shared.get(kind)
    if memo is None:
        memo = shared[kind] = IdentityMap()
    return memo


class DraftView(ABC):
    """How a full-attention layer of a draft step is made to read the budget of its path."""

    #: How the view simulates the saving on the Hugging Face attention paths (``None``: it realizes it).
    simulation: ClassVar[Simulation | None] = None

    @abstractmethod
    def call(self, path: SparseKVPath, index: int, layer: Any, hidden: torch.Tensor, args: tuple[Any, ...],
             kwargs: dict[str, Any], ctx: LayerContext, shared: dict[Any, Any]) -> Any:
        """Run decoder layer ``index`` (``layer``) of a draft step on its budget.

        Args:
            path: The draft path.
            index: The layer (1-indexed).
            layer: The decoder layer or the stand-in the runner calls.
            hidden: The layer's input.
            args: Positional arguments of the call after ``hidden``.
            kwargs: Keyword arguments of the call (``attention_mask``, ``past_key_values``, ...).
            ctx: The step's context.
            shared: Per-step scratch the view shares across layers (for budgets of a
                :attr:`~.path.SparseKVPath.layer_independent` path).
        """

    @staticmethod
    def mask_of(path: SparseKVPath, index: int, kwargs: dict[str, Any]) -> torch.Tensor:
        """The layer's attention mask, which must be a tensor."""
        mask = kwargs.get("attention_mask")
        if not isinstance(mask, torch.Tensor):
            raise UnsupportedModelError(f"{type(path).__name__} restricts tensor attention masks; layer {index} "
                                        f"got {type(mask).__name__}")
        return mask


class MaskedView(DraftView):
    """The reference view: the layer's mask hides the entries outside the budget; the layer reads its whole cache."""

    simulation = MASKED_READS

    def call(self, path: SparseKVPath, index: int, layer: Any, hidden: torch.Tensor, args: tuple[Any, ...],
             kwargs: dict[str, Any], ctx: LayerContext, shared: dict[Any, Any]) -> Any:
        mask = self.mask_of(path, index, kwargs)
        memo = _mask_memo(shared, "masked")
        restricted = memo.get(mask) if path.layer_independent else None
        if restricted is None:
            keep = path.keep(index, mask, ctx)
            if keep is None:
                return path.call_dense(index, layer, hidden, args, kwargs, ctx)
            restricted = restrict(mask, keep)
            if path.layer_independent:
                memo[mask] = restricted
        kwargs["attention_mask"] = restricted
        return layer(hidden, *args, **kwargs)


class GatheredView(DraftView):
    """The default view: the layer attends over tensors that hold only the budget's entries.

    The layer writes its K/V through the cache as usual; the cache it receives returns the budget's entries
    (:func:`gather_entries`) in place of the whole buffers, and its mask is the matching columns of the step's mask
    (restricted by :attr:`DraftBudget.keep`, repeated to the query heads for a per-head budget). KV-shared layers
    (Gemma-4) get the budget of their source's entries. A layer without a budget (the path returns ``None``, the
    budget is no narrower than the cache, or the mask is narrower than the cache) runs on the masked view.

    The gathered tensors are copies, so formats that a cache attaches to its key buffer for a specialized kernel
    (:func:`economize.attention.attach_packed`: KIVI's codes, Quest's pages) are not used on this path; the layer
    attends over the dense entries the cache keeps next to them.
    """

    simulation = None

    def __init__(self) -> None:
        self._fallback = MaskedView()

    def call(self, path: SparseKVPath, index: int, layer: Any, hidden: torch.Tensor, args: tuple[Any, ...],
             kwargs: dict[str, Any], ctx: LayerContext, shared: dict[Any, Any]) -> Any:
        mask = self.mask_of(path, index, kwargs)
        memo = _mask_memo(shared, "gathered")
        entry = memo.get(mask) if path.layer_independent else None
        if entry is None:
            budget = path.budget(index, mask, ctx)
            num_slots = ctx.cache.num_slots
            if budget is None or budget.width >= num_slots or mask.shape[-1] < num_slots:
                return self._fallback.call(path, index, layer, hidden, args, kwargs, ctx, shared)
            entry = (budget.slots, self._gathered_mask(mask, budget, path.group_size(index)))
            if path.layer_independent:
                memo[mask] = entry
        slots, gathered = entry
        kwargs["attention_mask"] = gathered
        kwargs["past_key_values"] = _GatheringCache(kwargs.get("past_key_values", ctx.cache), slots)
        states = kwargs.get("shared_kv_states")
        if isinstance(states, dict):
            kwargs["shared_kv_states"] = {
                name: tuple(gather_entries(t, slots) for t in value) for name, value in states.items()
            }
        return layer(hidden, *args, **kwargs)

    @staticmethod
    def _gathered_mask(mask: torch.Tensor, budget: DraftBudget, groups: int) -> torch.Tensor:
        """The columns ``budget.slots`` of ``mask`` (``[1 | B, 1 | H_q, Q, N]``), restricted by ``budget.keep``.

        A per-KV-head budget gives a per-query-head mask (each KV head's columns repeated to its ``groups`` query
        heads), any other budget a mask with the heads of ``mask``.
        """
        slots = budget.slots.to(mask.device)
        batch, mask_heads, queries = slots.shape[0], mask.shape[1], mask.shape[2]
        if mask_heads > slots.shape[1] > 1:
            slots = slots.repeat_interleave(mask_heads // slots.shape[1], dim=1)
        heads = max(slots.shape[1], mask_heads)
        index = slots[:, :, None, :].expand(batch, heads, queries, slots.shape[-1])
        gathered = mask.expand(batch, heads, queries, mask.shape[-1]).gather(3, index)
        if budget.keep is not None:
            gathered = restrict(gathered, budget.keep.to(mask.device))
        if heads > 1 and mask_heads == 1:
            gathered = gathered.repeat_interleave(groups, dim=1)
        return gathered


#: Views of a sparse draft path by name (the ``view`` field of the MagicDec and TriForce configurations).
DRAFT_VIEWS: Registry[type[DraftView]] = Registry("sparse draft view")
DRAFT_VIEWS.register("gathered", GatheredView)
DRAFT_VIEWS.register("masked", MaskedView)
