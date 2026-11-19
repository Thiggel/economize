"""The draft path of sparse-KV self-speculation: the whole model, with full-attention layers reading part of the cache.

A :class:`SparseKVPath` is a :class:`~economize.techniques.early_exit.SelfDraftStep`: a draft step runs layers
``1..L`` with the model's own weights, and every full-attention layer is replaced by a stand-in that runs the layer
on the path's budget of cache entries. The path's :class:`~.views.DraftView` decides how: the default
:class:`~.views.GatheredView` hands the layer only the budget's K/V entries (:meth:`SparseKVPath.budget`), so its
attention reads a budget-sized cache on any kernel; the reference :class:`~.views.MaskedView` hides the other entries
in the mask (:meth:`SparseKVPath.keep`). The cache is the model's own, so the draft path writes the K/V of the drafted
positions where the next exact pass overwrites them. The exact pass never runs through the path and reads the whole
cache, which keeps decoding lossless.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from ...arch.spec import StateKind
from .views import DRAFT_VIEWS, DraftBudget, DraftView, StepMemo, restrict

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...decoding.cost import LayerFootprint
    from ...runtime import LayerCall, LayerContext, LayerRunner

__all__ = ["SparseKVPath", "restrict", "sparse_layers", "streaming_budget", "streaming_keep"]


def sparse_layers(adapter: ArchAdapter) -> tuple[int, ...]:
    """The layers (1-indexed) whose cache a sparse draft path reads in part: the full-attention layers.

    Sliding-window and chunked layers already read a bounded part of the cache, and recurrent layers keep no
    per-position cache, so they run unchanged.
    """
    return tuple(
        i for i in range(1, adapter.num_layers + 1)
        if adapter.attention_spec(i).kind == "full" and adapter.attention_spec(i).state_kind is StateKind.KV
    )


def streaming_keep(ctx: LayerContext, num_keys: int, sinks: int, window: int, device: torch.device) -> torch.Tensor:
    """StreamingLLM's view of the cache for the queries of ``ctx``, ``[B, 1, Q, num_keys]`` boolean.

    A query at position ``p`` keeps the slots of positions ``0 .. sinks - 1`` and ``p - window + 1 .. p``. Slots at
    or above a draft tree's base hold tree nodes, which the layer's own mask already restricts to the query's
    ancestors, so they are kept. The mask is built on the device from the context's tensors, without host
    synchronization.
    """
    keys = ctx.cache.key_positions(device)[:num_keys]
    queries = ctx.positions.to(device)[:, :, None]  # [B, Q, 1]
    keep = (keys[None, None, :] < sinks) | (keys[None, None, :] > queries - window)
    if ctx.tree is not None:
        slots = torch.arange(num_keys, device=device)
        keep = keep | (slots[None, None, :] >= ctx.tree.base.to(device)[:, None, None])
    return keep[:, None]


def streaming_budget(ctx: LayerContext, sinks: int, window: int, device: torch.device) -> DraftBudget | None:
    """The slots of :func:`streaming_keep` as a :class:`~.views.DraftBudget`, or ``None`` for a multi-token chain.

    A chain step (one query per row at position ``p``) reads the ``sinks`` leading slots and the ``window`` slots
    from ``max(p - window + 1, sinks)`` on, ``sinks + window`` entries. A tree step with ``T`` nodes from slot
    ``base`` on reads the sinks and the ``window - 1 + T`` slots from ``max(base - window + 1, sinks)`` on, which
    hold every window a node can see and the nodes; :attr:`~.views.DraftBudget.keep` then cuts each node's window
    as :func:`streaming_keep` does. Slots past the cache map to the trash slot; those and the slots after a query
    are hidden by the layer's mask, so the attention covers the entries of :func:`streaming_keep` exactly. The
    slots are computed on the device, without host synchronization.
    """
    tree = ctx.tree
    queries = ctx.positions.to(device)
    if tree is None:
        if queries.shape[1] != 1:
            return None
        start, length = queries[:, 0] - window + 1, window
    else:
        base = tree.base.to(device)
        start, length = base - window + 1, window - 1 + tree.slots.shape[1]
    lead = torch.arange(sinks, device=device).expand(queries.shape[0], sinks)
    recent = start.clamp(min=sinks)[:, None] + torch.arange(length, device=device)
    slots = torch.cat([lead, recent], dim=-1).clamp(max=ctx.cache.num_slots - 1)  # [B, W]
    keys = ctx.cache.key_positions(device)[slots][:, None, :]  # [B, 1, W]
    keep = (keys < sinks) | (keys > queries[:, :, None] - window)
    if tree is not None:
        keep = keep | (slots[:, None, :] >= base[:, None, None])
    return DraftBudget(slots[:, None], keep[:, None])


class _SparseLayer:
    """Decoder layer ``index`` called on the path's budget through the path's view."""

    def __init__(self, path: SparseKVPath, index: int, ctx: LayerContext, shared: StepMemo) -> None:
        self.path, self.index, self.ctx, self.shared = path, index, ctx, shared

    def __call__(self, hidden: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        layer = self.path.adapter.layer(self.index)
        return self.path.view.call(self.path, self.index, layer, hidden, args, kwargs, self.ctx, self.shared)


class SparseKVPath(ABC):
    """A self-drafting path over all layers whose full-attention layers attend to a budget of their cache.

    Subclasses decide the budget, both as cache slots (:meth:`budget`, read by the gathered view) and as a mask over
    the whole cache (:meth:`keep`, read by the masked view and wherever a layer has no budget); the two describe the
    same entries. The path adds no parameters and runs every layer, so the cost log counts a full ``L``-layer step
    per draft token; what it saves is the KV traffic and attention FLOPs of the full-attention layers.

    Args:
        adapter: The model's adapter.
        view: Name of the :class:`~.views.DraftView` in :data:`~.views.DRAFT_VIEWS` (``"gathered"`` reads only the
            budget, ``"masked"`` masks the rest of the cache).
    """

    #: Whether :meth:`keep` and :meth:`budget` are the same for every layer with the same mask (computed once per
    #: step then).
    layer_independent: bool = False

    def __init__(self, adapter: ArchAdapter, view: str = "gathered") -> None:
        self._adapter = adapter
        self.layers = sparse_layers(adapter)
        self.view: DraftView = DRAFT_VIEWS.get(view)()
        self._groups: dict[int, int] = {}

    @property
    def adapter(self) -> ArchAdapter:
        return self._adapter

    @property
    def layer_span(self) -> tuple[int, int]:
        """A draft step runs every layer."""
        return 1, self._adapter.num_layers

    @property
    def footprint(self) -> LayerFootprint | None:
        return None

    @abstractmethod
    def keep(self, index: int, mask: torch.Tensor, ctx: LayerContext) -> torch.Tensor | None:
        """Keys layer ``index`` may read (boolean, broadcastable to ``[B, H_q, Q, K]`` with ``K`` the last dimension
        of ``mask``), or ``None`` to run the layer on its whole visible cache (:meth:`call_dense`)."""

    def budget(self, index: int, mask: torch.Tensor, ctx: LayerContext) -> DraftBudget | None:
        """The cache slots layer ``index`` reads in this step (the entries :meth:`keep` allows), or ``None`` to run
        the layer through :meth:`keep` on its whole cache. The default has no budget."""
        return None

    def group_size(self, index: int) -> int:
        """Query heads per KV head of layer ``index``."""
        groups = self._groups.get(index)
        if groups is None:
            value = getattr(self._adapter.attention_module(index), "num_key_value_groups", 1)
            groups = self._groups[index] = value if isinstance(value, int) and value > 0 else 1
        return groups

    def call_dense(self, index: int, layer: LayerCall, hidden: torch.Tensor, args: tuple[object, ...],
                   kwargs: dict[str, object], ctx: LayerContext) -> torch.Tensor:
        """Run full-attention layer ``index`` on its whole visible cache (when :meth:`keep` returns ``None``)."""
        return layer(hidden, *args, **kwargs)

    def run(self, runner: LayerRunner, embedding: torch.Tensor, ctx: LayerContext) -> torch.Tensor:
        """Output of the last layer for token embeddings ``[B, S, d]``, full-attention layers on the sparse view."""
        shared: StepMemo = {}
        overrides = {i: _SparseLayer(self, i, ctx, shared) for i in self.layers}
        hidden, _ = runner.run(embedding, ctx, 1, self._adapter.num_layers, overrides=overrides)
        return hidden

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._adapter.logits(hidden)
