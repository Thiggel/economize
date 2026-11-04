"""The draft path of sparse-KV self-speculation: the whole model, with full-attention layers reading part of the cache.

A :class:`SparseKVPath` is a :class:`~economize.techniques.early_exit.SelfDraftStep`: a draft step runs layers
``1..L`` with the model's own weights, and every full-attention layer is replaced by a stand-in that intersects the
layer's attention mask with the path's :meth:`~SparseKVPath.keep` mask before calling the layer. The cache is the
model's own, so the draft path writes the K/V of the drafted positions where the next exact pass overwrites them.
The exact pass never runs through the path and reads the whole cache, which keeps decoding lossless.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

from ...arch.spec import StateKind
from ...errors import UnsupportedModelError

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...decoding.cost import LayerFootprint
    from ...runtime import LayerContext, LayerRunner

__all__ = ["SparseKVPath", "restrict", "sparse_layers", "streaming_keep"]


def sparse_layers(adapter: ArchAdapter) -> tuple[int, ...]:
    """The layers (1-indexed) whose cache a sparse draft path reads in part: the full-attention layers.

    Sliding-window and chunked layers already read a bounded part of the cache, and recurrent layers keep no
    per-position cache, so they run unchanged.
    """
    return tuple(
        i for i in range(1, adapter.num_layers + 1)
        if adapter.attention_spec(i).kind == "full" and adapter.attention_spec(i).state_kind is StateKind.KV
    )


def restrict(mask: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """``mask`` (a boolean ``sdpa`` mask or an additive ``eager`` mask, ``[B, 1, Q, K]``) hiding every key that
    ``keep`` (boolean, broadcastable to ``[B, H, Q, K]``) drops; per-head ``keep`` gives a per-head mask."""
    if mask.dtype == torch.bool:
        return mask & keep
    return mask.masked_fill(~keep, torch.finfo(mask.dtype).min)


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


class _SparseLayer:
    """Decoder layer ``index`` called with its attention mask restricted by the path's keep mask."""

    def __init__(self, path: SparseKVPath, index: int, ctx: LayerContext, shared: dict[int, torch.Tensor]) -> None:
        self.path, self.index, self.ctx, self.shared = path, index, ctx, shared

    def __call__(self, hidden: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        layer = self.path.adapter.layer(self.index)
        mask = kwargs.get("attention_mask")
        if not isinstance(mask, torch.Tensor):
            raise UnsupportedModelError(f"{type(self.path).__name__} restricts tensor attention masks; layer "
                                        f"{self.index} got {type(mask).__name__}")
        restricted = self.shared.get(id(mask)) if self.path.layer_independent else None
        if restricted is None:
            keep = self.path.keep(self.index, mask, self.ctx)
            if keep is None:
                return self.path.call_dense(self.index, layer, hidden, args, kwargs, self.ctx)
            restricted = restrict(mask, keep)
            if self.path.layer_independent:
                self.shared[id(mask)] = restricted
        kwargs["attention_mask"] = restricted
        return layer(hidden, *args, **kwargs)


class SparseKVPath(ABC):
    """A self-drafting path over all layers whose full-attention layers attend to the keys of :meth:`keep`.

    Subclasses decide the keep mask. The path adds no parameters and runs every layer, so the cost log counts a
    full ``L``-layer step per draft token; what it saves is the KV traffic of the full-attention layers, which
    only a kernel that skips hidden keys realizes (economize's decode kernel skips every block of keys the mask
    hides).

    Args:
        adapter: The model's adapter.
    """

    #: Whether :meth:`keep` is the same for every layer with the same mask (it is then computed once per step).
    layer_independent: bool = False

    def __init__(self, adapter: ArchAdapter) -> None:
        self._adapter = adapter
        self.layers = sparse_layers(adapter)

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

    def call_dense(self, index: int, layer: Any, hidden: torch.Tensor, args: tuple[Any, ...],
                   kwargs: dict[str, Any], ctx: LayerContext) -> Any:
        """Run full-attention layer ``index`` on its whole visible cache (when :meth:`keep` returns ``None``)."""
        return layer(hidden, *args, **kwargs)

    def run(self, runner: LayerRunner, embedding: torch.Tensor, ctx: LayerContext) -> torch.Tensor:
        """Output of the last layer for token embeddings ``[B, S, d]``, full-attention layers on the sparse view."""
        shared: dict[int, torch.Tensor] = {}
        overrides = {i: _SparseLayer(self, i, ctx, shared) for i in self.layers}
        hidden, _ = runner.run(embedding, ctx, 1, self._adapter.num_layers, overrides=overrides)
        return hidden

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._adapter.logits(hidden)
