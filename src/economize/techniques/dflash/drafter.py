"""The DFlash drafter: one forward of the draft layers over ``[anchor, mask, ..., mask]`` drafts a whole block."""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from ...decoding.cost import LayerFootprint
from ...decoding.protocols import Draft, Drafter, draft_footprint
from ...modules import DraftKVCache
from .._drafting import CommittedColumns, GraphedStep, ObserveCosts, take_rows
from ..draft_model.drafter import CATCHUP
from .layer import DFlashLayer

if TYPE_CHECKING:
    from ...decoding.request import DraftRequest
    from ...runtime import ExactPass, KVCache, LayerRunner
    from .technique import DFlashTechnique

__all__ = ["DFlashDrafter"]


class DFlashDrafter(Drafter):
    """Drafts up to ``block_size - 1`` tokens per cycle with one forward of a
    :class:`~economize.techniques.dflash.DFlashTechnique`.

    The drafter keeps every draft layer's context keys and values ``[B, H_kv, slots, D]`` in a cache of its own,
    laid out like the model's (slot ``p`` holds position ``p``). After every exact pass, :meth:`observe` fuses the
    model's states of the committed tokens and writes their entries. A proposal for a row whose last token ``x_p``
    the model emitted at position ``p`` runs the draft layers over the block ``[x_p, mask, ..., mask]`` at the
    positions ``p .. p + block_size - 1``, attending to the context entries before ``p`` and to the block, and
    drafts the token at ``p + i`` from the output at block index ``i``. The block's own keys and values are not
    kept (the reference crops its draft cache back to ``p`` after every draft). A cycle drafts
    ``n = min(budget, block_size - 1)`` tokens; the block always runs at its full width, as in training, since the
    bidirectional layers let every mask attend to the masks after it.

    Cost log: the ``n`` drafts are ``"draft"`` events of one token through all draft layers of the module
    ``"dflash"`` (:attr:`footprint`), the anchor and the ``block_size - 1 - n`` unused masks one ``"draft_catchup"``
    event, and the context writes of every exact pass ``"draft_observe"`` events of the same module.

    With CUDA graphs (:meth:`enable_graphs`) the forward is replayed from one graph per batch size and ``n``. The
    draft cache then lives in one persistent buffer per batch size and capacity, and the block attends over the
    whole cache (masked from ``p`` on) instead of the positions before the batch's last anchor, so its shapes stay
    fixed.
    """

    layer_span = (1, 0)

    def __init__(self, technique: DFlashTechnique) -> None:
        self.technique = technique
        self.source_layers = technique.target_layers
        self.footprint = LayerFootprint(1, len(technique.layers), "dflash")
        self._cache: DraftKVCache | None = None
        self._capacity = 0
        self._limit = 1  # eager mode: context slots a proposal reads (the batch's largest anchor position)
        self._observe_costs = ObserveCosts(self.footprint)
        self._step_fn = GraphedStep(self._full_block)
        self._graphs: dict[int, Callable[..., Any]] = {}  # graph mode: one replay per number of drafts
        self._buffers: dict[tuple[int, int], DraftKVCache] = {}  # graph mode: cache per (batch size, capacity)
        self._observed = False

    @property
    def observed_layers(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.source_layers)))

    def draft_capacity(self, cycle_length: int) -> int:
        return max(0, min(cycle_length - 1, self.technique.block_size - 1))

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        self._capacity = cache.capacity
        if self._step_fn.enabled:
            self._cache = self._persistent(batch_size)  # stale entries stay masked: a block reads slots < p only
        else:
            self._cache = self._new_cache(batch_size)
        self._observed = False
        self._observe_costs.reset()

    def _new_cache(self, batch_size: int) -> DraftKVCache:
        technique = self.technique
        layer = technique.layers[0]
        assert isinstance(layer, DFlashLayer)
        return DraftKVCache(len(technique.layers), batch_size, layer.num_kv_heads, layer.head_dim, self._capacity,
                            device=technique.device, dtype=technique.dtype)

    def _persistent(self, batch_size: int) -> DraftKVCache:
        """The graph mode cache of ``batch_size`` rows at the current capacity (allocated once)."""
        key = (batch_size, self._capacity)
        if key not in self._buffers:
            # a new capacity means a new model cache, whose allocation dropped every recorded graph
            self._buffers = {k: v for k, v in self._buffers.items() if k[1] == self._capacity}
            self._buffers[key] = self._new_cache(batch_size)
        return self._buffers[key]

    def enable_graphs(self, wrap: Callable[[Callable[..., Any]], Callable[..., Any]]) -> bool:
        self._step_fn.enable(wrap)
        return True

    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None:
        if self._cache is None:
            raise RuntimeError("DFlashDrafter.observe needs begin() first")
        technique = self.technique
        cols = CommittedColumns.of(exact, columns, technique.device, "DFlashDrafter")
        context = technique.fuse([exact.hidden[layer] for layer in self.source_layers])  # [B, S, d]
        rotary = technique.rotary_table(cols.positions)
        slots = self._cache.slots(cols.positions, cols.committed)
        for index, layer in enumerate(technique.layers):
            assert isinstance(layer, DFlashLayer)
            keys, values = layer.context_kv(context, rotary)
            self._cache.scatter(index, keys, values, slots)
        self._observe_costs.add_columns(cols.valid, cols.positions)
        self._observed = True

    def propose(self, request: DraftRequest) -> Draft:
        if self._cache is None or not self._observed:
            raise RuntimeError("DFlashDrafter.propose needs begin() and observe() first")
        self._observe_costs.flush(request.costs)
        if not request.drafting:
            return request.draft()
        active = request.active
        width = self.technique.block_size
        drafts = min(max(b for b, on in zip(request.budgets, active) if on), width - 1)
        anchors = [end - 1 for end in request.ends]
        for b, on in enumerate(active):
            if on:
                request.costs[b].record_footprint(CATCHUP, width - drafts, anchors[b], draft_footprint(self))
        tokens, positions = request.tokens, request.positions
        if not self._step_fn.enabled:
            self._limit = max(1, min(max(a for a, on in zip(anchors, active) if on), self._capacity))
            logits = self._block(tokens, positions, drafts)
        elif drafts == width - 1:
            logits = self._step_fn(tokens, positions, None)
        else:
            if drafts not in self._graphs:
                block = weakref.WeakMethod(self._block)  # as GraphedStep holds the step: no reference cycle
                self._graphs[drafts] = self._step_fn.graph(
                    lambda tokens, positions, n=drafts: block()(tokens, positions, n)  # type: ignore[misc]
                )
            logits = self._graphs[drafts](tokens, positions)
        for i in range(drafts):
            if not request.drafting:
                break
            request.extend(logits[:, i])
        return request.draft()

    def _full_block(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        return self._block(tokens, positions, self.technique.block_size - 1)

    def _block(self, tokens: torch.Tensor, positions: torch.Tensor, drafts: int) -> torch.Tensor:
        """Float logits ``[B, drafts, V]`` of the block anchored at ``tokens`` ``[B]`` at ``positions`` ``[B]``:
        entry ``i`` drafts the token at ``positions + i + 1``."""
        assert self._cache is not None
        technique = self.technique
        device, width = technique.device, technique.block_size
        anchor = positions.to(device)
        offsets = torch.arange(width, device=device)
        block_positions = anchor[:, None] + offsets  # [B, W]
        ids = torch.full((anchor.shape[0], width), technique.config.mask_token_id, dtype=torch.long, device=device)
        ids[:, 0] = tokens.to(device)
        hidden = technique.embed(ids)
        rotary = technique.rotary_table(block_positions)
        limit = self._capacity if self._step_fn.enabled else self._limit
        masks: dict[tuple[bool, bool], torch.Tensor] = {}
        for index, layer in enumerate(technique.layers):
            kind = (technique.sliding(index), technique.causal(index))
            if kind not in masks:
                masks[kind] = self._visible(anchor, block_positions, limit, *kind)
            keys, values = self._cache.read(index, limit)
            hidden = layer(hidden, rotary, keys, values, masks[kind])
        return technique.draft_logits(hidden[:, 1:drafts + 1])

    def _visible(self, anchor: torch.Tensor, block_positions: torch.Tensor, limit: int, sliding: bool,
                 causal: bool) -> torch.Tensor:
        """Bool mask ``[B, 1, W, limit + W]`` over the context slots and the block (the reference's
        ``_attention_mask`` on absolute positions)."""
        device, width = anchor.device, block_positions.shape[1]
        query = block_positions[:, :, None]  # [B, W, 1]
        slots = torch.arange(limit, device=device)[None, None, :]
        context = (slots < anchor[:, None, None]).expand(-1, width, -1)
        index = torch.arange(width, device=device)
        block = torch.ones(width, width, dtype=torch.bool, device=device)
        if causal:
            block = index[None, :] <= index[:, None]
        if sliding:
            window = self.technique.config.sliding_window
            context = context & (query - slots < window)
            block = block & ((index[:, None] - index[None, :]).abs() < window)
        block = block[None].expand(anchor.shape[0], -1, -1)
        return torch.cat([context, block], 2)[:, None]

    def reorder(self, rows: torch.Tensor) -> None:
        self._observe_costs.reorder(rows)
        if self._cache is None:
            return
        if not self._step_fn.enabled:
            self._cache.reorder(rows)
            return
        old, index = self._cache, rows.to(self.technique.device)
        kept = [(take_rows(k, index), take_rows(v, index)) for k, v in zip(old.keys, old.values)]
        cache = self._persistent(len(rows))
        for layer, (keys, values) in enumerate(kept):
            cache.keys[layer].copy_(keys)
            cache.values[layer].copy_(values)
        self._cache = cache
