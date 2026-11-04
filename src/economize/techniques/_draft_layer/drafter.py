"""The drafter of a one-layer draft decoder: the layer with its own KV cache, fed by the fused exact states of the
model."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from ...decoding.cost import LayerFootprint
from ...decoding.protocols import Draft
from ...modules import DraftKVCache
from ...runtime.mask import TreeLayout
from .._drafting import (
    BoundedDrafter,
    ChainGraphs,
    CommittedColumns,
    GraphedStep,
    GreedyHead,
    LogitsGreedyHead,
    ObserveCosts,
    previous_states,
    take_rows,
)

if TYPE_CHECKING:
    from ...decoding.request import DraftRequest
    from ...runtime import ExactPass, KVCache, LayerRunner
    from .technique import DraftLayerTechnique

__all__ = ["DraftLayerDrafter"]


class DraftLayerDrafter(BoundedDrafter, ChainGraphs):
    """Drafts with the layer of a :class:`~economize.techniques._draft_layer.DraftLayerTechnique`.

    The drafter keeps the draft layer's keys and values ``[B, H_kv, slots, D]`` in a cache of its own, laid out
    like the model's (slot ``p`` holds position ``p``). After every exact pass, :meth:`observe` writes the entries
    of the committed tokens: token ``p``'s entry reads the fused exact states of token ``p - 1``. A draft step at
    position ``p`` writes its entry at ``p`` (from the previous draft step's output, or from the last exact token's
    fused states at the first step) and attends to all entries up to ``p``. The next exact pass overwrites the
    entries of rejected drafts.

    The draft layer is none of the model's decoder layers: the cost log records ``draft`` events of one layer of
    the technique's module (:attr:`footprint`; :attr:`layer_span` is ``(1, 0)``, no layer of the model), and the
    writes of every exact pass as :data:`~economize.decoding.CostKind.DRAFT_OBSERVE` events of the same module. The
    drafter drafts chains, and trees through :meth:`tree_step`.

    With CUDA graphs (:meth:`enable_graphs`) every draft step is replayed from a graph. The draft cache and the
    per-row states then live in one set of persistent buffers per batch size and capacity, updated in place, and a
    step attends over the whole cache (masked past the query's position) instead of the proposal's
    :attr:`step_limit`, so its shapes stay fixed. Eagerly, a step reads only up to :attr:`step_limit`. A greedy
    proposal of ``k`` steps can also be replayed from one graph (:class:`~economize.techniques._drafting.ChainGraphs`),
    its tokens chosen by :meth:`greedy_head`.

    Subclasses override :meth:`greedy_head` when the technique's head has a fused kernel.
    """

    layer_span = (1, 0)
    footprint: LayerFootprint

    def __init__(self, technique: DraftLayerTechnique) -> None:
        self.technique = technique
        self.source_layers = technique.source_layers
        self.footprint = technique.footprint
        self._cache: DraftKVCache | None = None
        self._source: torch.Tensor | None = None  # fused exact states of every row's last processed token
        self._state: torch.Tensor | None = None  # source of the next draft step
        self._capacity = 0
        self._observe_costs = ObserveCosts(self.footprint)
        self._step_fn = GraphedStep(self._step)
        # graph mode: (cache, source, state) per (batch size, capacity), reused across generations
        self._buffers: dict[tuple[int, int], tuple[DraftKVCache, torch.Tensor, torch.Tensor]] = {}
        self._observed = False

    @property
    def observed_layers(self) -> tuple[int, ...]:
        return tuple(sorted({0, *self.source_layers}))

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        technique = self.technique
        layer = technique.layer
        self._capacity = cache.capacity
        device, dtype = technique.device, technique.dtype  # the cache holds what the draft layer computes
        if self._step_fn.enabled:
            self._cache, self._source, state = self._persistent(batch_size)
            self._source.zero_()  # stale entries of the cache stay masked: every read stops at the query's position
            self._state = state
            self._observed = False
        else:
            self._cache = DraftKVCache(
                1, batch_size, layer.num_kv_heads, layer.head_dim, cache.capacity, device=device, dtype=dtype
            )
            self._source = torch.zeros(batch_size, technique.adapter.hidden_size, device=device, dtype=dtype)
            self._state = None
        self._observe_costs.reset()

    def _persistent(self, batch_size: int) -> tuple[DraftKVCache, torch.Tensor, torch.Tensor]:
        """The graph mode buffers of ``batch_size`` rows at the current capacity (allocated once)."""
        key = (batch_size, self._capacity)
        if key not in self._buffers:
            # a new capacity means a new model cache, whose allocation dropped every recorded graph
            self._buffers = {k: v for k, v in self._buffers.items() if k[1] == self._capacity}
            technique = self.technique
            layer, device, dtype = technique.layer, technique.device, technique.dtype
            cache = DraftKVCache(
                1, batch_size, layer.num_kv_heads, layer.head_dim, self._capacity, device=device, dtype=dtype
            )
            zeros = torch.zeros(batch_size, technique.adapter.hidden_size, device=device, dtype=dtype)
            self._buffers[key] = (cache, zeros, zeros.clone())
        return self._buffers[key]

    def enable_graphs(self, wrap: Callable[[Callable[..., Any]], Callable[..., Any]]) -> bool:
        self._step_fn.enable(wrap)
        return True

    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None:
        if self._cache is None or self._source is None:
            raise RuntimeError(f"{type(self).__name__}.observe needs begin() first")
        technique = self.technique
        cols = CommittedColumns.of(exact, columns, technique.device, type(self).__name__)
        fused = technique.fuse([exact.hidden[layer] for layer in self.source_layers])  # [B, S, d]
        source = previous_states(fused, cols.valid, self._source)
        keys, values = technique.layer.keys_values(source, exact.hidden[0], technique.rotary_table(cols.positions))
        self._cache.scatter(0, keys, values, self._cache.slots(cols.positions, cols.committed))
        self._observe_costs.add_columns(cols.valid, cols.positions)
        if self._step_fn.enabled:
            assert self._state is not None
            self._source.copy_(cols.last(fused))
            self._state.copy_(self._source)
            self._observed = True
        else:
            self._source = cols.last(fused)
            self._state = self._source

    def draft_slots(self) -> int | None:
        return self._capacity

    def propose(self, request: DraftRequest) -> Draft:
        self._observe_costs.flush(request.costs)
        return super().propose(request)

    def step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        if self._cache is None or self._state is None or (self._step_fn.enabled and not self._observed):
            raise RuntimeError(f"{type(self).__name__}.step needs begin() and observe() first")
        return self._step_fn(tokens, positions, valid)

    # ---- dynamic trees (economize.decoding.dynamic_tree) ----------------------------------------------------
    def supports_tree_steps(self) -> bool:
        return True

    def tree_begin(self, request: DraftRequest) -> None:
        self._observe_costs.flush(request.costs)
        if self._cache is None or self._state is None or (self._step_fn.enabled and not self._observed):
            raise RuntimeError(f"{type(self).__name__} needs begin() and observe() before drafting a tree")

    def tree_state(self) -> tuple[torch.Tensor, ...]:
        assert self._state is not None
        return (self._state,)

    def tree_step(self, tokens: torch.Tensor, parents: torch.Tensor, base: torch.Tensor,
                  state: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """One draft step over the frontier of a tree (see :class:`~economize.decoding.TreeStepDrafter`): every
        frontier node reads its parent's draft layer output ``state`` ``[B, W, d]``, writes its entry at its slot and
        attends to the committed prefix and its ancestors."""
        assert self._cache is not None
        technique = self.technique
        layer, device = technique.layer, technique.device
        parents, base = parents.to(device), base.to(device)
        batch, width = tokens.shape
        n = parents.shape[1]
        tree = TreeLayout.build(parents, base)
        slots, positions = tree.slots[:, n - width:], tree.positions[:, n - width:]
        residual, q, k, v = layer.project(state, technique.embed(tokens), technique.rotary_table(positions))
        self._cache.scatter(0, k, v, slots)
        limit = self._capacity
        keys = torch.arange(limit, device=device)[None, :]
        prefix = (keys < base[:, None]) & (keys >= technique.first_visible())
        visible = prefix[:, None].expand(batch, width, limit).scatter(
            2, tree.slots[:, None].expand(batch, width, n), tree.visible[:, n - width:])
        out = layer.finish(residual, layer.attend(q, *self._cache.read(0, limit), visible[:, None]))
        return technique.draft_logits(out), (out,)

    def _chain_ready(self) -> bool:
        return self._cache is not None and self._state is not None and self._observed

    def greedy_head(self, fused: bool = True) -> GreedyHead:
        """The greedy choice from the technique's :meth:`~DraftLayerTechnique.draft_logits` of a step's output
        (subclasses return a fused kernel when ``fused`` and the head allows one)."""
        return LogitsGreedyHead(self.technique.draft_logits)

    def _step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        return self.technique.draft_logits(self._advance(tokens, positions, valid))

    def _advance(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        """A draft step up to the draft layer's output ``[B, d]``, with the step's cache and state updates."""
        assert self._cache is not None and self._state is not None
        technique = self.technique
        layer, device = technique.layer, technique.device
        positions = positions.to(device)
        graphed = self._step_fn.enabled
        limit = self._capacity if graphed else (self.step_limit or self._capacity)
        residual, q, k, v = layer.project(self._state[:, None], technique.embed(tokens[:, None]),
                                          technique.rotary_table(positions[:, None]))
        live = None if valid is None else valid.to(device)
        self._cache.write(0, k[:, :, 0], v[:, :, 0], self._cache.slots(positions, live))
        query = positions.clamp(max=limit - 1)[:, None]  # stopped rows may sit past the limit
        keys = torch.arange(limit, device=device)[None, :]
        visible = (keys <= query) & ((keys >= technique.first_visible()) | (keys == query))
        attended = layer.attend(q, *self._cache.read(0, limit), visible[:, None, None])
        out = layer.finish(residual, attended)[:, 0]
        state = out if live is None else torch.where(live[:, None], out, self._state)
        if graphed:
            self._state.copy_(state)
        else:
            self._state = state
        return out

    def reorder(self, rows: torch.Tensor) -> None:
        self._observe_costs.reorder(rows)
        if not self._step_fn.enabled:
            if self._cache is not None:
                self._cache.reorder(rows)
            self._source, self._state = take_rows(self._source, rows), take_rows(self._state, rows)
            return
        if self._cache is None or self._source is None or self._state is None:
            return
        on_device = rows.to(self.technique.device)
        kept = [self._cache.keys[0].index_select(0, on_device), self._cache.values[0].index_select(0, on_device),
                self._source.index_select(0, rows.to(self._source.device)),
                self._state.index_select(0, rows.to(self._state.device))]
        cache, source, state = self._persistent(len(rows))
        for buffer, value in zip((cache.keys[0], cache.values[0], source, state), kept):
            buffer.copy_(value)
        self._cache, self._source, self._state = cache, source, state
