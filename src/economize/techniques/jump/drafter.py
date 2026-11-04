"""The drafter a :class:`JumpTechnique` hands to the decoding loop."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from ...decoding.cost import LayerFootprint
from ...decoding.protocols import Draft
from .._drafting import (
    AdapterHeadSlicing,
    BoundedDrafter,
    ChainGraphs,
    CommittedColumns,
    GraphedStep,
    ObserveCosts,
    take_rows,
    tree_step_context,
)
from .predictor import CachedPredictor

if TYPE_CHECKING:
    from ...decoding.request import DraftRequest
    from ...modules import DraftKVCache
    from ...runtime import ExactPass, KVCache, LayerRunner
    from .technique import JumpTechnique

__all__ = ["JumpDrafter"]


class JumpDrafter(AdapterHeadSlicing, BoundedDrafter, ChainGraphs):
    """Jump drafting for a batch of rows.

    After each exact pass the drafter adopts, for every row, the exact states of the last accepted token at the
    source layers. A draft step embeds the row's last token, predicts its state after layer ``m``, runs layers
    ``m+1..r`` at the token's position (writing their K/V into the shared cache) and reads out the next-token
    logits. It then replaces the states of the source layers at or above ``m`` with its own. Source layers below
    ``m`` are never recomputed on the jump path and keep their exact states until the next exact pass.

    Every step is one batched call per layer with fixed shapes ``[B, 1, d]`` and no host synchronization. Rows that
    stopped drafting run on the cache's trash slot and leave the cache unchanged. With CUDA graphs
    (:meth:`enable_graphs`) the states live in one set of persistent buffers per batch size, updated in place, and
    every step is replayed from a graph; a greedy proposal of ``k`` steps can also be replayed from one graph
    (:class:`~economize.techniques._drafting.ChainGraphs`).

    A :class:`~economize.techniques.jump.CachedPredictor` (one that attends to earlier positions) keeps its memory
    entries in a cache of its own: :meth:`observe` writes the entries of the committed tokens from their exact states
    after the source layer ``u``, and every draft step at position ``p`` writes the entry of ``p`` from the step's
    own state after ``u``. Such a drafter runs eagerly (no CUDA graphs), and the cost log records the memory writes of
    every exact pass as :data:`~economize.decoding.CostKind.DRAFT_OBSERVE` events of the module ``"jump_predictor"``.
    """

    def __init__(self, technique: JumpTechnique, runner: LayerRunner) -> None:
        self.technique = technique
        self.runner = runner
        self.source_layers = tuple(technique.source_layers)
        self.layer_span = technique.layer_span
        self._updated = technique.updated_layers
        self._captured = tuple(layer for layer in self._updated if layer > technique.target_layer)
        self._overrides = technique.layer_overrides()
        self._cache: KVCache | None = None
        self._states: dict[int, torch.Tensor] | None = None
        self._step_fn = GraphedStep(self._step)
        self._buffers: dict[int, dict[int, torch.Tensor]] = {}  # graph mode: persistent states per batch size
        predictor = technique.predictor
        self._predictor: CachedPredictor | None = predictor if isinstance(predictor, CachedPredictor) else None
        self._memory: DraftKVCache | None = None  # the predictor's entries of earlier positions
        self._observe_costs: ObserveCosts | None = None
        if self._predictor is not None:
            footprint = LayerFootprint(1, self._predictor.memory_layers, "jump_predictor")
            self._observe_costs = ObserveCosts(footprint)
        self._head: Callable[[torch.Tensor], torch.Tensor] | None = None

    @property
    def observed_layers(self) -> tuple[int, ...]:
        """The source layers plus the target layer ``m`` (whose exact states verify observers receive, e.g. for
        on-policy training of the predictor), and with a :class:`CachedPredictor` the embedding (layer ``0``) for
        its memory entries."""
        memory = (0, self.technique.config.source_layer) if self._predictor is not None else ()
        return tuple(sorted({*self.source_layers, self.technique.target_layer, *memory}))

    @property
    def streams_prompt(self) -> bool:  # type: ignore[override]
        """Without a :class:`CachedPredictor` the drafter reads the last accepted token's states alone; a cached
        predictor fills its memory from every prompt position."""
        return self._predictor is None

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        self.runner, self._cache = runner, cache
        self._states = self._buffers.setdefault(batch_size, {}) if self._step_fn.enabled else None
        if self._predictor is not None:
            self._memory = self._predictor.new_cache(batch_size, cache.capacity, self.technique.adapter.device)
        if self._observe_costs is not None:
            self._observe_costs.reset()

    def enable_graphs(self, wrap: Callable[[Callable[..., Any]], Callable[..., Any]]) -> bool:
        if self._predictor is not None:  # its cache is reallocated by reorder and read up to a varying limit
            return False
        self._step_fn.enable(wrap)
        return True

    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None:
        device = self.technique.adapter.device
        states = self._buffers.setdefault(len(columns), {}) if self._step_fn.enabled else {}
        for layer in self.source_layers:
            hidden = exact.hidden[layer]
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            self._put(states, layer, hidden[rows, columns.to(hidden.device)].to(device))
        self._states = states
        if self._predictor is not None:
            self._observe_memory(exact, columns)

    def _observe_memory(self, exact: ExactPass, columns: torch.Tensor) -> None:
        """Write the memory entries of the committed tokens of the exact pass (columns ``<= columns`` per row)."""
        assert self._predictor is not None and self._observe_costs is not None
        if self._memory is None:
            raise RuntimeError("JumpDrafter.observe needs begin() first")
        device = self.technique.adapter.device
        cols = CommittedColumns.of(exact, columns, device, "JumpDrafter")
        states = exact.hidden[self.technique.config.source_layer].to(device)
        slots = self._memory.slots(cols.positions, cols.committed)
        with torch.no_grad():
            self._predictor.prefill(self._memory, states, exact.hidden[0].to(device), cols.positions, slots)
        self._observe_costs.add_columns(cols.valid, cols.positions)

    def draft_slots(self) -> int | None:
        return None if self._memory is None else self._memory.capacity

    def propose(self, request: DraftRequest) -> Draft:
        if self._observe_costs is not None:
            self._observe_costs.flush(request.costs)
        return super().propose(request)

    def step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        if self._cache is None or self._states is None:
            raise RuntimeError("JumpDrafter.step needs begin() and observe() first")
        return self._step_fn(tokens, positions, valid)

    # ---- dynamic trees (economize.decoding.dynamic_tree) ----------------------------------------------------
    def supports_tree_steps(self) -> bool:
        return self._predictor is None  # a cached predictor's memory holds one entry per position, not per node

    def tree_begin(self, request: DraftRequest) -> None:
        if self._cache is None or self._states is None:
            raise RuntimeError("JumpDrafter needs begin() and observe() before drafting a tree")

    def tree_state(self) -> tuple[torch.Tensor, ...]:
        assert self._states is not None
        return tuple(self._states[layer] for layer in self.source_layers)

    def tree_step(self, tokens: torch.Tensor, parents: torch.Tensor, base: torch.Tensor,
                  *states: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """One jump step over the frontier of a tree (see :class:`~economize.decoding.TreeStepDrafter`): every
        frontier node reads its parent's source states ``[B, W, d]``, and layers ``m+1..r`` write its K/V at its slot
        and attend to the committed prefix and its ancestors."""
        assert self._cache is not None
        t = self.technique
        m, r = t.target_layer, t.exit_layer
        ctx = tree_step_context(self.runner, self._cache, parents, base, tokens.shape[1])
        embedding = self.runner.embed(tokens, ctx)
        current = dict(zip(self.source_layers, states))
        x = t.predict([current[layer] for layer in self.source_layers], embedding)
        if m in self._updated and m in current:
            current[m] = x
        with t.jump_path():
            hidden, captured = self.runner.run(x, ctx, m + 1, r, overrides=self._overrides, capture=self._captured)
        for layer, state in captured.items():
            if layer in current:
                current[layer] = state.to(x.device)
        return self._tree_logits(hidden), tuple(current[layer] for layer in self.source_layers)

    def _chain_ready(self) -> bool:
        return self._cache is not None and self._states is not None

    def _step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        return self._draft_logits(self._advance(tokens, positions, valid))

    def _advance(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        """A draft step up to the exit layer's output ``[B, 1, d]``, with the step's state and cache updates."""
        assert self._cache is not None and self._states is not None
        t, states = self.technique, self._states
        m, r = t.target_layer, t.exit_layer
        ctx = self.runner.context(self._cache, positions[:, None], None if valid is None else valid[:, None],
                                  trusted=True)
        embedding = self.runner.embed(tokens[:, None], ctx)
        sources = [states[layer][:, None] for layer in self.source_layers]
        memory = self._memory
        if memory is None:
            x = t.predict(sources, embedding)
        else:
            x = t.predict_step(memory, sources, embedding, positions[:, None], limit=self.step_limit)
        if m in self._updated:
            self._put(states, m, x[:, 0])
        with t.jump_path():
            hidden, captured = self.runner.run(x, ctx, m + 1, r, overrides=self._overrides, capture=self._captured)
        for layer, state in captured.items():
            self._put(states, layer, state[:, 0].to(x.device))
        if memory is not None:  # the entry of this step's position, from the step's own state after u
            assert self._predictor is not None
            slots = memory.slots(positions, valid)[:, None]
            state = states[t.config.source_layer][:, None]
            self._predictor.prefill(memory, state, embedding, positions[:, None], slots)
        return hidden

    def reorder(self, rows: torch.Tensor) -> None:
        if self._predictor is not None and self._memory is not None:
            self._predictor.reorder(self._memory, rows)
        if self._observe_costs is not None:
            self._observe_costs.reorder(rows)
        if self._states is None:
            return
        target = self._buffers.setdefault(len(rows), {}) if self._step_fn.enabled else {}
        for layer, s in self._states.items():
            self._put(target, layer, take_rows(s, rows))  # type: ignore[arg-type]
        self._states = target

    def _put(self, states: dict[int, torch.Tensor], layer: int, value: torch.Tensor) -> None:
        """``states[layer] = value``; in graph mode copied into the existing buffer of the same shape, so that the
        graphs that read or write the buffer stay valid."""
        buffer = states.get(layer)
        if self._step_fn.enabled and buffer is not None and buffer.shape == value.shape:
            buffer.copy_(value)
        else:
            states[layer] = value
