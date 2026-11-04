"""The Kangaroo drafter: the model's shallow layers on the model's cache, then the adapter on a cache of its own."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from ...decoding.cost import CostEvent, CostKind, CostLog, LayerFootprint
from ...decoding.protocols import Draft
from ...modules import DraftKVCache
from .._drafting import BoundedDrafter, CommittedColumns, GraphedStep, ObserveCosts

if TYPE_CHECKING:
    from ...decoding.request import DraftRequest
    from ...runtime import ExactPass, KVCache, LayerRunner
    from .technique import KangarooTechnique

__all__ = ["ADAPTER_KIND", "KangarooDrafter"]

ADAPTER_KIND = "kangaroo_adapter"
"""Cost kind of the adapter's part of a draft step: one layer of the module ``"kangaroo"`` on one token."""


class KangarooDrafter(BoundedDrafter):
    """Drafts with a :class:`~economize.techniques.kangaroo.KangarooTechnique`.

    A draft step at position ``p`` runs the model's layers ``1..l`` on the token at ``p`` (their keys and values go
    into the model's cache, which the next exact pass overwrites), writes the adapter's entry of ``p`` into the
    drafter's own cache ``[B, H_kv, slots, D]`` and attends to the entries up to ``p``. After every exact pass,
    :meth:`observe` writes the entries of the committed tokens from the pass's output of layer ``l`` (no shift:
    the entry of ``p`` reads ``h^l_p``). The drafter has no per-row state besides the two caches.

    The cost log records a draft step as layers ``1..l`` of the model (:attr:`layer_span`), preceded by an
    :data:`ADAPTER_KIND` event of one layer of the module ``"kangaroo"`` for the adapter's attention layer of the same
    step (placed before the step's draft event, so that the loop's cut after a row's last step keeps it). The
    adapter's writes after every exact pass are :data:`~economize.decoding.CostKind.DRAFT_OBSERVE` events of one layer
    of the module ``"kangaroo"``.

    With CUDA graphs (:meth:`enable_graphs`) the drafter's cache lives in one persistent buffer per batch size and
    capacity, and a step attends over the whole cache (masked past the query's position), as
    :class:`~economize.techniques.eagle3.Eagle3Drafter` does.
    """

    footprint = None
    observe_footprint = LayerFootprint(1, 1, "kangaroo")

    def __init__(self, technique: KangarooTechnique, runner: LayerRunner) -> None:
        self.technique = technique
        self.runner = runner
        self.source_layers = (technique.exit_layer,)
        self._model_cache: KVCache | None = None
        self._cache: DraftKVCache | None = None
        self._capacity = 0
        self._observe_costs = ObserveCosts(self.observe_footprint)
        self._step_fn = GraphedStep(self._step)
        self._buffers: dict[tuple[int, int], DraftKVCache] = {}  # graph mode: per (batch size, capacity)

    @property
    def layer_span(self) -> tuple[int, int]:  # type: ignore[override]
        return self.technique.layer_span

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        self.runner, self._model_cache, self._capacity = runner, cache, cache.capacity
        self._cache = self._persistent(batch_size) if self._step_fn.enabled else self._new_cache(batch_size)
        self._observe_costs.reset()

    def _new_cache(self, batch_size: int) -> DraftKVCache:
        technique = self.technique
        net = technique.adapter_net
        return DraftKVCache(1, batch_size, net.num_kv_heads, net.head_dim, self._capacity, device=technique.device,
                            dtype=technique.dtype)

    def _persistent(self, batch_size: int) -> DraftKVCache:
        """The graph mode cache of ``batch_size`` rows at the current capacity (allocated once). Stale entries
        stay masked: every read stops at the query's position."""
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
            raise RuntimeError("KangarooDrafter.observe needs begin() first")
        technique = self.technique
        cols = CommittedColumns.of(exact, columns, technique.device, "KangarooDrafter")
        keys, values = technique.adapter_net.keys_values(exact.hidden[technique.exit_layer],
                                                         technique.rotary_table(cols.positions))
        self._cache.scatter(0, keys, values, self._cache.slots(cols.positions, cols.committed))
        self._observe_costs.add_columns(cols.valid, cols.positions)

    def draft_slots(self) -> int | None:
        return self._capacity

    def propose(self, request: DraftRequest) -> Draft:
        self._observe_costs.flush(request.costs)
        starts = [len(log.events) for log in request.costs]
        draft = super().propose(request)
        for log, start in zip(request.costs, starts):
            _log_adapter_steps(log, start)
        return draft

    def step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        if self._cache is None or self._model_cache is None:
            raise RuntimeError("KangarooDrafter.step needs begin() first")
        return self._step_fn(tokens, positions, valid)

    def _step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        assert self._cache is not None and self._model_cache is not None
        technique, runner = self.technique, self.runner
        ctx = runner.context(self._model_cache, positions[:, None], None if valid is None else valid[:, None],
                             trusted=True)
        shallow = technique.run(runner, runner.embed(tokens[:, None], ctx), ctx)
        device = technique.device
        positions = positions.to(device)
        limit = self._capacity if self._step_fn.enabled else (self.step_limit or self._capacity)
        net = technique.adapter_net
        residual, q, k, v = net.project(shallow, technique.rotary_table(positions[:, None]))
        live = None if valid is None else valid.to(device)
        self._cache.write(0, k[:, :, 0], v[:, :, 0], self._cache.slots(positions, live))
        query = positions.clamp(max=limit - 1)[:, None]  # stopped rows may sit past the limit
        visible = torch.arange(limit, device=device)[None, :] <= query
        attended = net.attend(q, *self._cache.read(0, limit), visible[:, None, None])
        return technique.logits(net.finish(residual, attended)[:, 0])

    def reorder(self, rows: torch.Tensor) -> None:
        self._observe_costs.reorder(rows)
        if self._cache is None:
            return
        if not self._step_fn.enabled:
            self._cache.reorder(rows)
            return
        index = rows.to(self.technique.device)
        kept = (self._cache.keys[0].index_select(0, index), self._cache.values[0].index_select(0, index))
        cache = self._persistent(len(rows))
        cache.keys[0].copy_(kept[0])
        cache.values[0].copy_(kept[1])
        self._cache = cache


_STEP_KINDS = frozenset((CostKind.DRAFT, CostKind.DRAFT_WASTED))


def _log_adapter_steps(log: CostLog, start: int) -> None:
    """Insert an :data:`ADAPTER_KIND` event before every draft step of the model logged from event ``start`` on
    (a wasted step ran its adapter too)."""
    events: list[CostEvent] = []
    for event in log.events[start:]:
        if event.kind in _STEP_KINDS and event.module is None:
            events.append(CostEvent(ADAPTER_KIND, event.tokens, event.position, 1, 1, event.step, "kangaroo"))
        events.append(event)
    log.events[start:] = events
