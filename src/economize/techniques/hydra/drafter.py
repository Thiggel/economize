"""The Hydra drafter: sequential heads on the base state of the last exact token, as a chain or a static tree."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ...decoding.cost import CostKind, LayerFootprint
from ...decoding.protocols import Draft, Drafter
from ...modules import DraftKVCache
from ...runtime.transfer import device_tensor
from .._drafting import CommittedColumns, ObserveCosts, take_rows

if TYPE_CHECKING:
    from ...decoding.protocols import Verifier
    from ...decoding.request import DraftRequest
    from ...runtime import ExactPass, KVCache, LayerRunner
    from ..medusa.trees import TreePlan
    from .technique import HydraTechnique

__all__ = ["HydraDrafter"]


@dataclass(frozen=True)
class _Level:
    """The head calls of one tree level ``j``: head ``j`` runs once per parent node of depth ``j - 1``.

    Attributes:
        parents: Parent node of every call ``[U]`` (``-1``: the root, the committed token).
        ancestors: Nodes on the path of every parent below the root ``[U, j - 1]`` (the head's drafted inputs).
        nodes: First and one past the last node of depth ``j`` (nodes are ordered by depth).
        calls: Call (index into :attr:`parents`) whose output every node of depth ``j`` reads ``[n]``.
        ranks: Rank of every node of depth ``j`` among its call's candidates ``[n]``.
    """

    parents: torch.Tensor
    ancestors: torch.Tensor
    nodes: tuple[int, int]
    calls: torch.Tensor
    ranks: torch.Tensor


class _TreeLayout:
    """The levels of a :class:`~economize.techniques.medusa.TreePlan` cut at ``depth``, and which exact-pass columns
    of the tree pass lie on the path to every column (``[n + 1, n + 1]``, column ``0`` is the root)."""

    def __init__(self, plan: TreePlan, depth: int, device: torch.device) -> None:
        _, ranks, parents = plan.layout(depth)
        size = len(parents)
        paths: list[list[int]] = []
        for parent in parents:
            paths.append([] if parent < 0 else paths[parent] + [parent])
        self.levels: list[_Level] = []
        start = 0
        for level in range(1, depth + 1):
            end = start + sum(1 for p in paths[start:] if len(p) == level - 1)
            calls = list(dict.fromkeys(parents[start:end]))  # parents in order of first child
            self.levels.append(_Level(
                parents=torch.tensor(calls, dtype=torch.long, device=device),
                ancestors=torch.tensor([[] if p < 0 else paths[p] + [p] for p in calls], dtype=torch.long,
                                       device=device).view(len(calls), level - 1),
                nodes=(start, end),
                calls=torch.tensor([calls.index(p) for p in parents[start:end]], dtype=torch.long, device=device),
                ranks=torch.tensor(ranks[start:end], dtype=torch.long, device=device),
            ))
            start = end
        on_path = torch.eye(size + 1, dtype=torch.bool)
        on_path[:, 0] = True
        for node, path in enumerate(paths):
            on_path[node + 1, [p + 1 for p in path]] = True
        self.on_path = on_path.to(device)
        self.parents = torch.tensor(parents, dtype=torch.long, device=device)


class HydraDrafter(Drafter):
    """Drafts with the heads of a :class:`~economize.techniques.hydra.HydraTechnique`.

    After every exact pass :meth:`observe` keeps the base state ``P_t`` of every row's last processed token ``t``:
    the model's last hidden state there, or with the prefix layer (Hydra++) that layer's output at ``t``. The prefix
    layer keeps the keys and values of every committed token in a cache of its own (slot ``p`` holds position ``p``);
    after a tree pass only the accepted path is written. A chain draft runs head ``j`` at step ``j`` on ``P_t`` and the
    tokens so far (the committed token ``x_{t+1}`` and the drafts of steps ``1..j - 1``), through
    :meth:`DraftRequest.extend`, so the selector, the draft-length policy and rejection sampling see every head's
    distribution. A tree draft runs head ``j`` once per node of depth ``j - 1`` on that node's path, and node
    ``(r_1..r_j)`` is the rank-``r_j`` candidate of its parent's call (deterministic drafts, ``probs=None``).

    The cost log records one zero-layer ``draft`` event of the network ``"hydra"`` per head that runs, per row (one
    event per tree level), and the prefix layer's pass over every exact pass's tokens as
    :data:`~economize.decoding.CostKind.DRAFT_OBSERVE` events of one layer of the module ``"hydra"``. Drafts beyond
    ``K`` heads are never made: cycles longer than ``K + 1`` end early.

    With CUDA graphs (:meth:`enable_graphs`) every head call is replayed from a graph (one per head and batch of
    calls); the prefix layer runs eagerly in :meth:`observe`.
    """

    layer_span = (1, 0)
    footprint = LayerFootprint.none("hydra")

    def __init__(self, technique: HydraTechnique) -> None:
        self.technique = technique
        self.source_layers = (technique.adapter.num_layers,)
        self._head: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = technique.head_logits
        self._state: torch.Tensor | None = None  # base state P_t of every row
        self._cache: DraftKVCache | None = None  # the prefix layer's keys and values
        self._tree_depth: int | None = None  # depth of the tree whose exact pass observe() sees next
        self._layouts: dict[int, _TreeLayout] = {}
        self._observe_costs = ObserveCosts(LayerFootprint(1, 1, "hydra"))

    @property
    def observed_layers(self) -> tuple[int, ...]:
        return ()  # the heads and the prefix layer read the exact pass's final output, which every pass returns

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        technique = self.technique
        self._state, self._tree_depth = None, None
        self._observe_costs.reset()
        prefix = technique.prefix
        self._cache = None if prefix is None else DraftKVCache(
            1, batch_size, prefix.num_kv_heads, prefix.head_dim, cache.capacity, device=technique.device,
            dtype=technique.dtype,
        )

    def enable_graphs(self, wrap: Callable[[Callable[..., Any]], Callable[..., Any]]) -> bool:
        self._head = wrap(self.technique.head_logits)
        return True

    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None:
        technique = self.technique
        prefix = technique.prefix
        depth, self._tree_depth = self._tree_depth, None
        if prefix is None:
            final = exact.final
            rows = torch.arange(final.shape[0], device=final.device)
            self._state = technique.base(final[rows, columns.to(final.device)])
            return
        if self._cache is None:
            raise RuntimeError("HydraDrafter.observe needs begin() first")
        cols = CommittedColumns.of(exact, columns, technique.device, "HydraDrafter")
        committed = cols.committed
        if depth is not None:  # a tree pass: siblings of the accepted path were rejected
            committed = cols.valid & self._layout(depth).on_path[cols.columns]
        base = technique.base(exact.final)
        keys, values = prefix.keys_values(base, technique.rotary_table(cols.positions))
        self._cache.scatter(0, keys, values, self._cache.slots(cols.positions, committed))
        self._observe_costs.add_columns(cols.valid, cols.positions)
        last, position = cols.last(base)[:, None], cols.last(cols.positions)[:, None]
        attended = prefix.attend(prefix.query(last, technique.rotary_table(position)), *self._cache.read(0),
                                 self._cache.visible(position)[:, None])
        self._state = prefix.finish(last, attended)[:, 0]

    def reorder(self, rows: torch.Tensor) -> None:
        self._observe_costs.reorder(rows)
        self._state = take_rows(self._state, rows)
        if self._cache is not None:
            self._cache.reorder(rows)

    def draft_capacity(self, cycle_length: int) -> int:
        chain = max(0, min(cycle_length - 1, self.technique.num_heads))
        plan = self.technique.plan
        return chain if plan is None else max(chain, plan.size(min(cycle_length - 1, plan.depth)))

    def drafts_trees(self, verifier: Verifier) -> bool:
        return self.technique.plan is not None and verifier.supports_trees

    def propose(self, request: DraftRequest) -> Draft:
        if self._state is None:
            raise RuntimeError("HydraDrafter.propose needs observe() first")
        self._observe_costs.flush(request.costs)
        plan = self.technique.plan
        if plan is not None and request.verifier.supports_trees:
            return self._tree(request, plan, self._state)
        depth = min(max((n for n, a in zip(request.budgets, request.active) if a), default=0),
                    self.technique.num_heads)
        history: list[torch.Tensor] = []
        for _ in range(depth):
            if not request.drafting:
                break
            history.append(request.tokens.to(self._state.device))
            request.extend(self._head(self._state, torch.stack(history, 1)))
        return request.draft()

    def _layout(self, depth: int) -> _TreeLayout:
        plan = self.technique.plan
        assert plan is not None
        if depth not in self._layouts:
            self._layouts[depth] = _TreeLayout(plan, depth, self.technique.device)
        return self._layouts[depth]

    def _tree(self, request: DraftRequest, plan: TreePlan, state: torch.Tensor) -> Draft:
        limit = min(plan.depth, self.technique.num_heads)
        depths = [min(n, limit) if a else 0 for n, a in zip(request.budgets, request.active)]
        depth = max(depths, default=0)
        batch, device = request.batch_size, request.device
        if depth == 0:
            return Draft.empty(batch, device)
        layout = self._layout(depth)
        root = request.tokens.to(state.device)
        width = plan.size(depth)
        tokens = root.new_zeros(batch, width)
        for j, level in enumerate(layout.levels, start=1):
            calls = level.parents.shape[0]
            history = torch.cat([root[:, None, None].expand(batch, calls, 1), tokens[:, level.ancestors]], -1)
            logits = self._head(state.repeat_interleave(calls, 0), history.reshape(batch * calls, j))
            candidates = logits.view(batch, calls, -1).topk(plan.top_k, -1).indices  # [B, calls, top_k]
            start, end = level.nodes
            tokens[:, start:end] = candidates[:, level.calls, level.ranks]
        lengths = [plan.size(d) for d in depths]
        tokens = tokens.to(device)
        pad = torch.arange(width, device=device) >= device_tensor(lengths, device)[:, None]
        tokens = tokens.masked_fill(pad, -1)
        parent_index = layout.parents.to(device).expand(batch, width).masked_fill(pad, -1)
        for b, d in enumerate(depths):
            for j in range(d):
                request.costs[b].record_footprint(CostKind.DRAFT, 1, request.ends[b] - 1 + j, self.footprint)
        self._tree_depth = depth
        return Draft(tokens, lengths, parent_index, None)
