"""The Medusa drafter: all heads at once on the last exact hidden state, as a chain or a static tree."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...decoding.cost import CostKind, LayerFootprint
from ...decoding.protocols import Draft, Drafter
from ...runtime.transfer import device_tensor
from .._drafting import take_rows

if TYPE_CHECKING:
    from ...decoding.protocols import Verifier
    from ...decoding.request import DraftRequest
    from ...runtime import ExactPass, KVCache, LayerRunner
    from .technique import MedusaTechnique
    from .trees import TreePlan

__all__ = ["MedusaDrafter"]


class MedusaDrafter(Drafter):
    """Drafts with the heads of a :class:`~economize.techniques.medusa.MedusaTechnique`.

    After every exact pass the drafter keeps the last hidden state of every row's last processed token. A cycle
    runs the heads once on it: draft ``i`` (0-based) is head ``i + 1``'s token. Without a tree (or with a verifier
    that takes chains only) the heads' distributions go through :meth:`DraftRequest.extend` one by one, so the
    selector, the draft-length policy and rejection sampling see them as usual. With a tree, node ``(r_1..r_j)`` is
    head ``j``'s rank-``r_j`` token (deterministic drafts, ``probs=None``) and the verifier checks every path at once.

    A draft never needs a decoder layer, so the cost log records one zero-layer ``draft`` event of the network
    ``"medusa"`` per head that runs (:attr:`footprint`). Drafts beyond ``K`` heads are never made: cycles longer
    than ``K + 1`` end early.
    """

    layer_span = (1, 0)
    footprint = LayerFootprint.none("medusa")

    def __init__(self, technique: MedusaTechnique) -> None:
        self.technique = technique
        self.source_layers = (technique.adapter.num_layers,)
        self._hidden: torch.Tensor | None = None

    @property
    def observed_layers(self) -> tuple[int, ...]:
        return ()  # the heads read the exact pass's final output, which every pass returns

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        self._hidden = None

    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None:
        final = exact.final
        rows = torch.arange(final.shape[0], device=final.device)
        self._hidden = final[rows, columns.to(final.device)]

    def reorder(self, rows: torch.Tensor) -> None:
        self._hidden = take_rows(self._hidden, rows)

    def draft_capacity(self, cycle_length: int) -> int:
        chain = max(0, min(cycle_length - 1, self.technique.num_heads))
        plan = self.technique.plan
        return chain if plan is None else max(chain, plan.size(min(cycle_length - 1, plan.depth)))

    def drafts_trees(self, verifier: Verifier) -> bool:
        return self.technique.plan is not None and verifier.supports_trees

    def propose(self, request: DraftRequest) -> Draft:
        if self._hidden is None:
            raise RuntimeError("MedusaDrafter.propose needs observe() first")
        active = request.active
        plan = self.technique.plan
        if plan is not None and request.verifier.supports_trees:
            return self._tree(request, plan, active, self._hidden)
        depth = min(max((n for n, a in zip(request.budgets, active) if a), default=0), self.technique.num_heads)
        if depth == 0:
            return Draft.empty(request.batch_size, request.device)
        logits = self.technique.head_logits(self._hidden, depth)
        for k in range(depth):
            if not request.drafting:
                break
            request.extend(logits[:, k])
        return request.draft()

    def _tree(self, request: DraftRequest, plan: TreePlan, active: list[bool], hidden: torch.Tensor) -> Draft:
        limit = min(plan.depth, self.technique.num_heads)
        depths = [min(n, limit) if a else 0 for n, a in zip(request.budgets, active)]
        depth = max(depths, default=0)
        batch, device = request.batch_size, request.device
        if depth == 0:
            return Draft.empty(batch, device)
        logits = self.technique.head_logits(hidden, depth)
        candidates = logits.topk(plan.top_k, -1).indices  # [B, depth, top_k]
        tokens = plan.tokens(candidates, depth).to(device)
        parents = plan.indices(depth, device)[2]
        lengths = [plan.size(d) for d in depths]
        width = tokens.shape[1]
        pad = torch.arange(width, device=device) >= device_tensor(lengths, device)[:, None]
        tokens = tokens.masked_fill(pad, -1)
        parent_index = parents.expand(batch, width).masked_fill(pad, -1)
        for b, d in enumerate(depths):
            for j in range(d):
                request.costs[b].record_footprint(CostKind.DRAFT, 1, request.ends[b] - 1 + j, self.footprint)
        return Draft(tokens, lengths, parent_index, None)
