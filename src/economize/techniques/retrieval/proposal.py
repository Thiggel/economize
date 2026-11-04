"""Turn the per-row proposals of a retrieval drafter into a :class:`~economize.decoding.Draft`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from ...decoding.cost import CostKind, LayerFootprint
from ...decoding.protocols import Draft
from ...decoding.request import DraftRequest
from ...runtime.transfer import device_tensor
from ..lookup.proposal import chain_draft

__all__ = ["Proposal", "retrieval_draft"]


@dataclass
class Proposal:
    """The draft of one row: a chain (``parents`` is ``None``) or a tree in topological order.

    Attributes:
        tokens: Draft tokens (tree nodes).
        parents: Parent of every node (``-1`` for a child of the last committed token), ``None`` for a chain.
    """

    tokens: list[int] = field(default_factory=list)
    parents: list[int] | None = None


def retrieval_draft(request: DraftRequest, proposals: Sequence[Proposal | None], footprint: LayerFootprint,
                    vocab_size: int, tree: bool) -> Draft:
    """The draft of every row (``None`` or an empty proposal: no draft) and one zero-layer ``draft`` cost event
    per row that was looked up.

    Chains go through :func:`~economize.techniques.lookup.chain_draft` (written into the request's buffer, clipped
    to the budgets, point masses for verifiers that need distributions). Trees are returned as they are: the
    drafter keeps every path within the row's budget.

    Args:
        request: The cycle's draft request.
        proposals: One proposal per row.
        footprint: The drafter's (zero-layer) cost footprint.
        vocab_size: Vocabulary size of the model.
        tree: Return a tree draft (every proposal has ``parents``).
    """
    for b, proposal in enumerate(proposals):
        if proposal is not None:
            request.costs[b].record_footprint(CostKind.DRAFT, 1, request.ends[b] - 1, footprint)
    rows = [p.tokens if p is not None else [] for p in proposals]
    lengths = [len(r) for r in rows]
    width = max(lengths, default=0)
    batch, device = request.batch_size, request.device
    if width == 0:
        return Draft.empty(batch, device)
    if not tree:  # padding 0: chain_draft writes the padding into the buffer's scratch column, which is embedded
        tokens = [r + [0] * (width - len(r)) for r in rows]
        return chain_draft(request, torch.tensor(tokens), lengths, vocab_size=vocab_size)
    tokens = [r + [-1] * (width - len(r)) for r in rows]
    parents = [(p.parents if p is not None and p.parents else []) for p in proposals]
    parents = [q + [-1] * (width - len(q)) for q in parents]
    return Draft(device_tensor(tokens, device), lengths, device_tensor(parents, device), None)
