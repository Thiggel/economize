"""The shared drafter of the retrieval techniques: read the context, look it up, hand the proposals over."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from ...decoding.cost import LayerFootprint
from ...decoding.protocols import Draft, Drafter
from .proposal import Proposal, retrieval_draft

if TYPE_CHECKING:
    from ...decoding.protocols import Verifier
    from ...decoding.request import DraftRequest
    from ...runtime import ExactPass, KVCache, LayerRunner

__all__ = ["RetrievalDrafter"]


class RetrievalDrafter(Drafter):
    """A drafter that runs no model layer: :meth:`proposals` turns every row's committed tokens into a chain or a
    tree, which the verifier checks in one exact pass.

    One cycle copies the context window of every drafting row to the host once (one synchronization), looks the rows
    up on the CPU and records one zero-layer ``draft`` event per row (:attr:`footprint`), so the cost log counts
    the lookups without layer costs. Drafts are deterministic: chains carry point masses for verifiers that need
    distributions, and trees leave ``probs`` unset, which the tree verifiers read as point masses.

    Args:
        vocab_size: Vocabulary size of the model.
        tree: Draft trees when the verifier verifies them.
        tree_nodes: Most nodes of a tree draft (for :meth:`draft_capacity`).
        context: Most trailing tokens of a row the lookup reads (``None``: all of them).
    """

    layer_span = (1, 0)
    footprint: LayerFootprint

    def __init__(self, vocab_size: int, *, tree: bool, tree_nodes: int, context: int | None) -> None:
        self.vocab_size, self.tree, self.tree_nodes, self.context = vocab_size, tree, tree_nodes, context

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        pass

    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None:
        pass

    def draft_capacity(self, cycle_length: int) -> int:
        chain = max(0, cycle_length - 1)
        return max(chain, self.tree_nodes) if self.tree else chain

    def drafts_trees(self, verifier: Verifier) -> bool:
        return self.tree and verifier.supports_trees

    @abstractmethod
    def proposals(self, contexts: Sequence[np.ndarray | None], budgets: Sequence[int], tree: bool
                  ) -> list[Proposal | None]:
        """The draft of every row.

        Args:
            contexts: The committed tokens of every drafting row (the last :attr:`context` of them), ``None`` for a
                row that does not draft.
            budgets: Most draft tokens (the deepest tree path) of every row.
            tree: Return trees (every proposal has ``parents``) instead of chains.

        Returns:
            One proposal per row, ``None`` for a row that was not looked up.
        """

    def contexts(self, request: DraftRequest) -> list[np.ndarray | None]:
        """The trailing committed tokens of every active row, read with one device synchronization."""
        active = request.active
        ends = [e if a else 0 for e, a in zip(request.ends, active)]
        stop = max(ends, default=0)
        if stop == 0:
            return [None] * len(ends)
        start = 0 if self.context is None else max(0, min(e - self.context for e, a in zip(ends, active) if a))
        window = request.seq[:, start:stop].cpu().numpy()
        firsts = [0 if self.context is None else max(0, e - self.context) for e in ends]
        return [window[b, max(first - start, 0) : end - start].astype(np.int64) if a and end > 0 else None
                for b, (first, end, a) in enumerate(zip(firsts, ends, active))]

    def propose_partial(self, request: DraftRequest) -> Draft:
        """Chain drafts; rows without a match have none (the primary-drafter role of a hybrid,
        :class:`~economize.techniques._drafting.PartialDrafter`)."""
        return self._propose(request, tree=False)

    def propose(self, request: DraftRequest) -> Draft:
        return self._propose(request, tree=self.drafts_trees(request.verifier))

    def _propose(self, request: DraftRequest, tree: bool) -> Draft:
        budgets = [max(0, n - done) if a else 0 for n, done, a in
                   zip(request.budgets, request.lengths, request.active)]
        proposals = self.proposals(self.contexts(request), budgets, tree)
        return retrieval_draft(request, proposals, self.footprint, self.vocab_size, tree)
