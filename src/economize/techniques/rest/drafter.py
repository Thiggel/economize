"""The REST drafter: longest suffix match in the datastore, continuation trie, most frequent nodes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ...decoding.cost import LayerFootprint
from ..retrieval import Proposal, RetrievalDrafter, SuffixIndex
from .config import RestConfig
from .trie import build_trie

__all__ = ["RestDrafter"]


class RestDrafter(RetrievalDrafter):
    """Drafts from the continuations of the context's longest suffix in the datastore (see :class:`RestConfig`).

    All rows and all span lengths are searched in one batched binary search; per row, the continuations of the
    longest matching span are read as one ``[k, depth]`` table, which the suffix array delivers sorted, so the trie
    is a few vectorized passes over it.

    Args:
        config: The REST configuration.
        index: The datastore.
        vocab_size: Vocabulary size of the model.
    """

    footprint = LayerFootprint.none("rest")

    def __init__(self, config: RestConfig, index: SuffixIndex, vocab_size: int) -> None:
        super().__init__(vocab_size, tree=config.tree, tree_nodes=config.tree_size, context=config.max_span)
        self.config, self.index = config, index

    def proposals(self, contexts: Sequence[np.ndarray | None], budgets: Sequence[int], tree: bool
                  ) -> list[Proposal | None]:
        c = self.config
        patterns, owners = [], []
        for b, context in enumerate(contexts):
            if context is None or budgets[b] == 0:
                continue
            for span in range(min(c.max_span, len(context)), c.min_span - 1, -1):
                patterns.append(context[-span:])
                owners.append(b)
        lo, hi = self.index.find(patterns)
        out: list[Proposal | None] = [None] * len(contexts)
        for b in sorted(set(owners)):
            rows = [i for i, o in enumerate(owners) if o == b and hi[i] > lo[i]]
            if not rows:
                out[b] = Proposal([], [] if tree else None)
                continue
            i = rows[0]  # the longest span with an occurrence
            out[b] = self.draft(len(patterns[i]), int(lo[i]), int(hi[i]), min(budgets[b], c.max_depth), tree)
        return out

    def draft(self, span: int, lo: int, hi: int, depth: int, tree: bool) -> Proposal:
        """The draft from the occurrences ``[lo, hi)`` of a span of ``span`` tokens, at most ``depth`` deep."""
        c = self.config
        count = hi - lo
        step = 1 if c.occurrences == "first" or count <= c.max_occurrences else count // c.max_occurrences
        starts = self.index.positions(lo, min(hi, lo + c.max_occurrences * step), step)[: c.max_occurrences]
        table = self.index.windows(starts + span, depth)
        if span + depth > self.index.depth:  # the suffix order is exact on fewer tokens than the table spans
            table = table[np.lexsort(table.T[::-1])]
        trie = build_trie(table)
        if not tree:
            return Proposal(trie.path())
        tokens, parents = trie.top(c.tree_size)
        return Proposal(tokens, parents)
