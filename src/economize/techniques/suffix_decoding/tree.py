"""Speculation over a suffix structure: path and tree expansion by empirical continuation probability.

A node is a range ``[lo, hi)`` of sorted continuations that agree on their first ``depth`` tokens; its count is
``hi - lo`` (continuations that end at the node count too, but belong to no child). Two structures provide the
ranges: the global :class:`~economize.techniques.retrieval.SuffixIndex` (:class:`IndexNodes`) and the sorted
continuation table of a row's local matches (:class:`TableNodes`).
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..retrieval import SuffixIndex

__all__ = ["IndexNodes", "Nodes", "Speculation", "TableNodes", "speculate"]


class Nodes(Protocol):
    """The children of a node of a suffix structure."""

    def children(self, lo: int, hi: int, depth: int, min_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tokens and sub-ranges of the children of the node ``[lo, hi)`` at ``depth`` below the match; children
        with fewer than ``min_count`` continuations may be left out."""
        ...


@dataclass
class IndexNodes:
    """Nodes of the global index below a match of ``span`` tokens."""

    index: SuffixIndex
    span: int

    def children(self, lo: int, hi: int, depth: int, min_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.index.children(lo, hi, self.span + depth, min_count)


@dataclass
class TableNodes:
    """Nodes of a lexicographically sorted continuation table ``[N, D]`` (``-1`` from a row's end on).

    Children are remembered: the matches of one occurrence set expand the same table several times."""

    table: np.ndarray
    _known: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict,
                                                                                          repr=False)

    @classmethod
    def of(cls, table: np.ndarray) -> TableNodes:
        """The nodes of an unsorted continuation table."""
        return cls(table[np.lexsort(table.T[::-1])] if table.shape[1] else table)

    def children(self, lo: int, hi: int, depth: int, min_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (lo, hi, depth)
        if key not in self._known:
            column = self.table[lo:hi, depth]
            starts = np.flatnonzero(np.concatenate([[True], column[1:] != column[:-1]]))
            ends = np.append(starts[1:], hi - lo)
            keep = column[starts] >= 0
            self._known[key] = column[starts][keep], lo + starts[keep], lo + ends[keep]
        return self._known[key]


@dataclass
class Speculation:
    """A draft of one match: tokens, parents (``-1``: child of the last committed token), probabilities and the
    score (the sum of the probabilities)."""

    tokens: list[int] = field(default_factory=list)
    parents: list[int] = field(default_factory=list)
    probs: list[float] = field(default_factory=list)

    @property
    def score(self) -> float:
        return float(sum(self.probs))


def _ranked(nodes: Nodes, lo: int, hi: int, depth: int, prob: float, min_prob: float
            ) -> list[tuple[float, int, int, int]]:
    """``(probability, token, lo, hi)`` of every child that reaches ``min_prob``, most frequent first (ties: the lower
    token)."""
    size = hi - lo
    need = min_prob * size / prob if prob > 0 else math.inf
    tokens, starts, ends = nodes.children(lo, hi, depth, max(1, math.floor(need)))
    counts = ends - starts
    order = np.lexsort((tokens, -counts))
    out = []
    for i in order.tolist():
        p = prob * int(counts[i]) / size
        if p < min_prob:
            break
        out.append((p, int(tokens[i]), int(starts[i]), int(ends[i])))
    return out


def speculate(nodes: Nodes, lo: int, hi: int, max_tokens: int, max_depth: int, min_prob: float, tree: bool
              ) -> Speculation:
    """Expand the node ``[lo, hi)`` (Arctic Inference's ``_speculate_path`` and ``_speculate_tree``).

    Args:
        nodes: The suffix structure.
        lo: First continuation of the matched node.
        hi: End of the matched node's continuations.
        max_tokens: Most draft tokens (tree nodes).
        max_depth: Deepest draft token.
        min_prob: Smallest probability of a drafted token.
        tree: Expand the most probable frontier node first (tree); otherwise follow the most frequent child.
    """
    draft = Speculation()
    if not tree:
        prob, parent = 1.0, -1
        while len(draft.tokens) < min(max_tokens, max_depth):
            ranked = _ranked(nodes, lo, hi, len(draft.tokens), prob, min_prob)
            if not ranked:
                break
            prob, token, lo, hi = ranked[0]
            draft.tokens.append(token)
            draft.parents.append(parent)
            draft.probs.append(prob)
            parent += 1
        return draft
    order = itertools.count()  # equal probabilities: first pushed, first expanded
    heap = [(-1.0, next(order), -1, -1, 0, lo, hi)]  # (-prob, order, parent, token, depth, lo, hi); token -1: match
    while heap and len(draft.tokens) < max_tokens:
        negative, _, parent, token, depth, lo, hi = heapq.heappop(heap)
        node = -1
        if token >= 0:
            node = len(draft.tokens)
            draft.tokens.append(token)
            draft.parents.append(parent)
            draft.probs.append(-negative)
        if depth < max_depth and len(draft.tokens) < max_tokens:
            for prob, child, start, end in _ranked(nodes, lo, hi, depth, -negative, min_prob):
                heapq.heappush(heap, (-prob, next(order), node, child, depth + 1, start, end))
    return draft
