"""The continuation trie of REST, built from a lexicographically sorted table of continuations (NumPy only)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Trie", "build_trie"]


@dataclass
class Trie:
    """The nodes of a trie in order of depth, then of their first table row (a topological order).

    Attributes:
        tokens: Token of every node.
        parents: Parent node of every node (``-1`` for a child of the root).
        counts: Number of continuations through every node.
        depths: Depth of every node (1 for a child of the root).
    """

    tokens: np.ndarray
    parents: np.ndarray
    counts: np.ndarray
    depths: np.ndarray

    def __len__(self) -> int:
        return len(self.tokens)

    def top(self, size: int) -> tuple[list[int], list[int]]:
        """The ``size`` most frequent nodes (ties: shallower, then earlier) as ``(tokens, parents)`` in topological
        order. A node never counts more than its parent and is shallower, so the selection is a subtree."""
        order = np.lexsort((np.arange(len(self)), self.depths, -self.counts))[:size]
        order.sort()  # node order is topological
        index = np.full(len(self) + 1, -1, dtype=np.int64)
        index[order] = np.arange(len(order))  # index[-1] stays -1: the root
        return self.tokens[order].tolist(), index[self.parents[order]].tolist()

    def path(self) -> list[int]:
        """The most frequent path from the root (at every node the most frequent child; ties: the lower token)."""
        tokens, node = [], -1
        for depth in range(1, int(self.depths.max(initial=0)) + 1):
            children = np.flatnonzero((self.depths == depth) & (self.parents == node))
            if len(children) == 0:
                break
            node = int(children[np.argmax(self.counts[children])])  # first maximum: the lowest token
            tokens.append(int(self.tokens[node]))
        return tokens


def build_trie(table: np.ndarray) -> Trie:
    """The trie of the rows of ``table`` ``[N, D]``, sorted lexicographically, ``-1`` from a row's end on.

    A node at depth ``j`` is a run of rows that agree on their first ``j`` tokens; its count is the run's length.
    """
    rows, width = table.shape
    empty = np.zeros(0, dtype=np.int64)
    if rows == 0 or width == 0:
        return Trie(empty, empty, empty, empty)
    change = np.ones_like(table, dtype=bool)
    change[1:] = table[1:] != table[:-1]
    change = np.logical_or.accumulate(change, axis=1)  # row i opens a new node at depth j
    tokens, parents, counts, depths = [], [], [], []
    previous_starts = np.zeros(1, dtype=np.int64)
    previous_ids = np.full(1, -1, dtype=np.int64)
    next_id = 0
    for j in range(width):
        starts = np.flatnonzero(change[:, j])
        sizes = np.diff(np.append(starts, rows))
        owner = previous_ids[np.searchsorted(previous_starts, starts, side="right") - 1]
        keep = (table[starts, j] >= 0) & ((owner >= 0) | (j == 0))
        ids = np.full(len(starts), -2, dtype=np.int64)
        ids[keep] = next_id + np.arange(int(keep.sum()))
        next_id += int(keep.sum())
        if not keep.any():
            break
        tokens.append(table[starts[keep], j])
        parents.append(owner[keep])
        counts.append(sizes[keep])
        depths.append(np.full(int(keep.sum()), j + 1, dtype=np.int64))
        previous_starts, previous_ids = starts, ids
    if not tokens:  # every continuation is empty (the matches end their records)
        return Trie(empty, empty, empty, empty)
    return Trie(*(np.concatenate(x).astype(np.int64) for x in (tokens, parents, counts, depths)))
