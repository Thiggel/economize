"""The n-gram pool of lookahead decoding and the verification tree made from it."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence

__all__ = ["NgramPool", "guess_tree"]


class NgramPool:
    """The n-grams the Jacobi window has made, keyed by their first token.

    Every key keeps at most ``capacity`` continuations (the n-gram without its first token), least recently added
    first. Adding a continuation that is already there makes it the most recent one; adding a new one to a full
    key evicts the least recent one. This is the LRU rule of the official ``update_token_map``.

    Args:
        capacity: Most continuations per key (``G``).
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._grams: dict[int, OrderedDict[tuple[int, ...], None]] = {}

    def add(self, key: int, gram: tuple[int, ...]) -> None:
        """Record that ``gram`` followed ``key``."""
        if self.capacity == 0:
            return
        grams = self._grams.setdefault(key, OrderedDict())
        if gram in grams:
            grams.move_to_end(gram)
            return
        if len(grams) >= self.capacity:
            grams.popitem(last=False)
        grams[gram] = None

    def add_all(self, grams: Iterable[tuple[int, tuple[int, ...]]]) -> None:
        """:meth:`add` every ``(key, gram)`` in order."""
        for key, gram in grams:
            self.add(key, gram)

    def get(self, key: int) -> list[tuple[int, ...]]:
        """The continuations of ``key``, least recently added first (empty for an unknown key)."""
        grams = self._grams.get(key)
        return [] if grams is None else list(grams)

    def __contains__(self, key: object) -> bool:
        return key in self._grams

    def __len__(self) -> int:
        """Number of keys."""
        return len(self._grams)


def guess_tree(grams: Sequence[Sequence[int]], depth: int) -> tuple[list[int], list[int]]:
    """Merge the first ``depth`` tokens of every n-gram into one prefix tree below the root.

    Returns the tokens and the parents (``-1`` for a child of the root) of the nodes in topological order. Shared
    prefixes become one path, so the tree verifies exactly what the n-grams verified one by one.
    """
    tokens: list[int] = []
    parents: list[int] = []
    children: dict[tuple[int, int], int] = {}
    for gram in grams:
        parent = -1
        for token in gram[:depth]:
            node = children.get((parent, token))
            if node is None:
                node = children[(parent, token)] = len(tokens)
                tokens.append(token)
                parents.append(parent)
            parent = node
    return tokens, parents
