"""The 2D Jacobi window of lookahead decoding."""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["JacobiWindow"]


class JacobiWindow:
    """The lookahead branch of one row: ``N - 1`` levels of guessed tokens, refined by one Jacobi step per pass.

    Level 0 continues the last committed token (the root). Together they form the window's first block, a chain;
    column ``j`` of level ``l >= 1`` hangs below column ``j`` of level ``l - 1`` (column ``j`` of the first block for
    ``l = 1``), so it sits ``j + l`` positions after the root and sees the first block up to column ``j`` and the
    column's earlier levels. This is the attention mask of the official lookahead branch, written as a tree.

    The window starts with level 0 alone, ``W + N - 3`` tokens drawn at random from the prompt, and gains one level
    per pass until it has ``N - 1`` (:attr:`full`). A full window has ``W - 1`` tokens in level 0 and ``W`` in the
    others; every pass then completes one ``N``-gram per column (:meth:`advance`). The update rules are those of
    ``jacobi_greedy_search_multilevel`` in the official code.

    Args:
        initial: The tokens of level 0.
        levels: ``N - 1``, the number of levels of a full window.
    """

    def __init__(self, initial: Sequence[int], levels: int) -> None:
        self.levels: list[list[int]] = [list(initial)]
        self.depth = levels

    @property
    def full(self) -> bool:
        return len(self.levels) == self.depth

    @property
    def size(self) -> int:
        """Number of window tokens (tree nodes) in the next pass."""
        return sum(len(level) for level in self.levels)

    def layout(self, offset: int) -> tuple[list[int], list[int], list[int]]:
        """The window as tree nodes numbered from ``offset`` on: their tokens, their parents (``-1`` for a child of
        the root) and the nodes of the last level, whose predictions :meth:`advance` takes."""
        tokens: list[int] = []
        parents: list[int] = []
        above: list[int] = [-1]  # the nodes of the level above, by column (column 0 of the first block is the root)
        nodes: list[int] = []
        for index, level in enumerate(self.levels):
            nodes = list(range(offset + len(tokens), offset + len(tokens) + len(level)))
            if index == 0:  # level 0 extends the first block's chain
                parents.extend([-1, *nodes][: len(nodes)])
                above = [-1, *nodes]
            else:
                parents.extend(above[: len(level)])
                above = nodes
            tokens.extend(level)
        return tokens, parents, nodes

    def advance(self, predictions: Sequence[int], root: int) -> list[tuple[int, tuple[int, ...]]]:
        """One Jacobi step from the greedy predictions at the last level's nodes, after a pass whose root was
        ``root``; the window then continues the next root. Returns the n-grams of a full window as
        ``(first token, continuation)`` pairs, one per column, for the pool (none while the window fills)."""
        predictions = list(predictions)
        if not self.full:
            if len(self.levels) == 1:
                self.levels = [self.levels[0][1:], predictions]
            else:
                self.levels = [level[1:] for level in self.levels] + [predictions[1:]]
            return []
        block = [root, *self.levels[0]]
        grams = [
            (block[j], (*(level[j] for level in self.levels[1:]), predictions[j]))
            for j in range(len(predictions))
        ]
        self.levels = [self.levels[1][1:], *self.levels[2:], predictions]
        return grams
