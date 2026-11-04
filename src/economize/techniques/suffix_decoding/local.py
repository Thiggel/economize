"""The local suffix structure of a row: earlier occurrences of the context's suffixes in the row's own tokens."""

from __future__ import annotations

import numpy as np

__all__ = ["continuation_table", "suffix_matches"]


def suffix_matches(seq: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Every position ``i`` of ``seq`` where an occurrence of the context's last token ends, with the number of
    tokens up to ``i`` that agree with the end of ``seq`` (at most ``limit``).

    The last position is included: the context is itself an occurrence (its continuation is empty), as in a suffix
    tree that holds every suffix of the sequence.
    """
    n = len(seq)
    ends = np.flatnonzero(seq == seq[-1])
    lengths = np.ones(len(ends), dtype=np.int64)
    alive = np.arange(len(ends) - 1)  # the context itself always agrees
    for k in range(1, limit):
        if len(alive) == 0:
            break
        at = ends[alive] - k
        agree = at >= 0
        agree[agree] = seq[at[agree]] == seq[n - 1 - k]
        alive = alive[agree]
        lengths[alive] += 1
    lengths[-1] = limit
    return ends, np.minimum(lengths, limit)


def continuation_table(seq: np.ndarray, ends: np.ndarray, width: int) -> np.ndarray:
    """The ``width`` tokens after every position in ``ends`` (``-1`` past the end of ``seq``), unsorted."""
    offsets = ends[:, None] + 1 + np.arange(width)
    return np.where(offsets < len(seq), seq[np.minimum(offsets, len(seq) - 1)], -1)
