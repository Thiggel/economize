"""The datastore of the retrieval drafters (REST, SuffixDecoding): a suffix array over records of tokens.

Build an index once from token spans of a corpus, then point a drafter's ``datastore`` at the folder:

    python -m economize.techniques.retrieval build store/ --tokens train.bin --dtype uint32 --spans spans.npy

or from Python, :meth:`SuffixIndex.build` (any iterable of token sequences) and :meth:`SuffixIndex.save`.
The package registers no technique.
"""

from __future__ import annotations

import numpy as np

from .drafter import RetrievalDrafter
from .index import FORMAT, SEPARATOR, SuffixIndex, load_index, suffix_array
from .proposal import Proposal, retrieval_draft

__all__ = [
    "FORMAT",
    "SEPARATOR",
    "Proposal",
    "RetrievalDrafter",
    "SuffixIndex",
    "build_from_spans",
    "load_index",
    "retrieval_draft",
    "suffix_array",
]


def build_from_spans(tokens: np.ndarray, spans: np.ndarray, depth: int = 64) -> SuffixIndex:
    """Index the records ``tokens[start:end]`` for every row ``(start, end)`` of ``spans``.

    Args:
        tokens: A flat token array (a NumPy array or memory map).
        spans: ``[R, 2]`` record boundaries.
        depth: See :meth:`SuffixIndex.build`.
    """
    return SuffixIndex.build((tokens[int(s) : int(e)] for s, e in spans), depth)
