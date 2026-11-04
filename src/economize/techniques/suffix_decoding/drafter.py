"""The SuffixDecoding drafter: the best-scoring match of the context in the global and the local structure."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ...decoding.cost import LayerFootprint
from ..retrieval import Proposal, RetrievalDrafter, SuffixIndex
from ..retrieval import index as suffix_index
from .config import SuffixDecodingConfig
from .local import continuation_table, suffix_matches
from .tree import IndexNodes, Speculation, TableNodes, speculate

__all__ = ["SuffixDecodingDrafter"]


class SuffixDecodingDrafter(RetrievalDrafter):
    """Drafts by suffix-tree speculation over a datastore and the row's own tokens (see
    :class:`SuffixDecodingConfig`).

    The global structure is a static :class:`~economize.techniques.retrieval.SuffixIndex`; all rows' match lengths
    are searched in one batched binary search. The local structure is rebuilt from the row's committed tokens every
    cycle (one vectorized pass over them finds every earlier occurrence of each suffix), so it needs no state.

    Match lengths are tried longest first. A length is skipped when it cannot change the result: when its adaptive
    length cannot beat the best score so far, or when a longer match with the same occurrences (an equal count) had
    at least its token and depth limits. Both rules return Arctic Inference's choice. The matches of one occurrence
    set share their continuations, which are read once into a table (up to ``GATHER_LIMIT`` occurrences; larger
    sets are expanded in the index itself).

    Args:
        config: The SuffixDecoding configuration.
        index: The global datastore (``None`` when the configuration names none).
        vocab_size: Vocabulary size of the model.
    """

    footprint = LayerFootprint.none("suffix_decoding")

    def __init__(self, config: SuffixDecodingConfig, index: SuffixIndex | None, vocab_size: int) -> None:
        self.global_depth = 0 if index is None else min(config.max_depth, index.depth)
        nodes = config.max_spec_tokens or 0
        super().__init__(vocab_size, tree=config.tree, tree_nodes=nodes,
                         context=None if config.local else self.global_depth)
        self.config, self.index = config, index

    def max_tokens(self, match: int, cap: int) -> int:
        """The adaptive speculation length of a match of ``match`` tokens."""
        c = self.config
        return min(cap, int(match * c.max_spec_factor + c.max_spec_offset + 1e-6))

    def proposals(self, contexts: Sequence[np.ndarray | None], budgets: Sequence[int], tree: bool
                  ) -> list[Proposal | None]:
        c = self.config
        rows = [b for b, ctx in enumerate(contexts) if ctx is not None and budgets[b] > 0]
        found = self._global_matches(contexts, rows) if self.index is not None else {}
        out: list[Proposal | None] = [None] * len(contexts)
        for b in rows:
            seq, budget = contexts[b], budgets[b]
            cap = budget if c.max_spec_tokens is None else c.max_spec_tokens
            if not tree:
                cap = min(cap, budget)
            best = Speculation()
            if b in found:
                best = self._global(*found[b], cap, budget, tree)
            if c.local and len(seq) >= 2:
                local = self._local(seq, cap, budget, tree)
                if local.score >= best.score:
                    best = local
            out[b] = Proposal(best.tokens, best.parents if tree else None)
        return out

    def _global_matches(self, contexts: Sequence[np.ndarray | None], rows: list[int]
                        ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """The suffix ranges of every match length ``1..`` of every row's context (cut at the first miss)."""
        patterns, owners = [], []
        for b in rows:
            ctx = contexts[b][-self.global_depth :]
            for m in range(1, len(ctx)):
                patterns.append(ctx[-m:])
                owners.append(b)
        lo, hi = self.index.find(patterns)
        owners_array = np.asarray(owners)
        out = {}
        for b in rows:
            mine = np.flatnonzero(owners_array == b)
            if len(mine) == 0:
                continue
            row_lo, row_hi = lo[mine], hi[mine]
            missing = np.flatnonzero(row_hi <= row_lo)
            stop = int(missing[0]) if len(missing) else len(mine)
            if stop:
                out[b] = row_lo[:stop], row_hi[:stop]
        return out

    def _global(self, lo: np.ndarray, hi: np.ndarray, cap: int, budget: int, tree: bool) -> Speculation:
        counts = hi - lo
        widths = _widths(counts, self.global_depth, budget)
        tables: dict[int, TableNodes] = {}

        def expand(m: int, n: int, d: int) -> Speculation:
            count, start = int(counts[m - 1]), int(lo[m - 1])
            if count > suffix_index.GATHER_LIMIT:
                return speculate(IndexNodes(self.index, m), start, start + count, n, d, self.config.min_token_prob,
                                 tree)
            if count not in tables:
                continuations = self.index.positions(start, start + count) + m
                tables[count] = TableNodes.of(self.index.windows(continuations, widths[count]))
            return speculate(tables[count], 0, count, n, d, self.config.min_token_prob, tree)

        return self._best(counts, cap, budget, self.global_depth, expand)

    def _local(self, seq: np.ndarray, cap: int, budget: int, tree: bool) -> Speculation:
        depth = min(len(seq), self.config.max_depth)
        ends, lengths = suffix_matches(seq, depth - 1)
        counts = np.array([np.count_nonzero(lengths >= m) for m in range(1, depth)])
        counts[counts <= 1] = 0  # the context alone has no continuation
        widths = _widths(counts, self.config.max_depth, budget)
        tables: dict[int, TableNodes] = {}

        def expand(m: int, n: int, d: int) -> Speculation:
            count = int(counts[m - 1])
            if count not in tables:
                tables[count] = TableNodes.of(continuation_table(seq, ends[lengths >= m], widths[count]))
            return speculate(tables[count], 0, count, n, d, self.config.min_token_prob, tree)

        return self._best(counts, cap, budget, self.config.max_depth, expand)

    def _best(self, counts: np.ndarray, cap: int, budget: int, depth: int,
              expand: Callable[[int, int, int], Speculation]) -> Speculation:
        """The best draft over match lengths ``m = len(counts), ..., 1`` (``counts[m - 1]`` occurrences; ties go to
        the longer match). ``expand(m, max_tokens, max_depth)`` drafts from the match of ``m`` tokens."""
        best, found = Speculation(), False
        limits: dict[int, list[tuple[int, int]]] = {}
        for m in range(len(counts), 0, -1):
            count = int(counts[m - 1])
            tokens = self.max_tokens(m, cap)
            reach = min(depth - m, tokens, budget)
            if count == 0 or reach <= 0 or (found and tokens <= best.score):
                continue
            if any(t >= tokens and r >= reach for t, r in limits.get(count, ())):
                continue  # a longer match with the same occurrences drafts at least as much
            limits.setdefault(count, []).append((tokens, reach))
            draft = expand(m, tokens, reach)
            if not found or draft.score > best.score:
                best, found = draft, True
        return best


def _widths(counts: np.ndarray, depth: int, budget: int) -> dict[int, int]:
    """The widest continuation table a match of each occurrence count needs. Counts fall with the match length,
    so the matches of one count share their occurrences, and the shortest of them reaches deepest."""
    widths: dict[int, int] = {}
    for m, count in enumerate(counts.tolist(), start=1):
        widths.setdefault(int(count), max(0, min(depth - m, budget)))
    return widths
