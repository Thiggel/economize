"""A suffix array over a token corpus of separate records, the datastore of the retrieval drafters.

The corpus is one ``int32`` array: every record is followed by :data:`SEPARATOR` (``-1``), and ``depth`` further
separators pad the end so that windows of up to ``depth`` tokens can be read past any position. The suffix array
lists every position that holds a token (never a separator), ordered by the first ``depth`` tokens of its suffix:
suffixes that agree on ``depth`` tokens keep an unspecified order among themselves. A pattern never contains the
separator, so its occurrences never cross a record boundary, and the continuations of one pattern are sorted by
their tokens as long as pattern and continuation together span at most ``depth`` tokens.

The array is built by prefix doubling (one ``argsort`` of ``int64`` keys per doubling, ``log2(depth)`` passes) in
NumPy. Lookups narrow every pattern of a batch at once by binary search (:meth:`SuffixIndex.find`), so one query of
up to 64 patterns costs about 25 vectorized steps whatever the corpus size.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

from ...errors import ConfigError

__all__ = ["FORMAT", "SEPARATOR", "SuffixIndex", "load_index", "suffix_array"]

#: Separator token written after every record (sorts before every token).
SEPARATOR = -1

#: The ``format`` entry of an index's ``meta.json``.
FORMAT = "economize-suffix-index"
_VERSION = 1
_FILES = {"tokens": "tokens.npy", "suffixes": "suffix_array.npy", "buckets": "buckets.npy"}

#: Ranges of at most this many suffixes are read whole when their children are listed.
GATHER_LIMIT = 4096


def suffix_array(tokens: np.ndarray, depth: int) -> np.ndarray:
    """The positions of ``tokens`` ordered by their first ``depth`` tokens (prefix doubling).

    Args:
        tokens: ``int32`` tokens, ``SEPARATOR`` between records. Reads past the end compare below every value.
        depth: Number of leading tokens the order is exact on.

    Returns:
        All positions (``int64``), separators included, in suffix order.
    """
    n = len(tokens)
    rank = tokens.astype(np.int64) + 2  # separator 1, past the end 0
    order = np.argsort(rank, kind="stable")
    rank = _dense_ranks(rank[order], order)
    width = 1
    while width < depth and int(np.max(rank)) < n:
        base = int(np.max(rank)) + 1
        key = rank * base
        key[: n - width] += rank[width:]
        order = np.argsort(key)
        rank = _dense_ranks(key[order], order)
        del key
        width *= 2
    return order


def _dense_ranks(sorted_keys: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Ranks ``1..`` of the positions in ``order`` (equal keys share a rank)."""
    new = np.empty(len(sorted_keys), dtype=bool)
    new[0] = True
    np.not_equal(sorted_keys[1:], sorted_keys[:-1], out=new[1:])
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.cumsum(new, dtype=np.int64)
    return ranks


class SuffixIndex:
    """Suffix array over a corpus of records (see the module documentation).

    Attributes:
        tokens: The corpus with separators and end padding (``int32``).
        suffixes: Start of every suffix that begins with a token, in suffix order (``int32`` or ``int64``).
        buckets: ``buckets[t]`` is the first suffix whose first token is ``>= t`` (``int64`` ``[V + 1]``).
        depth: Number of leading tokens the order is exact on.
        num_records: Number of records.
    """

    def __init__(self, tokens: np.ndarray, suffixes: np.ndarray, buckets: np.ndarray, depth: int,
                 num_records: int) -> None:
        self.tokens, self.suffixes, self.buckets = tokens, suffixes, buckets
        self.depth, self.num_records = depth, num_records
        self._offsets = np.arange(depth, dtype=np.int64)

    # ---- building and storage ----------------------------------------------------------------------------------
    @classmethod
    def build(cls, records: Iterable[Sequence[int] | np.ndarray], depth: int = 64) -> SuffixIndex:
        """Index ``records`` (token sequences; empty ones are skipped).

        Args:
            records: The token sequences of the datastore (non-negative token ids).
            depth: Number of leading tokens of every suffix the order is exact on; patterns plus continuations
                must fit in it.
        """
        parts = [np.asarray(r, dtype=np.int64) for r in records]
        parts = [p for p in parts if len(p)]
        if any(int(np.min(p)) < 0 for p in parts):
            raise ConfigError("token ids must be non-negative")
        total = sum(len(p) + 1 for p in parts)
        tokens = np.full(total + depth, SEPARATOR, dtype=np.int32)
        position = 0
        for part in parts:
            tokens[position : position + len(part)] = part
            position += len(part) + 1
        order = suffix_array(tokens, depth)
        dtype = np.int32 if len(tokens) < 2**31 else np.int64
        suffixes = order[tokens[order] >= 0].astype(dtype)
        firsts = tokens[suffixes]
        vocab = int(firsts[-1]) + 1 if len(firsts) else 0
        buckets = np.searchsorted(firsts, np.arange(vocab + 1)).astype(np.int64)
        return cls(tokens, suffixes, buckets, depth, len(parts))

    def save(self, path: str | Path) -> None:
        """Write the index to the folder ``path`` (created if missing)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name, file in _FILES.items():
            np.save(path / file, getattr(self, name))
        meta = {"format": FORMAT, "version": _VERSION, "depth": self.depth, "num_records": self.num_records,
                "num_tokens": len(self.suffixes)}
        (path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path, *, mmap: bool = False) -> SuffixIndex:
        """Read an index written by :meth:`save` (``mmap`` maps the arrays instead of reading them)."""
        path = Path(path)
        meta_file = path / "meta.json"
        if not meta_file.is_file():
            raise ConfigError(f"{path} holds no suffix index (no meta.json); build one with "
                              "`python -m economize.techniques.retrieval build`")
        meta = json.loads(meta_file.read_text())
        if meta.get("format") != FORMAT or meta.get("version") != _VERSION:
            raise ConfigError(f"{meta_file} is not a version-{_VERSION} {FORMAT}")
        # a plain view of a mapped array: memmap's subclass hooks cost more than the lookups themselves
        arrays = {name: np.load(path / file, mmap_mode="r" if mmap else None).view(np.ndarray)
                  for name, file in _FILES.items()}
        return cls(arrays["tokens"], arrays["suffixes"], arrays["buckets"], meta["depth"], meta["num_records"])

    def __len__(self) -> int:
        return len(self.suffixes)

    def __repr__(self) -> str:
        return f"SuffixIndex(tokens={len(self)}, records={self.num_records}, depth={self.depth})"

    # ---- lookups -----------------------------------------------------------------------------------------------
    def find(self, patterns: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
        """The suffix ranges ``[lo, hi)`` that begin with each pattern (all patterns at once).

        Args:
            patterns: Non-empty token sequences of at most :attr:`depth` tokens.

        Returns:
            ``lo`` and ``hi`` (``int64`` ``[P]``); ``hi - lo`` is the number of occurrences.
        """
        count = len(patterns)
        if count == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        width = max(len(p) for p in patterns)
        if width > self.depth or min(len(p) for p in patterns) == 0:
            raise ConfigError(f"patterns must have 1..{self.depth} tokens")
        table = np.full((count, width), SEPARATOR - 1, dtype=np.int64)
        for i, p in enumerate(patterns):
            table[i, : len(p)] = p
        valid = table >= 0
        first = table[:, 0]
        known = first < len(self.buckets) - 1
        first = np.where(known, first, 0)
        lo = np.where(known, self.buckets[first], 0)
        hi = np.where(known, self.buckets[first + 1], 0)
        # one batch of 2P searches: the lower bound (first suffix >= pattern) and the upper bound (first suffix
        # whose prefix is > pattern)
        lo, hi = np.concatenate([lo, lo]), np.concatenate([hi, hi])
        table, valid = np.concatenate([table, table]), np.concatenate([valid, valid])
        upper = np.repeat([False, True], count)
        offsets = self._offsets[:width]
        while True:
            rows = np.flatnonzero(lo < hi)
            if len(rows) == 0:
                break
            mid = (lo[rows] + hi[rows]) >> 1
            window = self.tokens[self.suffixes[mid].astype(np.int64)[:, None] + offsets]
            pattern = table[rows]
            differs = (window != pattern) & valid[rows]
            some = differs.any(1)
            column = differs.argmax(1)
            picked = np.arange(len(rows))
            below = some & (window[picked, column] < pattern[picked, column])
            right = below | (upper[rows] & ~some)
            lo[rows] = np.where(right, mid + 1, lo[rows])
            hi[rows] = np.where(right, hi[rows], mid)
        return lo[:count], lo[count:]

    def positions(self, lo: int, hi: int, step: int = 1) -> np.ndarray:
        """Corpus positions of suffixes ``lo, lo + step, ... < hi`` (``int64``)."""
        return self.suffixes[lo:hi:step].astype(np.int64)

    def windows(self, starts: np.ndarray, length: int) -> np.ndarray:
        """The ``length`` tokens from every corpus position in ``starts`` (``[N, length]``; separators past a record's
        end and everything after them read as ``SEPARATOR``)."""
        if length > self.depth:
            raise ConfigError(f"windows of at most {self.depth} tokens can be read")
        window = self.tokens[starts[:, None] + self._offsets[:length]].astype(np.int64)
        ended = np.logical_or.accumulate(window < 0, axis=1)
        window[ended] = SEPARATOR
        return window

    def children(self, lo: int, hi: int, offset: int, min_count: int = 1
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The distinct tokens at ``offset`` of the suffixes ``[lo, hi)``, which must agree on their first ``offset``
        tokens (``offset < depth``), with their sub-ranges.

        Tokens that occur fewer than ``min_count`` times may be left out: large ranges are sampled every
        ``min_count`` suffixes, and the exact boundaries of every sampled token are then found by binary search.

        Returns:
            ``tokens``, ``lo`` and ``hi`` of every child (``int64``), in token order; suffixes that end at ``offset``
            belong to no child.
        """
        if offset >= self.depth:
            raise ConfigError(f"the suffix order is exact on {self.depth} tokens only")
        size = hi - lo
        if size <= GATHER_LIMIT or min_count <= 1:
            column = self.tokens[self.positions(lo, hi) + offset]
            starts = np.flatnonzero(np.concatenate([[True], column[1:] != column[:-1]]))
            ends = np.append(starts[1:], size)
            keep = column[starts] >= 0
            return column[starts][keep].astype(np.int64), lo + starts[keep], lo + ends[keep]
        sample = self.tokens[self.positions(lo, hi, min_count) + offset]
        candidates = np.unique(sample[sample >= 0]).astype(np.int64)
        bounds = self._column_bounds(lo, hi, offset, candidates)
        return candidates, bounds[: len(candidates)], bounds[len(candidates) :]

    def _column_bounds(self, lo: int, hi: int, offset: int, values: np.ndarray) -> np.ndarray:
        """First suffix in ``[lo, hi)`` whose token at ``offset`` is ``>= v``, then ``> v``, for every value."""
        count = len(values)
        target = np.concatenate([values, values])
        upper = np.repeat([False, True], count)
        low = np.full(2 * count, lo, dtype=np.int64)
        high = np.full(2 * count, hi, dtype=np.int64)
        for _ in range(max(1, math.ceil(math.log2(hi - lo + 1)))):
            mid = (low + high) >> 1
            token = self.tokens[self.suffixes[np.minimum(mid, hi - 1)].astype(np.int64) + offset]
            right = (token < target) | (upper & (token == target))
            active = low < high
            low = np.where(active & right, mid + 1, low)
            high = np.where(active & ~right, mid, high)
        return low


@lru_cache(maxsize=4)
def load_index(path: str) -> SuffixIndex:
    """:meth:`SuffixIndex.load` of the folder ``path``, cached (drafters of one process share an index)."""
    return SuffixIndex.load(str(Path(path).expanduser().resolve()))
