"""The suffix index of the retrieval drafters: suffix order, batched search, children, storage and the CLI."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from economize import ConfigError
from economize.techniques import retrieval
from economize.techniques.retrieval import SEPARATOR, SuffixIndex, build_from_spans, load_index, suffix_array
from helpers import subprocess_env


def _records(seed: int = 0, count: int = 40, vocab: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, vocab, rng.integers(1, 30)) for _ in range(count)]


def _occurrences(records: list[np.ndarray], pattern: list[int]) -> list[tuple[int, int]]:
    """(record, offset) of every occurrence of ``pattern`` inside one record, by the definition."""
    n = len(pattern)
    return [
        (r, i) for r, rec in enumerate(records) for i in range(len(rec) - n + 1) if rec[i : i + n].tolist() == pattern
    ]


def _corpus_position(records: list[np.ndarray], record: int, offset: int) -> int:
    return sum(len(r) + 1 for r in records[:record]) + offset


# ---- the suffix order ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("depth", [1, 3, 8, 64])
def test_suffix_array_orders_by_the_first_depth_tokens(depth: int) -> None:
    tokens = np.random.default_rng(depth).integers(-1, 4, 300).astype(np.int32)
    order = suffix_array(tokens, depth)
    assert sorted(order.tolist()) == list(range(len(tokens)))
    keys = [tuple(tokens[i : i + depth].tolist()) for i in order]
    padded = [k + (-2,) * (depth - len(k)) for k in keys]  # past the end sorts first
    assert padded == sorted(padded)


def test_build_skips_separators_and_empty_records() -> None:
    index = SuffixIndex.build([[3, 1], [], [2]], depth=4)
    assert index.num_records == 2 and len(index) == 3
    assert index.tokens.tolist() == [3, 1, SEPARATOR, 2, SEPARATOR] + [SEPARATOR] * 4
    assert sorted(index.tokens[index.suffixes].tolist()) == [1, 2, 3]
    assert index.buckets.tolist() == [0, 0, 1, 2, 3]
    with pytest.raises(ConfigError, match="non-negative"):
        SuffixIndex.build([[1, -3]])


# ---- search --------------------------------------------------------------------------------------------------------
def test_find_counts_every_occurrence_within_records() -> None:
    records = _records()
    index = SuffixIndex.build(records, depth=16)
    rng = np.random.default_rng(1)
    patterns = [rng.integers(0, 5, rng.integers(1, 5)).tolist() for _ in range(200)] + [[7], [4, 9], [0] * 16]
    lo, hi = index.find(patterns)
    for pattern, a, b in zip(patterns, lo.tolist(), hi.tolist()):
        expected = sorted(_corpus_position(records, r, i) for r, i in _occurrences(records, pattern))
        assert sorted(index.positions(a, b).tolist()) == expected


def test_matches_never_cross_a_record_boundary() -> None:
    index = SuffixIndex.build([[1, 2, 3], [4, 5]], depth=8)
    lo, hi = index.find([[3, 4], [2, 3], [3], [5]])
    assert (hi - lo).tolist() == [0, 1, 1, 1]
    window = index.windows(index.positions(int(lo[2]), int(hi[2])), 4)
    assert window.tolist() == [[3, SEPARATOR, SEPARATOR, SEPARATOR]]


def test_find_rejects_patterns_longer_than_the_depth() -> None:
    index = SuffixIndex.build([[1, 2, 3]], depth=2)
    assert [x.tolist() for x in index.find([])] == [[], []]
    for bad in ([[1, 2, 3]], [[]]):
        with pytest.raises(ConfigError, match="patterns must have"):
            index.find(bad)


# ---- children ------------------------------------------------------------------------------------------------------
def _brute_children(index: SuffixIndex, lo: int, hi: int, offset: int) -> dict[int, int]:
    column = index.tokens[index.positions(lo, hi) + offset].tolist()
    return {t: column.count(t) for t in set(column) if t >= 0}


@pytest.mark.parametrize("gather_limit", [4096, 4], ids=["gathered", "sampled"])
@pytest.mark.parametrize("min_count", [1, 3])
def test_children_list_the_next_tokens_with_their_ranges(monkeypatch, gather_limit: int, min_count: int) -> None:
    monkeypatch.setattr(retrieval.index, "GATHER_LIMIT", gather_limit)
    records = _records(seed=2, count=120, vocab=4)
    index = SuffixIndex.build(records, depth=8)
    patterns = [[0], [1, 2], [3, 3], [2]]
    lo, hi = index.find(patterns)
    for pattern, a, b in zip(patterns, lo.tolist(), hi.tolist()):
        tokens, starts, ends = index.children(a, b, len(pattern), min_count)
        expected = _brute_children(index, a, b, len(pattern))
        got = dict(zip(tokens.tolist(), (ends - starts).tolist()))
        assert all(got[t] == c for t, c in expected.items() if c >= min_count) and set(got) <= set(expected)
        if min_count == 1:
            assert got == expected
        for t, s, e in zip(tokens.tolist(), starts.tolist(), ends.tolist()):
            assert set(index.tokens[index.positions(s, e) + len(pattern)].tolist()) == {t}
        assert tokens.tolist() == sorted(tokens.tolist())
    with pytest.raises(ConfigError, match="exact on 8"):
        index.children(0, len(index), 8)


def test_windows_are_sorted_below_a_match() -> None:
    index = SuffixIndex.build(_records(seed=3, count=200, vocab=3), depth=12)
    lo, hi = index.find([[0, 1]])
    table = index.windows(index.positions(int(lo[0]), int(hi[0])) + 2, 10)
    assert [tuple(r) for r in table.tolist()] == sorted(tuple(r) for r in table.tolist())


# ---- storage and the command line ----------------------------------------------------------------------------------
def test_save_load_and_the_cached_loader(tmp_path) -> None:
    index = SuffixIndex.build(_records(), depth=16)
    index.save(tmp_path / "store")
    for loaded in (SuffixIndex.load(tmp_path / "store"), SuffixIndex.load(tmp_path / "store", mmap=True)):
        assert (loaded.depth, loaded.num_records, len(loaded)) == (16, index.num_records, len(index))
        assert np.array_equal(loaded.suffixes, index.suffixes) and np.array_equal(loaded.tokens, index.tokens)
    assert load_index(str(tmp_path / "store")) is load_index(str(tmp_path / "store"))
    with pytest.raises(ConfigError, match=r"no meta\.json"):
        SuffixIndex.load(tmp_path)
    (tmp_path / "meta.json").write_text(json.dumps({"format": "other"}))
    with pytest.raises(ConfigError, match="is not a version-1"):
        SuffixIndex.load(tmp_path)


def test_build_from_spans_indexes_only_the_spans() -> None:
    tokens = np.arange(10, dtype=np.uint32)
    index = build_from_spans(tokens, np.array([[0, 3], [5, 7]]), depth=4)
    assert index.num_records == 2 and sorted(index.tokens[index.suffixes].tolist()) == [0, 1, 2, 5, 6]


def test_command_line_builds_and_describes_an_index(tmp_path) -> None:
    np.arange(20, dtype=np.uint32).tofile(tmp_path / "tokens.bin")
    np.save(tmp_path / "spans.npy", np.array([[0, 8], [10, 20]]))
    command = [sys.executable, "-m", "economize.techniques.retrieval"]
    out = subprocess.run([*command, "build", str(tmp_path / "store"), "--tokens", str(tmp_path / "tokens.bin"),
                          "--spans", str(tmp_path / "spans.npy"), "--depth", "8"],
                         capture_output=True, text=True, env=subprocess_env(), check=True).stdout
    assert "indexed 18 tokens of 2 records" in out
    info = subprocess.run([*command, "info", str(tmp_path / "store")], capture_output=True, text=True,
                          env=subprocess_env(), check=True).stdout
    assert json.loads(info) == {"tokens": 18, "records": 2, "depth": 8, "vocab": 20}
