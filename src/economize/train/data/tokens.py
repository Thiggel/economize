"""Flat token files: random training windows (:class:`TokenStream`) and evenly spaced evaluation windows
(:func:`fixed_windows`)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...errors import ConfigError
from ..objective import Batch

__all__ = ["TokenStream", "fixed_windows"]


class TokenStream:
    """Random windows of ``seq_len`` tokens from a flat ``uint32`` token file.

    Draws the windows of the reference implementation in ``tests/train/test_data_sources.py`` for the same file,
    length and seed: start offsets come from ``numpy.random.default_rng(seed).integers``, anywhere in the file, or
    at multiples of ``seq_len`` with ``aligned`` (files of whole records).

    Args:
        path: The token file (``numpy.memmap`` with dtype ``dtype``).
        seq_len: Tokens per row ``T``.
        batch_size: Rows per batch ``B``.
        seed: Seed of the window positions.
        aligned: Start windows at multiples of ``seq_len`` only.
        dtype: Token dtype of the file.
    """

    def __init__(
        self,
        path: str | Path,
        seq_len: int,
        batch_size: int,
        seed: int = 0,
        *,
        aligned: bool = False,
        dtype: Any = np.uint32,
    ) -> None:
        self.path = str(path)
        self.tokens = np.memmap(self.path, dtype=dtype, mode="r")
        if self.tokens.size <= seq_len + 1:
            raise ConfigError(f"{self.path} holds {self.tokens.size} tokens, fewer than one window of {seq_len + 1}")
        self.seq_len, self.batch_size, self.aligned, self.seed = seq_len, batch_size, aligned, seed
        self.rng = np.random.default_rng(seed)

    def shard(self, rank: int, world: int) -> None:
        """Draw the windows of process ``rank`` of ``world``: positions seeded with ``seed + rank``, independent of
        the other processes' (call before the first batch)."""
        if world > 1:
            self.rng = np.random.default_rng(self.seed + rank)

    def next_batch(self) -> Batch:
        size, length = self.tokens.size, self.seq_len
        if self.aligned:
            starts = self.rng.integers(0, size // length, self.batch_size) * length
        else:
            starts = self.rng.integers(0, size - length - 1, self.batch_size)
        rows = np.stack([self.tokens[i : i + length].astype(np.int64) for i in starts])
        return Batch(torch.from_numpy(rows))

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng"]


def fixed_windows(path: str | Path, n: int, seq_len: int, *, dtype: Any = np.uint32) -> torch.Tensor:
    """``n`` evenly spaced windows ``[n, seq_len]`` of a token file (as the reference implementation in
    ``tests/train/test_data_sources.py``)."""
    tokens = np.memmap(str(path), dtype=dtype, mode="r")
    stride = (tokens.size - seq_len - 1) // n
    if stride < 1:
        raise ConfigError(f"{path} is too short for {n} windows of {seq_len} tokens")
    return torch.from_numpy(np.stack([tokens[i * stride : i * stride + seq_len].astype(np.int64) for i in range(n)]))
