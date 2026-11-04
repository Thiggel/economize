"""Frequency ranking of token ids for FR-Spec draft vocabularies."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch

from ...core.calibration import token_batches
from ...errors import ConfigError

__all__ = ["frequency_order", "read_token_ids", "token_counts"]


def token_counts(tokens: object, vocab_size: int) -> torch.Tensor:
    """How often every token id occurs in ``tokens`` (anything :func:`~economize.core.calibration.token_batches`
    accepts: a tensor, an iterable of tensors or a callable returning either), ``[vocab_size]`` int64. Ids at or
    above ``vocab_size`` are an error."""
    counts = torch.zeros(vocab_size, dtype=torch.long)
    batches: Iterable[torch.Tensor] = token_batches(tokens)
    for batch in batches:
        flat = batch.reshape(-1).cpu()
        if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= vocab_size):
            raise ConfigError(f"token ids must lie in [0, {vocab_size}), got {int(flat.min())}..{int(flat.max())}")
        counts += torch.bincount(flat, minlength=vocab_size)
    return counts


def frequency_order(tokens: object, vocab_size: int) -> torch.Tensor:
    """All ``vocab_size`` token ids, most frequent in ``tokens`` first; ties keep the smaller id first (a stable
    sort; it matches the reference implementation in ``tests/oracle/test_drafter_frspec_oracle.py``). The first ``K``
    entries are an FR-Spec vocabulary."""
    return torch.sort(-token_counts(tokens, vocab_size), stable=True).indices


def read_token_ids(path: str | Path) -> tuple[int, ...]:
    """The draft ids of a token list file of the FR-Spec release (``freq_32768.pt`` of
    ``thunlp/LLaMA3-Instruct-8B-FR-Spec`` and its siblings): a ``torch.save``d list or 1-D tensor of distinct ids.

    The published lists are sorted by id, not by frequency, so they fix the draft vocabulary but not a ranking
    (truncating one to a smaller vocabulary does not give the most frequent ids). Pass the result as
    ``FRSpecConfig(vocab_size=len(ids), token_ids=ids)``.
    """
    raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(raw, torch.Tensor):
        if raw.dim() != 1 or raw.is_floating_point() or raw.is_complex():
            raise ConfigError(f"{path}: expected a 1-D integer tensor of token ids, got {tuple(raw.shape)} {raw.dtype}")
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)) or not all(isinstance(i, int) and not isinstance(i, bool) for i in raw):
        raise ConfigError(f"{path}: expected a list of token ids, got {type(raw).__name__}")
    if len(set(raw)) != len(raw) or (raw and min(raw) < 0):
        raise ConfigError(f"{path}: token ids must be distinct and non-negative")
    return tuple(raw)
