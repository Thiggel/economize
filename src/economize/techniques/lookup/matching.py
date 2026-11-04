"""Vectorized n-gram matching over the token buffer of a batch."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

import torch

from ...validation import check_choice

__all__ = ["MATCH_RULES", "MatchRule", "match_ngrams"]

#: Which earlier occurrence of the suffix n-gram is copied from.
MatchRule = Literal["earliest", "latest"]
MATCH_RULES: Final[tuple[str, ...]] = ("earliest", "latest")


def match_ngrams(
    seq: torch.Tensor, ends: Sequence[int], max_ngram: int, min_ngram: int, match: MatchRule = "earliest"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find, for every row, an earlier occurrence of its last ``n`` tokens, longest ``n`` first.

    Row ``b`` holds ``ends[b]`` tokens ``seq[b, :ends[b]]``. An occurrence at ``i`` counts when
    ``seq[b, i:i+n]`` equals the suffix ``seq[b, ends[b]-n:ends[b]]`` and a token follows it (``i + n < ends[b]``,
    which also excludes the suffix itself). Among the occurrences of the longest matching ``n``, ``match`` picks
    the first one (``"earliest"``, as in Saxena's prompt lookup, Hugging Face's ``prompt_lookup_num_tokens`` and
    vLLM's ``"ngram"`` method) or the most recent one (``"latest"``, which matches the reference implementation in
    ``tests/oracle/test_drafter_lookup_oracle.py``).

    Args:
        seq: Token buffer ``[B, C]``.
        ends: Number of tokens of every row.
        max_ngram: Longest suffix tried.
        min_ngram: Shortest suffix tried.
        match: ``"earliest"`` or ``"latest"`` occurrence.

    Returns:
        The matched length ``n`` (``0`` for no match) and the position of the first token after the match
        (``0`` for no match), both ``[B]`` LongTensors on ``seq``'s device.
    """
    check_choice("match", match, MATCH_RULES)
    device = seq.device
    batch = seq.shape[0]
    matched = torch.zeros(batch, dtype=torch.long, device=device)
    start = torch.zeros_like(matched)
    width = max(ends, default=0)
    if width < min_ngram + 1:
        return matched, start
    ends_t = torch.as_tensor(list(ends), dtype=torch.long, device=device)
    window = seq[:, :width]
    for n in range(max_ngram, min_ngram - 1, -1):
        if width < n + 1:
            continue
        grams = window.unfold(1, n, 1)  # [B, width - n + 1, n]
        suffix = window.gather(1, (ends_t[:, None] - n + torch.arange(n, device=device)).clamp(min=0))
        index = torch.arange(grams.shape[1], device=device)
        hit = (grams == suffix[:, None, :]).all(-1) & (index[None] + n < ends_t[:, None])
        if match == "latest":
            found = torch.where(hit, index[None], -1).amax(1)
        else:
            found = torch.where(hit, index[None], width).amin(1)
        take = (matched == 0) & (found >= 0) & (found < width)
        start = torch.where(take, found + n, start)
        matched = torch.where(take, torch.full_like(matched, n), matched)
    return matched, start
