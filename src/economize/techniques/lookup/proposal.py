"""Hand several draft tokens per row to the decoding loop at once (drafters that do not draft token by token)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ...decoding.protocols import Draft
from ...decoding.request import DraftRequest
from ...errors import ConfigError

__all__ = ["chain_draft", "one_hot_probs"]


def one_hot_probs(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Point-mass draft distributions ``[B, T, V]`` (float32) on ``tokens`` ``[B, T]`` (``-1`` padding gives zeros).

    Deterministic drafts (lookups, the most likely token of a head) are exact for the ``"rejection"`` verifier when
    their distribution is a point mass on the drafted token."""
    probs = torch.zeros(*tokens.shape, vocab_size, dtype=torch.float32, device=tokens.device)
    valid = tokens >= 0
    probs.scatter_(2, tokens.clamp_min(0)[..., None], valid[..., None].float())
    return probs


def chain_draft(
    request: DraftRequest,
    tokens: torch.Tensor,
    lengths: Sequence[int],
    probs: torch.Tensor | None = None,
    vocab_size: int | None = None,
) -> Draft:
    """Write chain drafts into the request's sequence buffer and return them as a :class:`Draft`.

    Row ``b``'s first ``lengths[b]`` entries of ``tokens`` go to positions ``request.ends[b], ...`` of
    ``request.seq`` (where the exact pass reads them). ``lengths`` are clipped to the request's budgets. Without
    ``probs`` and with a verifier that needs draft distributions, point masses on the tokens are attached
    (``vocab_size`` is then required).

    Args:
        request: The cycle's draft request.
        tokens: Draft tokens ``[B, T]`` (entries past a row's length are ignored).
        lengths: Number of draft tokens of every row.
        probs: The distribution every token was drawn from ``[B, T, V]``, if the drafter sampled.
        vocab_size: Vocabulary size for point-mass distributions.
    """
    lengths = [max(0, min(int(n), budget - done)) for n, budget, done in
               zip(lengths, request.budgets, request.lengths)]
    width = max(lengths, default=0)
    batch, device = request.batch_size, request.device
    if width == 0:
        return Draft.empty(batch, device)
    tokens = tokens[:, :width].to(device, torch.long)
    offsets = torch.arange(width, device=device)
    valid = offsets[None] < torch.tensor(lengths, device=device)[:, None]
    ends = torch.tensor(request.ends, device=device)
    target = torch.where(valid, ends[:, None] + offsets, torch.full_like(tokens, request.scratch))
    request.seq.scatter_(1, target, tokens.to(request.seq.dtype))
    tokens = tokens.masked_fill(~valid, -1)
    for b, n in enumerate(lengths):
        request.lengths[b] += n
    if probs is not None:
        probs = probs[:, :width].float() * valid[..., None]
    elif request.verifier.needs_draft_probs:
        if vocab_size is None:
            raise ConfigError("chain_draft needs vocab_size to build point-mass draft distributions")
        probs = one_hot_probs(tokens, vocab_size)
    return Draft(tokens, list(lengths), None, probs)
