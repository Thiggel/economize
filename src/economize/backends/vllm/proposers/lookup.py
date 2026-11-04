"""Prompt lookup (``lookup``) and the n-gram pool of lookahead decoding (``lookahead``) as vLLM token proposers.

vLLM's own ``ngram`` method serves ``lookup`` with ``match="earliest"``; this proposer serves every other lookup
configuration with economize's matching rule (:func:`~economize.techniques.lookup.matching.match_ngrams`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from ....techniques.lookahead.config import LookaheadConfig
from ....techniques.lookup.config import LookupConfig
from ....techniques.lookup.matching import match_ngrams
from .base import TokenProposer, register_proposer

__all__ = ["LookaheadPoolProposer", "LookupProposer"]


@register_proposer("lookup")
class LookupProposer(TokenProposer):
    """Copy the tokens after an earlier occurrence of the context's last n-gram (longest n-gram first).

    Args:
        config: The lookup configuration (n-gram range, proposal cap, match rule).
        vocab_size: Unused.
    """

    def __init__(self, config: LookupConfig, vocab_size: int = 0) -> None:
        self.config = config

    def propose(self, contexts: Sequence[np.ndarray | None], budgets: Sequence[int]) -> list[list[int]]:
        c = self.config
        out: list[list[int]] = []
        for context, budget in zip(contexts, budgets):
            if context is None or budget <= 0:
                out.append([])
                continue
            seq = torch.from_numpy(np.ascontiguousarray(context, dtype=np.int64))[None]
            matched, start = match_ngrams(seq, [len(context)], c.max_ngram, c.min_ngram, c.match)
            if int(matched[0]) == 0:
                out.append([])
                continue
            room = budget if c.max_proposal is None else min(budget, c.max_proposal)
            first = int(start[0])
            out.append([int(t) for t in context[first : first + room]])
        return out


@register_proposer("lookahead")
class LookaheadPoolProposer(TokenProposer):
    """The n-gram pool of lookahead decoding, filled from the committed tokens (``pool_from_prompt``), as chains.

    The pool keys every committed ``ngram``-gram by its first token and keeps the ``guesses`` most recent
    continuations per key. The proposal is the most recent continuation of the last committed token (the
    newest guess of the HF verification branch), cut to the budget. The Jacobi window, which adds the n-grams the
    model guesses, has no vLLM counterpart: vLLM's forward holds the verified tokens only.

    Args:
        config: The lookahead configuration (``ngram`` and ``guesses`` are read).
        vocab_size: Unused.
    """

    def __init__(self, config: LookaheadConfig, vocab_size: int = 0) -> None:
        self.config = config

    def propose(self, contexts: Sequence[np.ndarray | None], budgets: Sequence[int]) -> list[list[int]]:
        n = self.config.ngram
        out: list[list[int]] = []
        for context, budget in zip(contexts, budgets):
            if context is None or budget <= 0 or self.config.pool_size == 0 or len(context) < n + 1:
                out.append([])
                continue
            # pooled n-grams start at i <= end - n; the newest one keyed by the root is the last such i
            keys = np.flatnonzero(context[: len(context) - n + 1] == context[-1])
            if len(keys) == 0:
                out.append([])
                continue
            i = int(keys[-1])
            out.append([int(t) for t in context[i + 1 : i + 1 + min(n - 1, budget)]])
        return out
