"""The drafter of prompt-lookup decoding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...decoding.protocols import Draft, Drafter
from .matching import match_ngrams
from .proposal import chain_draft

if TYPE_CHECKING:
    from ...decoding.request import DraftRequest
    from ...runtime import ExactPass, KVCache, LayerRunner
    from .config import LookupConfig

__all__ = ["LookupDrafter"]


class LookupDrafter(Drafter):
    """Drafts the tokens that followed an earlier occurrence of every row's last n-gram (see :class:`LookupConfig`).

    The drafter runs no model layer and keeps no state: :meth:`propose` reads the committed tokens from the
    request's buffer, makes one host synchronization for the match table and writes the copied tokens back. It
    records no cost event (the cost log lists model calls); the acceptance log of the result still lists every
    proposal. Lookup drafts are deterministic, so with the ``"rejection"`` verifier they carry point-mass
    distributions.

    Args:
        config: The lookup configuration.
        vocab_size: Vocabulary size of the model (for point-mass draft distributions).
    """

    layer_span = (1, 0)

    def __init__(self, config: LookupConfig, vocab_size: int) -> None:
        self.config, self.vocab_size = config, vocab_size

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        pass

    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None:
        pass

    def lookup(self, request: DraftRequest) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        """Match every row and return its proposal length, the matched lengths ``[B]`` and the proposed tokens
        ``[B, T]`` (without writing them)."""
        c = self.config
        matched, start = match_ngrams(request.seq, request.ends, c.max_ngram, c.min_ngram, c.match)
        table = torch.stack([matched, start]).tolist()
        counts = []
        for b, (n, first) in enumerate(zip(*table)):
            room = request.budgets[b] - request.lengths[b]
            if c.max_proposal is not None:
                room = min(room, c.max_proposal)
            counts.append(max(0, min(room, request.ends[b] - first)) if n > 0 else 0)
        width = max(counts, default=0)
        offsets = torch.arange(width, device=request.device)
        source = (start[:, None].to(request.device) + offsets).clamp(max=request.scratch)
        return counts, matched, request.seq.gather(1, source)

    def propose_partial(self, request: DraftRequest) -> Draft:
        """The lookup drafts; rows without a match have none (the primary-drafter role of a hybrid,
        :class:`~economize.techniques._drafting.PartialDrafter`)."""
        counts, _, tokens = self.lookup(request)
        return chain_draft(request, tokens, counts, vocab_size=self.vocab_size)

    def propose(self, request: DraftRequest) -> Draft:
        return self.propose_partial(request)
