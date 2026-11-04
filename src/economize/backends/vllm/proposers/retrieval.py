"""The datastore drafters (``rest``, ``suffix_decoding``) as vLLM token proposers.

They run economize's own drafters (:class:`~economize.techniques.rest.RestDrafter`,
:class:`~economize.techniques.suffix_decoding.SuffixDecodingDrafter`) on the request contexts, in chain mode: vLLM
verifies one chain per request, so a tree configuration drafts its best path.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ....techniques.rest.config import RestConfig
from ....techniques.retrieval.drafter import RetrievalDrafter
from ....techniques.suffix_decoding.config import SuffixDecodingConfig
from .base import TokenProposer, register_proposer, trailing

__all__ = ["RetrievalProposer", "rest_proposer", "suffix_decoding_proposer"]


class RetrievalProposer(TokenProposer):
    """A :class:`~economize.techniques.retrieval.drafter.RetrievalDrafter` drafting chains.

    Args:
        drafter: The drafter (its :attr:`~RetrievalDrafter.context` becomes :attr:`context`).
    """

    def __init__(self, drafter: RetrievalDrafter) -> None:
        self.drafter = drafter
        self.context = drafter.context

    def propose(self, contexts: Sequence[np.ndarray | None], budgets: Sequence[int]) -> list[list[int]]:
        trimmed = [trailing(c, self.context) for c in contexts]
        proposals = self.drafter.proposals(trimmed, list(budgets), tree=False)
        return [[] if p is None else [int(t) for t in p.tokens] for p in proposals]


@register_proposer("rest")
def rest_proposer(config: RestConfig, vocab_size: int) -> RetrievalProposer:
    """REST over its datastore folder (loaded once per worker)."""
    from ....techniques.rest.drafter import RestDrafter
    from ....techniques.retrieval import load_index

    return RetrievalProposer(RestDrafter(config, load_index(config.datastore), vocab_size))


@register_proposer("suffix_decoding")
def suffix_decoding_proposer(config: SuffixDecodingConfig, vocab_size: int) -> RetrievalProposer:
    """Suffix decoding over the request's own tokens and, when set, a prebuilt datastore."""
    from ....techniques.retrieval import load_index
    from ....techniques.suffix_decoding.drafter import SuffixDecodingDrafter

    index = load_index(config.datastore) if config.datastore else None
    return RetrievalProposer(SuffixDecodingDrafter(config, index, vocab_size))
