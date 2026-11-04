"""Prompt-lookup (n-gram) drafting.

Drafts are the tokens that followed the earliest (or most recent) earlier occurrence of the last n committed tokens
(Saxena, 2023; vLLM's ``"ngram"`` speculative method). No model layer runs while drafting.

    import economize
    from economize.techniques.lookup import LookupConfig

    model = economize.apply(hf_model, LookupConfig(max_ngram=3), decoding=economize.DecodeConfig(cycle_length=6))
"""

from __future__ import annotations

from .config import LookupConfig
from .drafter import LookupDrafter
from .matching import MATCH_RULES, MatchRule, match_ngrams
from .proposal import chain_draft, one_hot_probs
from .technique import LookupTechnique

__all__ = [
    "MATCH_RULES",
    "LookupConfig",
    "LookupDrafter",
    "LookupTechnique",
    "MatchRule",
    "chain_draft",
    "match_ngrams",
    "one_hot_probs",
]
