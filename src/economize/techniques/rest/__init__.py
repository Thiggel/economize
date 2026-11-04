"""REST, retrieval-based speculative decoding (He et al., 2024).

Drafts are the continuations of the context's longest suffix in a datastore of text, merged into a trie whose most
frequent nodes are verified as one tree. No model layer runs while drafting.

    import economize
    from economize.techniques.rest import RestConfig

    model = economize.apply(hf_model, RestConfig(datastore="store/"), decoding=economize.DecodeConfig(cycle_length=11))

The datastore is a :class:`~economize.techniques.retrieval.SuffixIndex` folder.
"""

from __future__ import annotations

from .config import OCCURRENCE_RULES, OccurrenceRule, RestConfig
from .drafter import RestDrafter
from .technique import RestTechnique
from .trie import Trie, build_trie

__all__ = ["OCCURRENCE_RULES", "OccurrenceRule", "RestConfig", "RestDrafter", "RestTechnique", "Trie", "build_trie"]
