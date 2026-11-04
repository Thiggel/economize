"""SuffixDecoding (Oliaro et al., NeurIPS 2025).

Drafts are grown from the context's longest-scoring match in two suffix structures, one over a datastore of earlier
responses and one over the row's own prompt and output, by the empirical probability of each continuation; the
speculation length adapts to the match length. No model layer runs while drafting.

    import economize
    from economize.techniques.suffix_decoding import SuffixDecodingConfig

    config = SuffixDecodingConfig(datastore="responses/")
    model = economize.apply(hf_model, config, decoding=economize.DecodeConfig(cycle_length=17))

The datastore is a :class:`~economize.techniques.retrieval.SuffixIndex` folder.
"""

from __future__ import annotations

from .config import SuffixDecodingConfig
from .drafter import SuffixDecodingDrafter
from .local import continuation_table, suffix_matches
from .technique import SuffixDecodingTechnique
from .tree import IndexNodes, Nodes, Speculation, TableNodes, speculate

__all__ = [
    "IndexNodes",
    "Nodes",
    "Speculation",
    "SuffixDecodingConfig",
    "SuffixDecodingDrafter",
    "SuffixDecodingTechnique",
    "TableNodes",
    "continuation_table",
    "speculate",
    "suffix_matches",
]
