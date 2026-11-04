"""economize's token-level drafters on vLLM, through one ``speculative_config.method="custom_class"`` proposer.

=====================  ==============================================================================
technique              proposer
=====================  ==============================================================================
``rest``               :class:`~.retrieval.RetrievalProposer` over the REST datastore (best path)
``suffix_decoding``    :class:`~.retrieval.RetrievalProposer` (prebuilt datastore and/or own tokens)
``lookup``             :class:`~.lookup.LookupProposer` (any match rule; vLLM's ``ngram`` is earliest-only)
``lookahead``          :class:`~.lookup.LookaheadPoolProposer` (the n-gram pool from the committed tokens)
=====================  ==============================================================================

See :mod:`.base` for the interface. Every module of this package registers its proposers on import, so a new
module needs no edit here. Importing this package does not import vLLM.
"""

from __future__ import annotations

from ....registry import import_submodules
from .base import (
    CONFIG_KEY,
    PROPOSER_CLASS,
    PROPOSERS,
    EconomizeProposer,
    ProposerFactory,
    TokenProposer,
    build_proposer,
    proposer_entry,
    proposer_for,
    register_proposer,
    trailing,
)

__all__ = [
    "CONFIG_KEY",
    "PROPOSERS",
    "PROPOSER_CLASS",
    "EconomizeProposer",
    "ProposerFactory",
    "TokenProposer",
    "build_proposer",
    "proposer_entry",
    "proposer_for",
    "register_proposer",
    "trailing",
]

import_submodules(__name__, __path__, packages_only=False)
