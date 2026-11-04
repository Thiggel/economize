"""FR-Spec: draft over a vocabulary of frequent tokens, verify over the full one (Zhao et al., 2025).

Example::

    import economize
    from economize.techniques.early_exit import EarlyExitConfig
    from economize.techniques.frspec import FRSpecConfig

    config = FRSpecConfig(drafter=EarlyExitConfig(exit_layer=12), vocab_size=32768)
    model = economize.apply(base, config, calibration=corpus_ids)  # the 32768 most frequent ids of corpus_ids

The published FR-Spec vocabularies (``freq_32768.pt``) load with :func:`read_token_ids`.
"""

from .config import FRSpecConfig
from .drafter import FRSpecDrafter, restrict_logits, widen_logits
from .technique import FRSpecTechnique
from .vocab import frequency_order, read_token_ids, token_counts

__all__ = ["FRSpecConfig", "FRSpecDrafter", "FRSpecTechnique", "frequency_order", "read_token_ids", "restrict_logits",
           "token_counts", "widen_logits"]
