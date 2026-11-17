"""Draft views: any technique that declares how it runs on a draft path, used as a lossless self-speculative drafter.

A technique applied to the model changes its outputs (TEAL, weight pruning and quantization, depth pruning, KV-cache
eviction). Held by :func:`DraftWith`, the same technique changes only the drafter: each draft step runs the model as
the technique changes it, and the unmodified model verifies the drafted tokens, so greedy decoding returns the base
model's tokens::

    import economize
    from economize.techniques.draft_with import DraftWith
    from economize.techniques.prune import PruneConfig
    from economize.techniques.sparsity import TealConfig

    model = economize.apply(hf_model, DraftWith(TealConfig(sparsity=0.5), PruneConfig(sparsity=0.5)),
                            calibration=token_ids)
"""

from .config import DraftWith, DraftWithConfig
from .drafter import DraftWithDrafter
from .technique import DraftWithTechnique

__all__ = ["DraftWith", "DraftWithConfig", "DraftWithDrafter", "DraftWithTechnique"]
