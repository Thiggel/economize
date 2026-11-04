"""Self-speculative whole-layer greedy skip: draft with the model minus a fixed set of whole layers.

    import economize
    from economize.techniques.layer_skip import LayerSkipConfig

    model = economize.apply(hf_model, LayerSkipConfig(skip_layers=(5, 9, 13)))            # a given set
    model = economize.apply(hf_model, LayerSkipConfig(num_skip=8), calibration=token_ids)  # a searched set

The technique is training-free and approximates Draft & Verify, SWIFT and CLaSp (which skip sub-layers or
re-optimize the set while decoding; see :class:`LayerSkipTechnique`). Searches are registered in
:data:`SKIP_SEARCHES`, scores in :data:`SKIP_METRICS`.
"""

from __future__ import annotations

from .config import LayerSkipConfig
from .drafter import LayerSkipDrafter
from .metrics import SKIP_METRICS, KLMetric, MismatchMetric, SkipMetric
from .policy import SkipLayers
from .search import SKIP_SEARCHES, GreedySearch, SkipScorer, SkipSearch, UniformSearch
from .technique import LayerSkipTechnique

__all__ = [
    "SKIP_METRICS",
    "SKIP_SEARCHES",
    "GreedySearch",
    "KLMetric",
    "LayerSkipConfig",
    "LayerSkipDrafter",
    "LayerSkipTechnique",
    "MismatchMetric",
    "SkipLayers",
    "SkipMetric",
    "SkipScorer",
    "SkipSearch",
    "UniformSearch",
]
