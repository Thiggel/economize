"""KnapSpec (Cha et al., ICML 2026): self-speculative drafting with sublayers skipped, chosen by a knapsack search.

    import economize
    from economize.techniques.knapspec import KnapSpecConfig

    model = economize.apply(hf_model, KnapSpecConfig(), decoding=economize.DecodeConfig(cycle_length=11,
                                                                                         min_confidence=0.7))

The technique is training-free. A draft step runs the model with the attention of some layers and the MLP of others
skipped; every ``interval`` decoded tokens the drafter searches the skip set and the draft length again on the
hidden states of its recent exact passes, weighing each sublayer by its latency measured on the GPU
(:func:`measure_latency`). Families whose layers name their sublayers differently register a
:class:`SublayerLayout` function in :data:`SUBLAYER_LAYOUTS`.
"""

from __future__ import annotations

from .config import KnapSpecConfig
from .drafter import SEARCH_COST, KnapSpecDrafter
from .latency import SublayerLatency, measure_latency
from .search import KnapsackSearch, SearchStats, SearchTokens, SkipPlan
from .sublayer import SUBLAYER_LAYOUTS, PartialLayer, SublayerLayout, SublayerSkipper, default_layout
from .technique import KnapSpecTechnique

__all__ = [
    "SEARCH_COST",
    "SUBLAYER_LAYOUTS",
    "KnapSpecConfig",
    "KnapSpecDrafter",
    "KnapSpecTechnique",
    "KnapsackSearch",
    "PartialLayer",
    "SearchStats",
    "SearchTokens",
    "SkipPlan",
    "SublayerLatency",
    "SublayerLayout",
    "SublayerSkipper",
    "default_layout",
    "measure_latency",
]
