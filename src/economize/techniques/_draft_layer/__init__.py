"""The one-layer draft decoder that EAGLE-1/2 (:mod:`economize.techniques.eagle`) and EAGLE-3
(:mod:`economize.techniques.eagle3`) share: its config fields, the layer's attention and MLP, the technique's rotary
table, training-time test and training interfaces, and the drafter with its own KV cache, CUDA graphs and dynamic
trees.

A new drafter of this kind subclasses :class:`DraftLayerConfig`, :class:`DraftLayer` (how the source and the token
embedding enter the layer), :class:`DraftLayerTechnique` (which states feed it, how its outputs are read out) and,
for a fused greedy head, :class:`DraftLayerDrafter`. The package name starts with an underscore, so it registers no
technique of its own.
"""

from .config import DraftLayerConfig
from .drafter import DraftLayerDrafter
from .layer import DraftAttention, DraftLayer
from .technique import DraftLayerShape, DraftLayerTechnique, last_full_attention_layer, target_bias

__all__ = [
    "DraftAttention",
    "DraftLayer",
    "DraftLayerConfig",
    "DraftLayerDrafter",
    "DraftLayerShape",
    "DraftLayerTechnique",
    "last_full_attention_layer",
    "target_bias",
]
