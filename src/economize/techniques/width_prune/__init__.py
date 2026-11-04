"""Structured width pruning (``width_prune``): remove attention head groups, FFN channels and residual channels by
the importance of FLAP (An et al., AAAI 2024), LLM-Pruner (Ma et al., NeurIPS 2023) or Minitron (Muralidharan et
al., NeurIPS 2024), which leaves a smaller dense model. Importing it also registers Minitron's contiguous-window
depth criterion as the ``depth_prune`` method ``"minitron"`` (:mod:`.depth`)."""

from .config import STRUCTURES, WidthPruneConfig
from .criteria import FLAP, WIDTH_CRITERIA, LLMPruner, Minitron, Scores, WidthCriterion, standardize
from .depth import MinitronDepth, contiguous_windows, minitron_depth_select
from .prune import compensation_bias, describes, group_channels, prune_hidden, prune_layer
from .recovery import RecoveryLoRA, merge_lora
from .selection import Selection, keep_count, select_adaptive, select_uniform
from .stats import ChannelMoments, LayerShape, SampleNorm, layer_shape
from .technique import WidthPruneTechnique

__all__ = [
    "FLAP",
    "STRUCTURES",
    "WIDTH_CRITERIA",
    "ChannelMoments",
    "LLMPruner",
    "LayerShape",
    "Minitron",
    "MinitronDepth",
    "RecoveryLoRA",
    "SampleNorm",
    "Scores",
    "Selection",
    "WidthCriterion",
    "WidthPruneConfig",
    "WidthPruneTechnique",
    "compensation_bias",
    "contiguous_windows",
    "describes",
    "group_channels",
    "keep_count",
    "layer_shape",
    "merge_lora",
    "minitron_depth_select",
    "prune_hidden",
    "prune_layer",
    "select_adaptive",
    "select_uniform",
    "standardize",
]
