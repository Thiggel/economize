"""Execution runtime: slot-addressed caches, attention masks, layer contexts, the layer runner, CUDA graphs and
non-blocking host-to-device transfers."""

from __future__ import annotations

from .cache import (
    CACHES,
    CacheSpec,
    HybridCache,
    KVCache,
    PositionalKVCache,
    RecurrentSnapshot,
    RingSpec,
    StaticKVCache,
)
from .context import ContextLayout, LayerContext
from .graphs import GraphPool, StaticStep
from .mask import (
    MASK_FORMATS,
    MASKS,
    STRUCTURED_MASKS,
    TreeLayout,
    additive_mask,
    boolean_mask,
    tree_ancestors,
    visibility,
)
from .phase import PHASES, Phase, current_phase, phase
from .policy import LayerAction, LayerPolicy, Route
from .runner import ExactPass, LayerCall, LayerRunner
from .transfer import device_tensor

__all__ = [
    "CACHES",
    "MASKS",
    "MASK_FORMATS",
    "PHASES",
    "STRUCTURED_MASKS",
    "CacheSpec",
    "ContextLayout",
    "ExactPass",
    "GraphPool",
    "HybridCache",
    "KVCache",
    "LayerAction",
    "LayerCall",
    "LayerContext",
    "LayerPolicy",
    "LayerRunner",
    "Phase",
    "PositionalKVCache",
    "RecurrentSnapshot",
    "RingSpec",
    "Route",
    "StaticKVCache",
    "StaticStep",
    "TreeLayout",
    "additive_mask",
    "boolean_mask",
    "current_phase",
    "device_tensor",
    "phase",
    "tree_ancestors",
    "visibility",
]
