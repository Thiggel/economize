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
from .graphs import GraphPool, PoolMember, StaticStep
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
from .policy import STEP_FILL, STEP_RUN, STEP_SKIP, LayerAction, LayerPolicy, Route, StepAction, StepOps
from .runner import ExactPass, LayerCall, LayerRunner
from .transfer import device_tensor

__all__ = [
    "CACHES",
    "MASKS",
    "MASK_FORMATS",
    "PHASES",
    "STEP_FILL",
    "STEP_RUN",
    "STEP_SKIP",
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
    "PoolMember",
    "PositionalKVCache",
    "RecurrentSnapshot",
    "RingSpec",
    "Route",
    "StaticKVCache",
    "StaticStep",
    "StepAction",
    "StepOps",
    "TreeLayout",
    "additive_mask",
    "boolean_mask",
    "current_phase",
    "device_tensor",
    "phase",
    "tree_ancestors",
    "visibility",
]
