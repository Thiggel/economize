"""Architecture adapters (one per model family, looked up by ``config.model_type``)."""

from __future__ import annotations

from .base import (
    ARCHITECTURES,
    SUPPORTED_ATTENTION,
    ArchAdapter,
    adapter_for,
    module_device,
    register_adapter,
)
from .config import (
    PER_LAYER_CONFIG_FIELDS,
    HFConfig,
    config_value,
    is_heterogeneous,
    kept_below,
    layer_overrides,
    renumbered,
    set_layer_overrides,
    text_config,
)
from .protocols import (
    AttentionLayout,
    DecoderLayer,
    DecoderModules,
    DepthEditing,
    DevicePlacement,
    KVProjection,
    LayerExecution,
    LayerInputs,
    LogitHead,
    ModelSizes,
    RotaryPositions,
    WeightLayout,
)
from .spec import (
    LAYER_KINDS,
    AttentionSpec,
    HeadGeometry,
    LayerKind,
    PassContext,
    StateKind,
    layer_kind_of,
    register_layer_kind,
)
from .standard import SoftcappedLogitsMixin, SparseStepMoeMixin, StandardDecoderAdapter

__all__ = [
    "ARCHITECTURES",
    "LAYER_KINDS",
    "PER_LAYER_CONFIG_FIELDS",
    "SUPPORTED_ATTENTION",
    "ArchAdapter",
    "AttentionLayout",
    "AttentionSpec",
    "DecoderLayer",
    "DecoderModules",
    "DepthEditing",
    "DevicePlacement",
    "HFConfig",
    "HeadGeometry",
    "KVProjection",
    "LayerExecution",
    "LayerInputs",
    "LayerKind",
    "LogitHead",
    "ModelSizes",
    "PassContext",
    "RotaryPositions",
    "SoftcappedLogitsMixin",
    "SparseStepMoeMixin",
    "StandardDecoderAdapter",
    "StateKind",
    "WeightLayout",
    "adapter_for",
    "config_value",
    "is_heterogeneous",
    "kept_below",
    "layer_kind_of",
    "layer_overrides",
    "module_device",
    "register_adapter",
    "register_layer_kind",
    "renumbered",
    "set_layer_overrides",
    "text_config",
]

from . import families  # noqa: F401  (registers every family adapter)
