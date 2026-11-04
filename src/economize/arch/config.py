"""Reading and rewriting Hugging Face configs: the text sub-config, global values of heterogeneous configs and the
fields that address decoder layers by index.

Heterogeneous configs (transformers >= 5.17, ``HeterogeneousConfigMixin``) let single layers override global
attributes; Gemma-4's full-attention layers override ``head_dim`` and ``num_key_value_heads`` this way. Such a config
refuses plain reads of an overridden attribute. Every access economize makes to this machinery goes through the helpers
below, so a change of the transformers API is absorbed here.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TypeAlias

__all__ = [
    "PER_LAYER_CONFIG_FIELDS",
    "HFConfig",
    "config_value",
    "is_heterogeneous",
    "kept_below",
    "layer_overrides",
    "renumbered",
    "set_layer_overrides",
    "text_config",
]

#: A Hugging Face ``PretrainedConfig`` or a stand-in with the same attributes. Its fields differ per family and are
#: read by name (``config.head_dim``, :func:`config_value`), so the alias stays dynamic; it names what is passed.
HFConfig: TypeAlias = Any

#: Config fields that hold one entry per decoder layer; :meth:`ArchAdapter.remove_layers` drops the entries of
#: removed layers. Families with more such fields extend :attr:`ArchAdapter.per_layer_config_fields`.
PER_LAYER_CONFIG_FIELDS: tuple[str, ...] = ("layer_types", "no_rope_layers")


def text_config(config: HFConfig) -> HFConfig:
    """The config of the text decoder (``config`` itself for a plain causal LM, the text sub-config of a wrapper)."""
    getter = getattr(config, "get_text_config", None)
    return getter() if callable(getter) else config


def is_heterogeneous(config: HFConfig) -> bool:
    """Whether ``config`` gives some layers attribute values of their own (``per_layer_config``)."""
    return bool(getattr(config, "is_heterogeneous", False))


def config_value(config: HFConfig, name: str, default: Any = None) -> Any:  # a field's type depends on the field
    """``config.<name>`` or ``default``, also for a per-layer attribute of a heterogeneous config.

    There it is the global value, the one of every layer without an override. Adapters that give some layers other
    sizes say so in :meth:`ArchAdapter.head_geometry`.
    """
    if not is_heterogeneous(config):
        return getattr(config, name, default)
    # transformers 5.17 reads the global value without the per-layer check through this private method (its own
    # serialization uses it). The public route below toggles a flag on the config and logs a warning.
    read = getattr(config, "_getattr_without_heterogeneous_validation", None)
    if callable(read):
        return read(name, default)
    previous = config.allow_global_per_layer_attribute_access
    config.allow_global_per_layer_attribute_access = True
    try:
        return getattr(config, name, default)
    finally:
        config.allow_global_per_layer_attribute_access = previous


def layer_overrides(config: HFConfig) -> dict[int, dict[str, Any]]:
    """The attributes each layer overrides, by 0-indexed layer (``{}`` for a homogeneous config).

    Read from the config's serialized form (``to_dict()["per_layer_config"]``), the public view of the overrides.
    """
    if not is_heterogeneous(config):
        return {}
    serialized: Mapping[str, Mapping[str, Any]] = config.to_dict().get("per_layer_config") or {}
    return {int(index): dict(overrides) for index, overrides in serialized.items()}


def set_layer_overrides(config: HFConfig, overrides: Mapping[int, Mapping[str, Any]]) -> None:
    """Replace the per-layer overrides of a heterogeneous ``config`` (0-indexed layers of the current
    ``num_hidden_layers``). transformers validates the indices and drops values equal to the global ones."""
    config.per_layer_config = {index: copy.deepcopy(dict(values)) for index, values in overrides.items()}


def kept_below(kept: Sequence[int], threshold: int) -> int:
    """How many of the kept layers (0-indexed old indices) lie below ``threshold``: the new value of a field that
    splits the layers at an index (``first_k_dense_replace``, ``max_window_layers``) after a removal."""
    return sum(1 for i in kept if i < threshold)


def renumbered(indices: Iterable[int], kept: Sequence[int]) -> list[int]:
    """The new 0-indexed positions of the kept layers among ``indices`` (old 0-indexed); removed layers drop out."""
    position = {old: new for new, old in enumerate(kept)}
    return sorted(position[i] for i in indices if i in position)
