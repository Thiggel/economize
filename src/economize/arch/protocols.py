"""The facets of an architecture adapter, one small protocol each.

:class:`~economize.arch.ArchAdapter` implements all of them, with defaults that follow the Hugging Face conventions
most families share, so a family adapter overrides only the facets in which its model differs. Code that needs one
aspect of a model types its argument with that facet (a logit transform needs a :class:`LogitHead`, a weight
rewrite a :class:`WeightLayout`), which keeps it usable with any object that provides the facet, including test
doubles and adapters of other libraries.

* :class:`DecoderModules`: embedding, decoder layers, final norm and LM head. Every adapter provides these four
  modules (the abstract properties of :class:`~economize.arch.ArchAdapter`).
* :class:`ModelSizes`: widths, head geometry and dtype. Families with per-layer head shapes (Gemma-4) override
  :meth:`~ModelSizes.head_geometry`.
* :class:`DevicePlacement`: where the embedding, each layer and the head compute (``device_map``, tensor
  parallelism).
* :class:`AttentionLayout`: layer types (full, sliding, chunked, recurrent), windows, KV sharing and the cache kind.
* :class:`RotaryPositions`: the rotary embedding and its tables. A family without one (NoPE, Nemotron-H) needs nothing.
* :class:`LayerExecution`: how one decoder layer is called (its keyword arguments, per-layer inputs).
* :class:`KVProjection`: the K/V a layer would write, without running the layer (used by layer skipping).
* :class:`LogitHead`: final norm, LM head and the family's logit transform (Gemma's softcap).
* :class:`WeightLayout`: token mixer, linear maps, LoRA targets and fused MoE experts of a layer.
* :class:`DepthEditing`: removing layers and renumbering the config fields that address layers by index.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

import torch
from torch import nn

from .config import HFConfig
from .spec import AttentionSpec, HeadGeometry, PassContext, StateKind

if TYPE_CHECKING:
    from ..attention.banded import StructuredMask

__all__ = [
    "AttentionLayout",
    "DecoderLayer",
    "DecoderModules",
    "DepthEditing",
    "DevicePlacement",
    "KVProjection",
    "LayerExecution",
    "LayerInputs",
    "LogitHead",
    "ModelSizes",
    "RotaryPositions",
    "WeightLayout",
]


#: A decoder layer as :meth:`LayerExecution.call_layer` runs it: an HF decoder layer module or a stand-in callable
#: ``layer(hidden, *args, **kwargs)`` that returns the new hidden state, alone or first in a tuple.
DecoderLayer: TypeAlias = Callable[..., torch.Tensor | tuple[torch.Tensor, ...]]

#: Positional arguments after ``hidden`` and keyword arguments of a decoder-layer call (masks, rotary tables,
#: caches: passed through to the family's layer unread).
LayerInputs: TypeAlias = tuple[tuple[object, ...], dict[str, object]]

@runtime_checkable
class DecoderModules(Protocol):
    """The modules of a decoder-only LM. Layers are 1-indexed; index 0 denotes the token embedding."""

    model: nn.Module
    config: HFConfig  # the text decoder's config

    @property
    def model_type(self) -> str: ...

    @property
    def layers(self) -> nn.ModuleList:
        """The decoder layers, 0-indexed as stored in the model."""
        ...

    @property
    def embed_tokens(self) -> nn.Module: ...

    @property
    def final_norm(self) -> nn.Module: ...

    @property
    def lm_head(self) -> nn.Module: ...

    def layer(self, index: int) -> nn.Module:
        """Decoder layer ``index`` (1-indexed)."""
        ...

    def module_path(self, module: nn.Module) -> str:
        """The dotted name of ``module`` inside the model, as a state dict names it."""
        ...


@runtime_checkable
class ModelSizes(Protocol):
    """Widths and head geometry of the model and of each layer."""

    @property
    def num_layers(self) -> int: ...

    @property
    def hidden_size(self) -> int: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def num_kv_heads(self) -> int: ...

    @property
    def head_dim(self) -> int: ...

    @property
    def dtype(self) -> torch.dtype: ...

    def layer_config(self, index: int) -> HFConfig:
        """The config of layer ``index`` (its own entry of a heterogeneous config, else the model's)."""
        ...

    def head_geometry(self, index: int) -> HeadGeometry: ...


@runtime_checkable
class DevicePlacement(Protocol):
    """Where each part of the model computes."""

    @property
    def device(self) -> torch.device:
        """Where every pass starts (the token embedding's device)."""
        ...

    @property
    def head_device(self) -> torch.device: ...

    @property
    def rotary_device(self) -> torch.device: ...

    @property
    def tensor_parallel_size(self) -> int: ...

    def layer_device(self, index: int) -> torch.device: ...


@runtime_checkable
class AttentionLayout(Protocol):
    """What each layer attends over and what it keeps in the cache."""

    @property
    def attn_implementation(self) -> str: ...

    @property
    def layer_types(self) -> list[str]: ...

    @property
    def cache_kind(self) -> str:
        """Key of :data:`economize.runtime.CACHES` the model needs."""
        ...

    def attention_spec(self, index: int) -> AttentionSpec:
        """Mask kind, window and rotary key of layer ``index``."""
        ...

    def state_kind(self, index: int) -> StateKind: ...

    def kv_source(self, index: int) -> int:
        """The layer whose cache entries layer ``index`` attends over (itself unless it shares K/V)."""
        ...


@runtime_checkable
class RotaryPositions(Protocol):
    """Rotary position tables, one per rotary key."""

    @property
    def rotary_emb(self) -> nn.Module | None:
        """The rotary embedding module, ``None`` for a model without one."""
        ...

    def rope_key(self, layer_type: str) -> str: ...

    def position_embeddings(self, like: torch.Tensor, positions: torch.Tensor, rope: str = "default") -> object:
        """The table layers receive as ``position_embeddings`` (``None`` without a rotary embedding)."""
        ...


@runtime_checkable
class LayerExecution(Protocol):
    """How the token embedding and one decoder layer are run inside a pass."""

    @property
    def uses_per_layer_inputs(self) -> bool: ...

    def embed(self, input_ids: torch.Tensor, ctx: PassContext | None = None) -> torch.Tensor: ...

    def layer_inputs(
        self,
        index: int,
        hidden: torch.Tensor,
        ctx: PassContext,
        cache: object,
        mask: torch.Tensor | StructuredMask | None,
    ) -> LayerInputs:
        """Positional and keyword arguments after ``hidden`` for calling layer ``index``."""
        ...

    def call_layer(self, index: int, layer: DecoderLayer, hidden: torch.Tensor,
                   ctx: PassContext) -> torch.Tensor: ...

    def layer_call_key(self, index: int) -> Hashable: ...


@runtime_checkable
class KVProjection(Protocol):
    """The K/V a layer writes, computed without running its attention or MLP."""

    def kv_projection(
        self, index: int, layer: DecoderLayer, hidden: torch.Tensor, ctx: PassContext
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def fill_kv(self, index: int, layer: DecoderLayer, hidden: torch.Tensor, ctx: PassContext) -> None: ...


@runtime_checkable
class LogitHead(Protocol):
    """Last-layer hidden states to logits."""

    def logits(self, hidden: torch.Tensor) -> torch.Tensor: ...

    def transform_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """The family's per-logit transform of the LM head's output."""
        ...


@runtime_checkable
class WeightLayout(Protocol):
    """The weights of one decoder layer, as weight rewrites (LoRA, quantization, pruning) address them."""

    def mixer(self, layer: nn.Module) -> nn.Module | None: ...

    def attention_module(self, index: int) -> nn.Module | None: ...

    def linear_modules(self, layer: nn.Module) -> dict[str, nn.Linear]: ...

    def lora_targets(self, layer: nn.Module) -> dict[str, nn.Module]: ...

    def expert_parameters(self, layer: nn.Module) -> dict[str, nn.Parameter]:
        """The fused weights of every MoE expert block of ``layer`` (``{}`` for a dense layer)."""
        ...


@runtime_checkable
class DepthEditing(Protocol):
    """Removing decoder layers in place, config included."""

    def remove_layers(self, indices: Iterable[int]) -> None: ...

    def renumber_config(self, kept: Sequence[int]) -> None: ...


if TYPE_CHECKING:  # mypy checks that the full adapter provides every facet
    from .base import ArchAdapter

    def _implements_every_facet(adapter: ArchAdapter) -> tuple[
        DecoderModules, ModelSizes, DevicePlacement, AttentionLayout, RotaryPositions, LayerExecution, KVProjection,
        LogitHead, WeightLayout, DepthEditing,
    ]:
        return adapter, adapter, adapter, adapter, adapter, adapter, adapter, adapter, adapter, adapter
