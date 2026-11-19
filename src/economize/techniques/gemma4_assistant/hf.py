"""What the technique reads of the ``transformers`` Gemma-4 modules (``Gemma4AssistantForCausalLM`` and the Gemma-4
attention), typed by use: the classes exist only in recent ``transformers`` versions and are imported nowhere."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from ...arch import HFConfig

__all__ = ["AssistantModel", "Gemma4Attention"]


class _Linear(Protocol):
    """A projection: its weight, and the call that applies it."""

    weight: torch.Tensor

    def __call__(self, x: torch.Tensor, /) -> torch.Tensor: ...


class _QueryProjection(Protocol):
    out_features: int


class Gemma4Attention(Protocol):
    """The fields of a Gemma-4 attention module that fix where and how it reads the cache."""

    head_dim: int
    q_proj: _QueryProjection
    num_key_value_groups: int
    layer_type: str
    is_kv_shared_layer: bool


class _Layer(Protocol):
    self_attn: Gemma4Attention


class _Output(Protocol):
    last_hidden_state: torch.Tensor


class _Backbone(Protocol):
    """The assistant's decoder: its layers, and the call that runs them over the base model's entries."""

    layers: Sequence[_Layer]
    __call__: Callable[..., _Output]


class AssistantModel(Protocol):
    """A loaded ``Gemma4AssistantForCausalLM``."""

    config: HFConfig
    model: _Backbone
    pre_projection: _Linear
    post_projection: _Linear
    lm_head: _Linear

    def masked_embedding(self, hidden: torch.Tensor, weight: torch.Tensor, /) -> torch.Tensor: ...
