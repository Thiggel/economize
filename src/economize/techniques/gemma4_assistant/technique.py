"""The Gemma-4 assistant technique: loads the assistant and checks it against the base model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import AttachContext, Capability, Technique, register_technique
from ...errors import ConfigError, UnsupportedModelError
from .config import Gemma4AssistantDraftConfig
from .drafter import Gemma4AssistantDrafter

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...runtime import LayerRunner

__all__ = ["Gemma4AssistantTechnique", "context_layers", "load_assistant"]


def _geometry(attention: Any) -> tuple[Any, ...]:
    """What fixes the shape of the cache entries an HF Gemma-4 attention reads: head width, query heads, query heads
    per key head and whether keys double as values (read from the module, as layers of one type may differ)."""
    return (attention.head_dim, attention.q_proj.out_features // attention.head_dim, attention.num_key_value_groups,
            bool(getattr(attention, "use_alternative_attention", False)))


def load_assistant(config: Gemma4AssistantDraftConfig, device: torch.device, dtype: torch.dtype) -> nn.Module:
    """Load the assistant of ``config`` (eval mode, frozen) on ``device``."""
    from transformers import AutoModelForCausalLM

    resolved = dtype if config.dtype == "auto" else getattr(torch, config.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.model, revision=config.revision, dtype=resolved, attn_implementation=config.attn_implementation
    )
    if getattr(model.config, "model_type", None) != "gemma4_assistant":
        raise ConfigError(f"{config.model!r} is a {model.config.model_type!r} model, not a Gemma-4 assistant")
    # transformers types PreTrainedModel.to through a decorator that mypy cannot see through
    model.to(device).eval().requires_grad_(False)  # type: ignore[arg-type]
    return model


def context_layers(adapter: ArchAdapter) -> dict[str, int]:
    """The base layer (1-indexed) whose cache entries the assistant reads, per attention type.

    That is the last layer of each type that stores its keys and values (``store_full_length_kv``, which
    ``transformers`` sets on the layers whose entries later layers share); the entries are what the base model's
    ``return_shared_kv_states=True`` returns.

    Raises:
        UnsupportedModelError: The model is no Gemma-4 decoder.
    """
    layers: dict[str, int] = {}
    for index in range(1, adapter.num_layers + 1):
        attention: Any = getattr(adapter.layer(index), "self_attn", None)
        if attention is None or not hasattr(attention, "store_full_length_kv"):
            raise UnsupportedModelError(
                f"the Gemma-4 assistant drafts for Gemma-4 decoders only; layer {index} of this "
                f"{adapter.model_type!r} model has no Gemma-4 attention"
            )
        if attention.store_full_length_kv:
            layers[attention.layer_type] = index
    return layers


@register_technique
class Gemma4AssistantTechnique(Technique):
    """Speculative decoding with the Gemma-4 assistant (see :class:`Gemma4AssistantDraftConfig`).

    The assistant is loaded when the technique attaches and is kept outside the module tree of the technique, so
    it is neither saved with the economize checkpoint nor moved or trained with the techniques.
    """

    name = Gemma4AssistantDraftConfig.technique
    config_class = Gemma4AssistantDraftConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True)
    capabilities = Capability.DRAFTER
    config: Gemma4AssistantDraftConfig

    def __init__(self, config: Gemma4AssistantDraftConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)
        self._assistant: list[nn.Module] = []  # a list keeps the assistant out of the module tree
        self.context: dict[str, int] = {}

    @property
    def assistant(self) -> nn.Module:
        if not self._assistant:
            raise RuntimeError("the assistant is loaded when the technique attaches")
        return self._assistant[0]

    def attach(self, model: nn.Module, adapter: ArchAdapter, context: AttachContext | None = None) -> None:
        """Load the assistant and check that it reads the base model's cache entries and states.

        Raises:
            UnsupportedModelError: The model is no Gemma-4 decoder.
            ConfigError: The assistant belongs to another base model (state width, head geometry or layer types).
        """
        if self._assistant:
            return
        layers = context_layers(adapter)
        assistant = load_assistant(self.config, adapter.device, adapter.dtype)
        self.context = check_assistant(assistant, adapter, layers)
        self._assistant.append(assistant)

    def detach(self) -> None:
        self._assistant.clear()
        super().detach()

    def drafter(self, runner: LayerRunner) -> Gemma4AssistantDrafter:
        return Gemma4AssistantDrafter(self.assistant, self.adapter, self.context)

    def extra_repr(self) -> str:
        return f"model={self.config.model!r}"


def check_assistant(assistant: nn.Module, adapter: ArchAdapter, layers: dict[str, int]) -> dict[str, int]:
    """The context layer of every attention type the assistant uses, after checking that the assistant fits the base
    model (raises :class:`~economize.ConfigError` otherwise)."""
    config: Any = assistant.config
    if int(config.backbone_hidden_size) != adapter.hidden_size:
        raise ConfigError(f"the assistant reads states of width {config.backbone_hidden_size}, the model's are "
                          f"{adapter.hidden_size} wide: it belongs to another Gemma-4 model")
    context: dict[str, int] = {}
    for index, layer in enumerate(cast(Any, assistant).model.layers, 1):
        attention = layer.self_attn
        kind = attention.layer_type
        if not attention.is_kv_shared_layer:
            raise ConfigError(f"assistant layer {index} projects keys and values of its own; every assistant layer "
                              "must read the model's (num_kv_shared_layers = num_hidden_layers)")
        if kind not in layers:
            raise ConfigError(f"assistant layer {index} is a {kind} layer; the model stores no entries of that type")
        source = adapter.layer(layers[kind]).self_attn
        if _geometry(attention) != _geometry(source):
            raise ConfigError(f"assistant layer {index} ({kind}) has the attention geometry {_geometry(attention)}, "
                              f"the model's entries of layer {layers[kind]} {_geometry(source)} (head width, query "
                              "heads, query heads per key head, keys as values)")
        context[kind] = layers[kind]
    return dict(sorted(context.items()))
