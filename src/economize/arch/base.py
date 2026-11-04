"""Architecture adapters: the single place where economize touches a model's internal module layout.

Techniques and the decoding runtime never reach into ``model.model.layers`` themselves. They ask an
:class:`ArchAdapter`, so supporting a new model family means registering one adapter class, usually a small
subclass of :class:`~economize.arch.StandardDecoderAdapter` that overrides what differs::

    from economize.arch import StandardDecoderAdapter, register_adapter

    @register_adapter("my_model_type")
    class MyAdapter(StandardDecoderAdapter):
        def transform_logits(self, logits):
            return super().transform_logits(logits) * 0.5

Multimodal wrappers (``Gemma4ForConditionalGeneration`` and friends) are handled by the same adapters: the adapter
works on the text decoder (``model.model.language_model``) and the text config, and leaves the vision and audio
towers alone. :func:`adapter_for` looks up the wrapper's ``model_type`` first and then that of its text config.
"""

from __future__ import annotations

import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, TypeVar, cast

import torch
from torch import nn

from ..errors import ConfigError, IncompatibleTechniquesError, UnsupportedModelError
from ..registry import Registry
from ..unwrap import peft_base
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
from .protocols import DecoderLayer, LayerInputs
from .spec import LAYER_TYPE_KINDS, AttentionSpec, HeadGeometry, PassContext, StateKind, layer_kind_of

if TYPE_CHECKING:
    from ..attention.banded import StructuredMask

__all__ = [
    "ARCHITECTURES",
    "PER_LAYER_CONFIG_FIELDS",
    "SUPPORTED_ATTENTION",
    "ArchAdapter",
    "adapter_for",
    "config_value",
    "kept_below",
    "module_device",
    "register_adapter",
    "renumbered",
    "text_config",
]

logger = logging.getLogger(__name__)

#: Attention implementations whose layers accept an explicit mask over the whole cache buffer (a dense tensor, or a
#: flex-attention ``BlockMask``). The runtime registers one mask format per implementation in
#: :data:`economize.runtime.MASK_FORMATS` and refuses adapters whose implementation has none.
SUPPORTED_ATTENTION = ("sdpa", "eager", "flex_attention")

#: Module names under which decoder layers keep their token mixer (LoRA targets its linear children).
_MIXERS = ("self_attn", "linear_attn", "attention", "mixer")


def module_device(module: nn.Module, default: torch.device | None = None) -> torch.device:
    """Where ``module`` computes: the execution device of an ``accelerate`` hook, else its first parameter's device."""
    hook = getattr(module, "_hf_hook", None)
    device = getattr(hook, "execution_device", None)
    if device is not None:
        return torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    for tensor in module.parameters():
        return tensor.device
    for buffer in module.buffers():
        return buffer.device
    if default is None:
        raise UnsupportedModelError(f"cannot tell the device of {type(module).__name__}: it has no parameters")
    return default


class _KVCaptured(Exception):
    """Raised by :class:`_CaptureCache` to stop a layer right after it produced its K/V."""


class _CaptureCache:
    """Stands in for the KV cache during :meth:`ArchAdapter.kv_projection`: records K/V and aborts the layer."""

    def __init__(self) -> None:
        self.key: torch.Tensor | None = None
        self.value: torch.Tensor | None = None
        self.layer_idx: int | None = None

    def update(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int, *args: object, **kwargs: object) -> tuple:
        self.key, self.value, self.layer_idx = key, value, layer_idx
        raise _KVCaptured


class ArchAdapter(ABC):
    """Uniform view of a decoder-only causal LM.

    Layers are 1-indexed, and index 0 denotes the token embedding, so ``hidden[l]`` is the output of layer ``l``.
    The adapter implements every facet protocol of :mod:`economize.arch.protocols`, one section each below.
    Subclasses provide the modules (the four abstract properties of :class:`~economize.arch.protocols.DecoderModules`);
    every other member has a default that follows the Hugging Face conventions shared by most families and is
    overridden only in the facets where a family differs. Code that needs one aspect of a model should accept the
    facet protocol instead of the whole adapter.

    Attributes:
        model: The wrapped Hugging Face model (a ``*ForCausalLM`` or a multimodal ``*ForConditionalGeneration``).
        config: The text decoder's config (``model.config`` for a plain causal LM).
        attention_implementations: Attention implementations the family's layers support with an explicit mask, by
            name or by the family a registered kernel reports.
    """

    attention_implementations: ClassVar[tuple[str, ...]] = SUPPORTED_ATTENTION
    per_layer_config_fields: ClassVar[tuple[str, ...]] = PER_LAYER_CONFIG_FIELDS
    #: Config fields holding a layer index ``k`` that splits the layers into ``i < k`` and ``i >= k`` (0-indexed),
    #: such as DeepSeek-V3's ``first_k_dense_replace``. :meth:`renumber_config` sets them to the number of kept
    #: layers below ``k``.
    layer_threshold_config_fields: ClassVar[tuple[str, ...]] = ()
    #: Config fields holding a list of 0-indexed layer indices, such as Qwen-MoE's ``mlp_only_layers``.
    #: :meth:`renumber_config` maps them to the new indices and drops removed layers.
    layer_index_config_fields: ClassVar[tuple[str, ...]] = ()
    #: Name of the decoder-layer module layout that weight-rewrite methods key their tables on (``"llama"``: the
    #: ``self_attn.{q,k,v,o}_proj`` and ``mlp.{gate,up,down}_proj`` linears behind ``input_layernorm`` and
    #: ``post_attention_layernorm``, as AWQ's scale groups expect; ``"gemma4"``: the same names with sandwich norms),
    #: or ``None`` when the family has no shared layout.
    layer_layout: ClassVar[str | None] = None

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.config = text_config(model.config)
        self.validate()

    # ---- DecoderModules: the modules (every adapter provides the four abstract ones) ---------------------------------
    @property
    @abstractmethod
    def layers(self) -> nn.ModuleList:
        """The decoder layers, 0-indexed as stored in the model."""

    @property
    @abstractmethod
    def embed_tokens(self) -> nn.Module:
        """The token embedding module."""

    @property
    @abstractmethod
    def final_norm(self) -> nn.Module:
        """The norm applied after the last decoder layer."""

    @property
    @abstractmethod
    def lm_head(self) -> nn.Module:
        """The output projection to the vocabulary."""

    def layer(self, index: int) -> nn.Module:
        """Decoder layer ``index`` (1-indexed)."""
        if not 1 <= index <= self.num_layers:
            raise IndexError(f"layer index {index} outside 1..{self.num_layers}")
        return self.layers[index - 1]

    def module_path(self, module: nn.Module) -> str:
        """The dotted name of ``module`` inside :attr:`model` (``"model.layers.3"``, ``"lm_head"``), as a state dict
        or a serving engine names it. Raise :class:`UnsupportedModelError` if ``module`` is not part of the model."""
        for name, candidate in self.model.named_modules(remove_duplicate=False):
            if candidate is module:
                return name
        raise UnsupportedModelError(f"{type(module).__name__} is not a submodule of {type(self.model).__name__}")

    @property
    def model_type(self) -> str:
        """``model_type`` of the wrapped model (of the wrapper for multimodal models)."""
        return str(getattr(self.model.config, "model_type", None) or type(self.model).__name__)

    # ---- ModelSizes --------------------------------------------------------------------------------------------------
    @property
    def num_layers(self) -> int:
        """Number of decoder layers ``L``."""
        return len(self.layers)

    @property
    def hidden_size(self) -> int:
        """Width of the residual stream."""
        return int(self.config.hidden_size)

    @property
    def vocab_size(self) -> int:
        """Number of rows of the LM head."""
        return int(self.config.vocab_size)

    @property
    def num_kv_heads(self) -> int:
        """Key/value heads of a standard attention layer."""
        return _geometry(self.config, self.hidden_size).kv_heads

    @property
    def head_dim(self) -> int:
        """Head width of a standard attention layer."""
        return _geometry(self.config, self.hidden_size).head_dim

    @property
    def dtype(self) -> torch.dtype:
        """Compute dtype of the model (that of the token embedding)."""
        return cast(torch.Tensor, self.embed_tokens.weight).dtype

    def layer_config(self, index: int) -> HFConfig:
        """The config of layer ``index`` (1-indexed): its entry of ``per_layer_config`` on a heterogeneous config
        (transformers >= 5.17), the model's text config otherwise."""
        if is_heterogeneous(self.config):
            return self.config.per_layer_config[index - 1]
        return self.config

    def head_geometry(self, index: int) -> HeadGeometry:
        """Head counts and head width of attention layer ``index`` (1-indexed); the standard sizes unless the
        layer's own config (heterogeneous configs) or the family gives some layer types other ones."""
        return _geometry(self.layer_config(index), self.hidden_size)

    @classmethod
    def shape_fields(cls, config: HFConfig, fields: Mapping[str, object]) -> Mapping[str, object]:
        """Corrections to the generic reading ``fields`` of :meth:`economize.bench.ModelShape.from_config` for this
        family's text ``config`` (for example ``{"gated": False}``). None by default."""
        return {}

    # ---- DevicePlacement ---------------------------------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        """Device of the token embedding, where every pass starts."""
        return module_device(self.embed_tokens)

    @property
    def head_device(self) -> torch.device:
        """Device of the final norm and LM head."""
        return module_device(self.lm_head, self.device)

    @property
    def rotary_device(self) -> torch.device:
        """Device of the rotary embedding, where its tables are computed (differs from a layer's under
        ``device_map``); the embedding's device without a rotary embedding."""
        rotary = self.rotary_emb
        return self.device if rotary is None else module_device(rotary, self.device)

    def layer_device(self, index: int) -> torch.device:
        """Device of layer ``index`` (differs between layers when the model is split with ``device_map``)."""
        return module_device(self.layer(index), self.device)

    @property
    def tensor_parallel_size(self) -> int:
        """Processes the model is split over by transformers' tensor parallelism (``tp_plan``); 1 when unsplit.

        A split model holds ``1 / size`` of the attention heads and of the MLP width of every layer in each process.
        :attr:`num_kv_heads` stays the model's total; the caches of the layer runner take their shapes from the
        key/value states they store and therefore hold the local heads.
        """
        size = getattr(self.model, "_tp_size", None)
        return size if isinstance(size, int) and size > 1 else 1

    # ---- AttentionLayout ---------------------------------------------------------------------------------------------
    @property
    def attn_implementation(self) -> str:
        """The attention implementation the decoder was loaded with, as its family.

        A kernel registered in :data:`economize.attention.ATTENTION_KERNELS` reports the built-in implementation
        whose masks and semantics it shares (``"economize"`` reports ``"sdpa"``).
        """
        from ..attention.registry import attention_family

        return attention_family(str(getattr(self.config, "_attn_implementation", None) or "eager"))

    @property
    def takes_structured_masks(self) -> bool:
        """Whether the decoder's attention function takes a :class:`~economize.attention.StructuredMask` in place
        of a mask tensor (see :func:`~economize.attention.accepts_structured_masks`)."""
        from ..attention.banded import accepts_structured_masks

        return accepts_structured_masks(str(getattr(self.config, "_attn_implementation", None) or "eager"))

    @property
    def layer_types(self) -> list[str]:
        """``config.layer_types``, or the Mistral convention (sliding everywhere if ``sliding_window`` is set)."""
        types = getattr(self.config, "layer_types", None)
        if types:
            return list(types)[: self.num_layers]
        sliding = getattr(self.config, "sliding_window", None)
        return ["sliding_attention" if sliding else "full_attention"] * self.num_layers

    def attention_spec(self, index: int) -> AttentionSpec:
        """Mask kind, window and rotary key of layer ``index`` (1-indexed)."""
        layer_type = self.layer_types[index - 1]
        kind = layer_kind_of(layer_type)
        if kind is None:
            raise UnsupportedModelError(
                f"layer {index} has type {layer_type!r}; economize supports {sorted(LAYER_TYPE_KINDS)} layers "
                "(register others with economize.arch.register_layer_kind)"
            )
        window = None if kind.window_field is None else int(config_value(self.config, kind.window_field))
        return AttentionSpec(kind.name, window, self.rope_key(layer_type))

    def state_kind(self, index: int) -> StateKind:
        """Whether layer ``index`` keeps per-position K/V or a recurrent state."""
        return self.attention_spec(index).state_kind

    @property
    def cache_kind(self) -> str:
        """Key of :data:`economize.runtime.CACHES` this model needs (``"hybrid"`` when some layer is recurrent)."""
        recurrent = any(self.state_kind(i) is StateKind.RECURRENT for i in range(1, self.num_layers + 1))
        return "hybrid" if recurrent else "positional"

    def kv_source(self, index: int) -> int:
        """The layer whose cache entries layer ``index`` attends over (itself, except for KV-sharing layers)."""
        return index

    # ---- RotaryPositions ---------------------------------------------------------------------------------------------
    @property
    def rotary_emb(self) -> nn.Module | None:
        """The model's rotary embedding module; ``None`` (the default) for a family without one, whose layers then
        receive ``position_embeddings=None``."""
        return None

    def rope_key(self, layer_type: str) -> str:
        """Rotary table of layers of type ``layer_type`` (``"default"``: one table for every layer)."""
        return "default"

    def position_embeddings(self, like: torch.Tensor, positions: torch.Tensor, rope: str = "default") -> object:
        """Rotary table (usually ``(cos, sin)``) for ``positions`` ``[B, S]`` in the dtype and on the device of
        ``like``; ``None`` without a rotary embedding."""
        rotary = self.rotary_emb
        return None if rotary is None else rotary(like, positions.to(like.device))

    # ---- LayerExecution ----------------------------------------------------------------------------------------------
    def embed(self, input_ids: torch.Tensor, ctx: PassContext | None = None) -> torch.Tensor:
        """Input of the first decoder layer for ``input_ids`` ``[..., S]``.

        Families that feed extra per-token inputs to every layer store them in ``ctx.token_inputs``.
        """
        return self.embed_tokens(input_ids.to(self.device))

    @property
    def uses_per_layer_inputs(self) -> bool:
        """Whether every layer receives a per-token input of its own besides the hidden state (Gemma-4's per-layer
        embeddings), so that layers of one type cannot share their call arguments."""
        return False

    def layer_inputs(
        self,
        index: int,
        hidden: torch.Tensor,
        ctx: PassContext,
        cache: object,
        mask: torch.Tensor | StructuredMask | None,
    ) -> LayerInputs:
        """Positional arguments after ``hidden`` and keyword arguments for calling decoder layer ``index``.

        Args:
            cache: What the layer receives as ``past_key_values``.
            mask: The attention mask (``None`` when the layer only projects K/V).
        """
        spec = self.attention_spec(index)
        return (), {
            "attention_mask": mask,
            "position_ids": ctx.position_ids(hidden.device),
            "past_key_values": cache,
            "use_cache": True,
            "position_embeddings": ctx.rotary(spec.rope, hidden),
        }

    def call_layer(self, index: int, layer: DecoderLayer, hidden: torch.Tensor, ctx: PassContext) -> torch.Tensor:
        """Run ``layer`` (decoder layer ``index`` or a stand-in callable) on ``hidden`` ``[B, S, d]`` within ``ctx``.

        The layer writes its K/V through ``ctx.cache`` at ``ctx.slots``.
        """
        mask = ctx.mask(self.attention_spec(index), hidden.device)
        args, kwargs = self.layer_inputs(index, hidden, ctx, ctx.cache, mask)
        out = layer(hidden, *args, **kwargs)
        return out[0] if isinstance(out, tuple) else out

    def layer_call_key(self, index: int) -> Hashable:
        """Key under which layer-wise calibration shares the call arguments of layer ``index`` (1-indexed).

        Layers with equal keys receive the same arguments besides the hidden state (mask, rotary embeddings), so
        :func:`economize.core.calibration.capture_inputs` records them once per key. The default is the layer type
        (:attr:`layer_types`); layers that also take an input of their own (:attr:`uses_per_layer_inputs`) get a key
        each, ``(type, index)``.
        """
        layer_type = self.layer_types[index - 1]
        return (layer_type, index) if self.uses_per_layer_inputs else layer_type

    # ---- KVProjection ------------------------------------------------------------------------------------------------
    def kv_projection(
        self, index: int, layer: DecoderLayer, hidden: torch.Tensor, ctx: PassContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The K/V ``layer`` would write for ``hidden`` ``[B, S, d]``, without running its attention or MLP.

        The default calls the layer with a stand-in cache that records K/V and aborts the layer, which reproduces
        the family's norms, projections and rotary embedding exactly at the cost of the query projection.
        """
        _, key, value = self._capture_kv(index, layer, hidden, ctx)
        return key, value

    def fill_kv(self, index: int, layer: DecoderLayer, hidden: torch.Tensor, ctx: PassContext) -> None:
        """Write the K/V layer ``index`` would produce for ``hidden`` at ``ctx.slots``, without running the layer.

        Layers that attend over another layer's cache entries, and token-wise layers, write nothing.

        Raises:
            IncompatibleTechniquesError: Layer ``index`` keeps a recurrent state, which cannot be filled.
        """
        kind = self.state_kind(index)
        if kind is StateKind.RECURRENT:
            raise IncompatibleTechniquesError(f"layer {index} keeps a recurrent state; it cannot be skipped")
        if kind is StateKind.NONE or self.kv_source(index) != index:
            return
        ctx.cache.write(*self._capture_kv(index, layer, hidden, ctx))

    def _capture_kv(
        self, index: int, layer: DecoderLayer, hidden: torch.Tensor, ctx: PassContext
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """``(cache layer index, K, V)`` that ``layer`` writes for ``hidden``, its attention and MLP not run."""
        capture = _CaptureCache()
        args, kwargs = self.layer_inputs(index, hidden, ctx, capture, None)
        with contextlib.suppress(_KVCaptured):
            layer(hidden, *args, **kwargs)
        if capture.key is None or capture.value is None or capture.layer_idx is None:
            raise UnsupportedModelError(f"layer {index} did not write K/V through past_key_values.update")
        return capture.layer_idx, capture.key, capture.value

    # ---- LogitHead ---------------------------------------------------------------------------------------------------
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Final norm, LM head and :meth:`transform_logits` applied to last-layer hidden states."""
        return self.transform_logits(self.lm_head(self.final_norm(hidden.to(self.head_device))))

    def transform_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """The family's transform of the LM head's output (identity; Gemma's softcap and Granite's
        ``logits_scaling`` override it).

        It must act on every logit on its own, so that a head restricted to some token ids (a draft vocabulary,
        :func:`~economize.techniques._drafting.adapter_head_slice`) can apply it to its columns only. A family whose
        transform mixes the logits of different tokens overrides :meth:`logits` instead.
        """
        return logits

    # ---- WeightLayout ------------------------------------------------------------------------------------------------
    def mixer(self, layer: nn.Module) -> nn.Module | None:
        """The token mixer of one decoder layer (its attention or recurrent block), found under the Hugging Face
        names in ``_MIXERS``; ``None`` when the layer has none. Families with another layout override this."""
        return next((m for name in _MIXERS if (m := getattr(layer, name, None)) is not None), None)

    def attention_module(self, index: int) -> nn.Module | None:
        """The softmax-attention module of layer ``index``: the module that receives the attention mask and, under
        eager attention, returns the attention weights. ``None`` for a recurrent layer or a layer without one."""
        if self.state_kind(index) is StateKind.RECURRENT:
            return None
        return self.mixer(self.layer(index))

    def linear_modules(self, layer: nn.Module) -> dict[str, nn.Linear]:
        """Every ``nn.Linear`` of one decoder layer, keyed by its dotted path inside the layer.

        Fused MoE expert weights are 3-D parameters, not ``nn.Linear`` modules; see :meth:`expert_parameters`.
        """
        return {name: module for name, module in layer.named_modules() if isinstance(module, nn.Linear)}

    def lora_targets(self, layer: nn.Module) -> dict[str, nn.Module]:
        """The linear maps LoRA may adapt, keyed by attribute name: the token mixer's projections and the dense MLP
        projections (``*_proj`` children of ``mlp``; routers, experts and shared-expert gates excluded).

        A map is an ``nn.Linear`` or a module standing in for one (``in_features`` and ``out_features`` set, such as
        a low-rank factorization or a PEFT LoRA layer). An economize LoRA on a PEFT LoRA layer adds its update to the
        layer's output, which already holds PEFT's own update (merged or not).
        """
        targets: dict[str, nn.Module] = {}
        mixer = self.mixer(layer)
        if mixer is not None:
            targets.update({n: m for n, m in mixer.named_children() if _is_linear(m)})
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            targets.update({n: m for n, m in mlp.named_children() if _is_linear(m) and n.endswith("_proj")})
        return targets

    def expert_parameters(self, layer: nn.Module) -> dict[str, nn.Parameter]:
        """The fused weights of every MoE expert block of one layer (``experts.gate_up_proj`` etc.), else ``{}``."""
        found: dict[str, nn.Parameter] = {}
        for name, module in layer.named_modules():
            if name.rsplit(".", 1)[-1] == "experts":
                found.update({f"{name}.{key}": p for key, p in module.named_parameters(recurse=False)})
        return found

    # ---- DepthEditing ------------------------------------------------------------------------------------------------
    def remove_layers(self, indices: Iterable[int]) -> None:
        """Delete decoder layers ``indices`` (1-indexed) from the model, in place.

        The remaining layers are renumbered (their ``layer_idx`` attributes), ``config.num_hidden_layers`` is
        updated and :meth:`renumber_config` rewrites every config field that addresses layers by index, so the
        shallower model also survives ``save_pretrained`` and ``from_pretrained``. Families whose config derives the
        layer count from a per-layer field (Nemotron-H) need nothing more.

        Raises:
            ConfigError: An index lies outside ``1..L``, or every layer would be removed.
            UnsupportedModelError: Some layer attends over another layer's cache entries (KV sharing), which a
                removal would break.
        """
        drop = sorted(set(indices))
        count = self.num_layers
        if not drop:
            return
        if not all(1 <= i <= count for i in drop) or len(drop) >= count:
            raise ConfigError(f"cannot remove layers {drop} from a model with {count} layers")
        if any(self.kv_source(i) != i for i in range(1, count + 1)):
            raise UnsupportedModelError(f"{self.model_type!r} shares K/V between layers; its layers cannot be removed")
        self.renumber_config([i for i in range(count) if i + 1 not in drop])
        layers = self.layers
        for index in reversed(drop):
            del layers[index - 1]
        for new_index, layer in enumerate(layers):
            for module in layer.modules():
                if isinstance(getattr(module, "layer_idx", None), int):
                    module.layer_idx = new_index  # type: ignore[assignment]  # Module.__setattr__ types tensors/modules

    def renumber_config(self, kept: Sequence[int]) -> None:
        """Rewrite the config fields that address decoder layers by index, for a removal that keeps ``kept``.

        :meth:`remove_layers` calls it before deleting any layer (``self.num_layers`` is still the old count). The
        default sets ``num_hidden_layers`` (unless the config derives it from a per-layer field), filters the
        per-layer lists (:attr:`per_layer_config_fields`), recounts the split indices
        (:attr:`layer_threshold_config_fields`) and renumbers the index lists (:attr:`layer_index_config_fields`).
        Families whose fields need more (a periodic pattern such as Qwen-MoE's ``decoder_sparse_step``) override it and
        call ``super()``. The per-layer overrides of a heterogeneous config follow their layers.

        Args:
            kept: 0-indexed old indices of the layers that stay, ascending.
        """
        config, count = self.config, self.num_layers
        overrides = layer_overrides(config)
        with contextlib.suppress(AttributeError):  # a read-only property derived from a per-layer field (Nemotron-H)
            config.num_hidden_layers = len(kept)
        for name in self.per_layer_config_fields:
            values = getattr(config, name, None)
            if isinstance(values, (list, tuple)) and len(values) == count:
                setattr(config, name, type(values)(values[i] for i in kept))
        for name in self.layer_threshold_config_fields:
            value = getattr(config, name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                setattr(config, name, kept_below(kept, value))
        for name in self.layer_index_config_fields:
            values = getattr(config, name, None)
            if isinstance(values, (list, tuple)):
                setattr(config, name, type(values)(renumbered(values, kept)))
        if overrides:
            # Re-keyed only now: transformers validates the indices against num_hidden_layers, and a layer count
            # that no longer matches makes it demand a global value for every overridden attribute.
            position = {old: new for new, old in enumerate(kept)}
            set_layer_overrides(config, {position[old]: values for old, values in overrides.items() if old in position})

    # ---- checks ------------------------------------------------------------------------------------------------------
    def validate(self) -> None:
        """Raise :class:`UnsupportedModelError` if the loaded model cannot be run by economize."""
        raw = str(getattr(self.config, "_attn_implementation", None) or "eager")
        if raw not in self.attention_implementations and self.attn_implementation not in self.attention_implementations:
            raise UnsupportedModelError(
                f"attention implementation {self.attn_implementation!r} is not supported for {self.model_type!r}; "
                f"load the model with attn_implementation in {self.attention_implementations}"
            )
        for index in range(1, self.num_layers + 1):
            self.attention_spec(index)
        if raw == "sdpa":  # same results, with contiguous K/V where PyTorch 2.13's masked kernel errs (sdpa_guard)
            from ..attention import sdpa_guard

            sdpa_guard.install()


def _is_linear(module: nn.Module) -> bool:
    """An ``nn.Linear`` or a stand-in that declares its feature sizes."""
    if isinstance(module, nn.Linear):
        return True
    return all(isinstance(getattr(module, name, None), int) for name in ("in_features", "out_features"))


def _geometry(config: HFConfig, hidden_size: int) -> HeadGeometry:
    """Head counts and head width ``config`` gives (the model's config or one layer's): ``num_key_value_heads``
    defaults to the query heads, ``head_dim`` to ``hidden_size / heads``."""
    heads = int(config_value(config, "num_attention_heads"))
    kv_heads = config_value(config, "num_key_value_heads") or heads
    head_dim = config_value(config, "head_dim") or hidden_size // heads
    return HeadGeometry(heads, int(kv_heads), int(head_dim))


#: Adapter classes by ``model_type``. Built as ``cls(model)``.
ARCHITECTURES: Registry[type[ArchAdapter]] = Registry("architecture", "economize.architectures")

_A = TypeVar("_A", bound=type[ArchAdapter])


def register_adapter(*model_types: str) -> Callable[[_A], _A]:
    """Class decorator registering an adapter for one or more ``model_type`` values.

    A multimodal wrapper needs no entry of its own when its text config's ``model_type`` is registered.
    """

    def add(cls: _A) -> _A:
        for model_type in model_types:
            ARCHITECTURES.register(model_type, cls)
        return cls

    return add


def adapter_for(model: nn.Module) -> ArchAdapter:
    """Build the adapter registered for the model's ``model_type`` (the wrapper's, else its text config's).

    A PEFT wrapper (``PeftModel``) is unwrapped: the adapter works on its base model, with the LoRA layers in place.

    Unknown model types fall back to the standard Llama-like layout when the model has it, with a warning.

    Raises:
        UnsupportedModelError: No adapter fits, or the model's layers or attention implementation are unsupported.
    """
    model = peft_base(model)
    config = getattr(model, "config", None)
    candidates = [getattr(config, "model_type", None), getattr(text_config(config), "model_type", None)]
    for model_type in candidates:
        if isinstance(model_type, str) and model_type in ARCHITECTURES:
            return ARCHITECTURES.get(model_type)(model)
    from .standard import StandardDecoderAdapter

    if StandardDecoderAdapter.matches(model):
        logger.warning(
            "no economize adapter is registered for model_type %r; using the standard decoder layout. Register an "
            "adapter in economize.arch.ARCHITECTURES once its outputs are verified against the model's forward.",
            candidates[0],
        )
        return StandardDecoderAdapter(model)
    raise UnsupportedModelError(
        f"no economize adapter for model_type {candidates[0]!r}; registered: {ARCHITECTURES.names()}"
    )
