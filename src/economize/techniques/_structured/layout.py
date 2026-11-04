"""Where the norms and linear maps of a decoder layer live, for the techniques that change the model's widths.

SliceGPT rotates the residual stream and needs every map that reads or writes it; width pruning (FLAP, LLM-Pruner,
Minitron) removes attention heads and FFN channels and needs the maps on both sides of each. Both read the module
names from a :class:`DecoderLayout` registered in :data:`DECODER_LAYOUTS` under an adapter's
:attr:`~economize.arch.ArchAdapter.layer_layout` (``"llama"`` for Llama, Mistral, Qwen2 and Qwen3, ``"gemma4"`` for
Gemma 4, see :mod:`.gemma4`) or its ``config.model_type``. A new family registers its layout (a subclass when its
config describes per-layer widths differently or some of its variants must be refused); neither this module nor a
technique changes.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from torch import nn

from ...errors import UnsupportedModelError
from ...registry import Registry

__all__ = ["DECODER_LAYOUTS", "DecoderLayout", "LayerShape", "layout_for"]


@dataclass(frozen=True)
class LayerShape:
    """The prunable units of one decoder layer.

    Attributes:
        heads: Query heads.
        kv_heads: Key/value heads, one per group (a group is a key and value head and the query heads sharing them).
        head_dim: Width of a head.
        intermediate: FFN channels.
    """

    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int

    @property
    def group_size(self) -> int:
        """Query heads per key/value group."""
        return self.heads // self.kv_heads

    @property
    def group_width(self) -> int:
        """Input channels of the attention output map per group."""
        return self.group_size * self.head_dim


@dataclass(frozen=True)
class DecoderLayout:
    """Module names (relative to a decoder layer) of a pre-norm block ``x + attn(norm1(x))``, ``h + mlp(norm2(h))``.

    The norms must compute ``weight * x / rms(x)`` (the Llama RMSNorm, no ``1 + weight`` offset), and the MLP must
    be gated: ``down(act(gate(x)) * up(x))``.

    Attributes:
        attn_norm: The norm in front of the attention block.
        attention: The attention module (holds ``num_key_value_groups`` and ``head_dim``).
        attn_inputs: The query, key and value maps, in that order.
        attn_output: The attention output map.
        mlp_norm: The norm in front of the MLP.
        mlp: The MLP module.
        mlp_inputs: The gate and up maps, in that order.
        mlp_output: The down map.
        optional_maps: Maps that some layers lack (Gemma 4's value map on the layers that read their values from the
            key map). :meth:`maps` skips them where absent; every other map must exist.
        residual_refusal: Why the residual stream of this layout cannot be rotated or cut (``None``: it can). The
            techniques that do either (SliceGPT, Minitron's residual pruning) ask for the layout with
            ``layout_for(..., residual=True)``, which raises with this reason.
    """

    attn_norm: str = "input_layernorm"
    attention: str = "self_attn"
    attn_inputs: tuple[str, str, str] = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
    attn_output: str = "self_attn.o_proj"
    mlp_norm: str = "post_attention_layernorm"
    mlp: str = "mlp"
    mlp_inputs: tuple[str, str] = ("mlp.gate_proj", "mlp.up_proj")
    mlp_output: str = "mlp.down_proj"
    optional_maps: frozenset[str] = frozenset()
    residual_refusal: str | None = None

    def linear(self, layer: nn.Module, name: str) -> nn.Linear:
        """The linear map ``name`` of ``layer``."""
        module = layer.get_submodule(name)
        if not isinstance(module, nn.Linear):
            raise UnsupportedModelError(f"{name} of {type(layer).__name__} is a {type(module).__name__}, not nn.Linear")
        return module

    def norm(self, layer: nn.Module, name: str) -> nn.Module:
        """The norm ``name`` of ``layer`` (checked to have a weight vector)."""
        module = layer.get_submodule(name)
        if not isinstance(getattr(module, "weight", None), nn.Parameter):
            raise UnsupportedModelError(f"{name} of {type(layer).__name__} has no weight vector")
        return module

    def maps(self, layer: nn.Module, names: Iterable[str]) -> dict[str, nn.Linear]:
        """The linear maps ``names`` of ``layer`` by name, in order, without the :attr:`optional_maps` it lacks."""
        found: dict[str, nn.Linear] = {}
        for name in names:
            parent, _, attribute = name.rpartition(".")
            owner = layer.get_submodule(parent) if parent else layer
            if name in self.optional_maps and getattr(owner, attribute, None) is None:
                continue
            found[name] = self.linear(layer, name)
        return found

    def check(self, adapter: Any, technique: str) -> None:
        """Refuse the variants of the family that ``technique`` cannot handle with this layout (none here).

        Raises:
            UnsupportedModelError: In a subclass, for a model the layout does not describe.
        """

    def write_widths(self, config: Any, shapes: Sequence[LayerShape]) -> bool:
        """Write the per-layer widths ``shapes`` (in layer order) into ``config`` where it can describe them.

        An explicit ``head_dim`` is always written (it otherwise follows ``hidden_size // num_attention_heads``);
        ``num_attention_heads``, ``num_key_value_heads`` and ``intermediate_size`` only when every layer has the
        same ones, since a layer is built from the global values.

        Returns:
            Whether a layer built from the config has the widths of every layer (all layers are alike).
        """
        uniform = len(set(shapes)) == 1
        config.head_dim = shapes[0].head_dim
        if uniform:
            config.num_attention_heads = shapes[0].heads
            config.num_key_value_heads = shapes[0].kv_heads
            config.intermediate_size = shapes[0].intermediate
        return uniform


#: Decoder-layer layouts by :attr:`ArchAdapter.layer_layout <economize.arch.ArchAdapter.layer_layout>` or
#: ``config.model_type`` (an entry for the model type wins).
DECODER_LAYOUTS: Registry[DecoderLayout] = Registry("decoder layout", "economize.decoder_layouts")
DECODER_LAYOUTS.register("llama", DecoderLayout())


def layout_for(adapter: Any, technique: str, *, residual: bool = False) -> DecoderLayout:
    """The layout registered for the adapter's ``config.model_type``, else for its ``layer_layout``, after its
    :meth:`~DecoderLayout.check`.

    Args:
        adapter: The model's adapter.
        technique: The asking technique, for the error messages.
        residual: The technique rotates or cuts the residual stream (refused for a layout with a
            :attr:`~DecoderLayout.residual_refusal`).

    Raises:
        UnsupportedModelError: Neither key is registered (Gemma 2 and 3, whose ``1 + weight`` norms have no layout),
            the layout's check refuses the model, or ``residual`` is set and the layout's stream cannot be changed
            (Gemma 4's sandwich norms).
    """
    model_type = getattr(adapter.config, "model_type", None)
    layout = getattr(adapter, "layer_layout", None)
    for key in (model_type, layout):
        if key is not None and key in DECODER_LAYOUTS:
            found = DECODER_LAYOUTS.get(key)
            if residual and found.residual_refusal is not None:
                raise UnsupportedModelError(f"{technique} changes the residual stream, which the {key!r} decoder "
                                            f"layout does not allow: {found.residual_refusal}")
            found.check(adapter, technique)
            return found
    raise UnsupportedModelError(
        f"{technique} has no decoder layout for model_type {model_type!r} (layer layout {layout!r}; registered: "
        f"{DECODER_LAYOUTS.names()}); register one with "
        "economize.techniques._structured.DECODER_LAYOUTS.register(model_type, DecoderLayout(...))"
    )
