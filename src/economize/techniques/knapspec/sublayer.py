"""Running a decoder layer without its attention or without its MLP: the sublayers KnapSpec skips.

A pre-norm decoder layer adds its attention and its MLP to the residual stream one after the other. Replacing the
modules of one of them with a stand-in that returns zeros leaves the residual stream unchanged by that sublayer (the
post-norms of Gemma and OLMo-2 map zeros to zeros), so the layer called with its MLP replaced computes ``h + A(h)``
and the layer called with its attention replaced computes ``h + M(h)``. Running the first and then the second on
its output reproduces the whole layer. A replaced attention writes no K/V, as a layer skipped by
:class:`~economize.techniques.layer_skip.SkipLayers`.

Which modules form each sublayer is a :class:`SublayerLayout`, found per model type in :data:`SUBLAYER_LAYOUTS`
(default: the layer's token mixer, and the first of ``mlp`` / ``feed_forward`` / ``block_sparse_moe``). A family
with another layout registers a function there.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ...arch.spec import StateKind
from ...errors import ConfigError
from ...registry import Registry
from ..layer_skip.policy import SkipLayers

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...runtime import LayerCall

__all__ = ["SUBLAYER_LAYOUTS", "PartialLayer", "SublayerLayout", "SublayerSkipper", "default_layout"]

_MLPS = ("mlp", "feed_forward", "block_sparse_moe")


@dataclass(frozen=True)
class SublayerLayout:
    """The sub-modules of one decoder layer that make up its two sublayers.

    Attributes:
        attention: Name of the attention module (``None``: the layer has none, its first sublayer cannot be skipped).
        mlp: Names of the modules whose outputs form the MLP sublayer (empty: it cannot be skipped).
        scale: Name of a buffer or parameter that scales the whole residual stream at the end of the layer
            (Gemma-4's ``layer_scalar``). It counts to the MLP sublayer: a layer whose MLP is skipped leaves the
            stream unscaled, as does a layer skipped whole.
    """

    attention: str | None
    mlp: tuple[str, ...]
    scale: str | None = None


#: Sublayer layouts by model type: ``fn(adapter, layer) -> SublayerLayout``. Unregistered types use
#: :func:`default_layout`.
SUBLAYER_LAYOUTS: Registry[Callable[[ArchAdapter, nn.Module], SublayerLayout]] = Registry(
    "sublayer layout", "economize.sublayer_layouts"
)


def default_layout(adapter: ArchAdapter, layer: nn.Module) -> SublayerLayout:
    """The layer's token mixer (:meth:`ArchAdapter.mixer`) and its first MLP-like module."""
    mixer = adapter.mixer(layer)
    attention = next((name for name, module in layer.named_children() if module is mixer), None)
    mlp = next((name for name in _MLPS if isinstance(getattr(layer, name, None), nn.Module)), None)
    return SublayerLayout(attention, () if mlp is None else (mlp,))


def _gemma4(adapter: ArchAdapter, layer: nn.Module) -> SublayerLayout:
    # The MoE block adds its experts to the MLP's output, and the per-layer-input branch follows the MLP.
    base = default_layout(adapter, layer)
    extra = tuple(n for n in ("experts", "per_layer_projection") if isinstance(getattr(layer, n, None), nn.Module))
    return SublayerLayout(base.attention, base.mlp + extra, "layer_scalar" if hasattr(layer, "layer_scalar") else None)


for _type in ("gemma4", "gemma4_text"):
    SUBLAYER_LAYOUTS.register(_type, _gemma4)


class _ZeroAttention(nn.Module):
    """Stands in for a skipped attention module: adds nothing to the residual stream and writes no K/V."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, None]:
        hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        return hidden.new_zeros(*hidden.shape[:-1], self.width), None


class _ZeroModule(nn.Module):
    """Stands in for a skipped MLP-like module: returns zeros of the residual width for its first input's tokens."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width

    def forward(self, first: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return first.new_zeros(*first.shape[:-1], self.width)


class PartialLayer:
    """A decoder layer called with some of its sub-modules (and its residual scale) replaced, as a
    :data:`~economize.runtime.LayerCall`.

    The replacements are swapped in for the duration of each call only, so the layer stays intact for every other
    caller (the exact pass). Swapping is a host-side attribute change: a captured CUDA graph records the kernels of
    the partial layer.
    """

    def __init__(self, layer: nn.Module, modules: Mapping[str, nn.Module], scale: str | None = None) -> None:
        self.layer = layer
        self.modules = dict(modules)
        self.scale = scale
        self._unit: torch.Tensor | None = None
        if scale is not None:
            value = getattr(layer, scale)
            self._store: dict[str, Any] = layer._buffers if scale in layer._buffers else layer._parameters
            self._unit = torch.ones_like(value.detach())

    def __call__(self, hidden: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        own = self.layer._modules
        saved = {name: own[name] for name in self.modules}
        own.update(self.modules)
        if self.scale is not None:
            saved_scale = self._store[self.scale]
            self._store[self.scale] = self._unit
        try:
            return self.layer(hidden, *args, **kwargs)
        finally:
            own.update(saved)
            if self.scale is not None:
                self._store[self.scale] = saved_scale


class SublayerSkipper:
    """The partial-layer calls of one model: which sublayers may be skipped, and the calls of a skip set.

    Sublayers are numbered ``0 .. 2L - 1``: ``2 (l - 1)`` is the attention of layer ``l`` (1-indexed), ``2 l - 1``
    its MLP. The attention of a layer cannot be skipped when the layer keeps a recurrent state, or when other layers
    attend over its cache entries (KV sharing); a sublayer without modules cannot be skipped either.

    Args:
        adapter: The model's adapter.
    """

    def __init__(self, adapter: ArchAdapter) -> None:
        self.adapter = adapter
        L = adapter.num_layers
        self.layouts = {i: SUBLAYER_LAYOUTS.get(adapter.model_type)(adapter, adapter.layer(i))
                        if adapter.model_type in SUBLAYER_LAYOUTS else default_layout(adapter, adapter.layer(i))
                        for i in range(1, L + 1)}
        sources = {adapter.kv_source(i) for i in range(1, L + 1) if adapter.kv_source(i) != i}
        self.skippable: tuple[bool, ...] = tuple(
            flag
            for i in range(1, L + 1)
            for flag in (
                self.layouts[i].attention is not None and i not in sources
                and adapter.state_kind(i) is not StateKind.RECURRENT,
                bool(self.layouts[i].mlp),
            )
        )
        self._calls: dict[tuple[int, str], PartialLayer] = {}

    @property
    def num_sublayers(self) -> int:
        return 2 * self.adapter.num_layers

    def attention_only(self, index: int) -> PartialLayer:
        """Layer ``index`` with its MLP sublayer skipped (``h + A(h)``)."""
        key = (index, "attention")
        if key not in self._calls:
            layout, width = self.layouts[index], self.adapter.hidden_size
            self._calls[key] = PartialLayer(self.adapter.layer(index), {n: _ZeroModule(width) for n in layout.mlp},
                                            layout.scale)
        return self._calls[key]

    def mlp_only(self, index: int) -> PartialLayer:
        """Layer ``index`` with its attention sublayer skipped (``h + M(h)``, scaled as the layer scales)."""
        key = (index, "mlp")
        if key not in self._calls:
            layout = self.layouts[index]
            if layout.attention is None:
                raise ConfigError(f"layer {index} has no attention module to skip")
            self._calls[key] = PartialLayer(self.adapter.layer(index),
                                            {layout.attention: _ZeroAttention(self.adapter.hidden_size)})
        return self._calls[key]

    def sublayer_call(self, sublayer: int) -> PartialLayer:
        """The call that runs sublayer ``sublayer`` (0-indexed) alone."""
        index = sublayer // 2 + 1
        return self.attention_only(index) if sublayer % 2 == 0 else self.mlp_only(index)

    def calls(self, attention: frozenset[int], mlp: frozenset[int]) -> tuple[dict[int, LayerCall], SkipLayers]:
        """The overrides and the policy of a draft step that skips the attention of the layers ``attention`` and
        the MLP of the layers ``mlp``: layers in both are skipped whole (no K/V), the others run partially."""
        both = attention & mlp
        overrides: dict[int, LayerCall] = {i: self.mlp_only(i) for i in attention - both}
        overrides.update({i: self.attention_only(i) for i in mlp - both})
        return overrides, SkipLayers(both)
