"""The decoder-layer forward of SliceGPT's per-block rotation: every residual connection multiplies by a matrix.

With one basis per residual point, the residual stream leaving the attention block is expressed in another basis
than the stream entering it, so the skip connection becomes ``x Q_in^T Q_mid`` (sliced to the kept widths). The
official code replaces each decoder layer by a compressed copy with ``attn_shortcut_Q`` and ``mlp_shortcut_Q``;
here the layer keeps its modules and parameters, and :func:`install_shortcuts` gives it a forward that reads the
two matrices from non-persistent buffers of the layer, so state-dict keys and ``device_map`` placement stay those of
the architecture.
"""

from __future__ import annotations

import inspect
from types import MethodType
from typing import Any

import torch
from torch import Tensor, nn

from .._structured.layout import DecoderLayout

__all__ = ["ATTN_SHORTCUT", "MLP_SHORTCUT", "install_shortcuts", "set_shortcuts", "shortcut_widths"]

#: Buffer names of the two shortcut matrices on a decoder layer.
ATTN_SHORTCUT = "slicegpt_attn_shortcut"
MLP_SHORTCUT = "slicegpt_mlp_shortcut"


def _shortcut(x: Tensor, matrix: Tensor | None) -> Tensor:
    return x if matrix is None else x @ matrix


def _named_arguments(signature: inspect.Signature, hidden: Tensor, args: tuple[Any, ...], kwargs: dict[str, Any]
                     ) -> dict[str, Any]:
    """The layer's call arguments other than the hidden state, as keywords (``**kwargs`` merged in)."""
    bound = signature.bind(hidden, *args, **kwargs)
    named: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        kind = signature.parameters[name].kind
        if kind is inspect.Parameter.VAR_KEYWORD:
            named.update(value)
        elif kind is not inspect.Parameter.VAR_POSITIONAL and name != "hidden_states":
            named[name] = value
    named.pop("hidden_states", None)
    return named


def install_shortcuts(layer: nn.Module, layout: DecoderLayout) -> None:
    """Replace ``layer.forward`` by the pre-norm block with shortcut matrices (idempotent).

    The new forward computes ``h = x S_attn + attn(norm1(x))`` and ``y = h S_mlp + mlp(norm2(h))``, where the
    matrices are the layer buffers :data:`ATTN_SHORTCUT` and :data:`MLP_SHORTCUT` (``None``: identity). It passes the
    layer's other call arguments (mask, rotary embeddings, cache, ...) to the attention module, as the
    architecture's own forward does.
    """
    if getattr(layer, "_slicegpt_layout", None) is not None:
        return
    signature = inspect.signature(layer.forward)
    for name in (ATTN_SHORTCUT, MLP_SHORTCUT):
        layer.register_buffer(name, None, persistent=False)

    def forward(self: nn.Module, hidden_states: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        named = _named_arguments(signature, hidden_states, args, kwargs)
        attention = self.get_submodule(layout.attention)(
            hidden_states=self.get_submodule(layout.attn_norm)(hidden_states), **named
        )
        attention = attention[0] if isinstance(attention, tuple) else attention
        hidden = _shortcut(hidden_states, getattr(self, ATTN_SHORTCUT)) + attention
        mlp = self.get_submodule(layout.mlp)(self.get_submodule(layout.mlp_norm)(hidden))
        return _shortcut(hidden, getattr(self, MLP_SHORTCUT)) + mlp

    layer.forward = MethodType(forward, layer)
    object.__setattr__(layer, "_slicegpt_layout", layout)


def set_shortcuts(layer: nn.Module, attn: Tensor | None, mlp: Tensor | None) -> None:
    """Store the shortcut matrices of ``layer`` (cast to its parameters' dtype and device; ``None``: identity)."""
    like = next(layer.parameters())
    for name, value in ((ATTN_SHORTCUT, attn), (MLP_SHORTCUT, mlp)):
        setattr(layer, name, None if value is None else value.to(dtype=like.dtype, device=like.device).contiguous())


def shortcut_widths(layer: nn.Module) -> tuple[torch.Size | None, torch.Size | None]:
    """The shapes of the two shortcut matrices of ``layer`` (for summaries and tests)."""
    attn, mlp = getattr(layer, ATTN_SHORTCUT, None), getattr(layer, MLP_SHORTCUT, None)
    return (None if attn is None else attn.shape, None if mlp is None else mlp.shape)
