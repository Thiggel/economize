"""Shared machinery of the techniques that change the widths of a model (SliceGPT, structured width pruning):
decoder-layer layouts (:mod:`.layout`, Gemma 4's in :mod:`.gemma4`) and in-place surgery of linear maps, embeddings
and norms (:mod:`.surgery`)."""

from .gemma4 import Gemma4Layout
from .layout import DECODER_LAYOUTS, DecoderLayout, LayerShape, layout_for
from .surgery import (
    fold_rotate_inputs,
    keep_input_channels,
    keep_output_channels,
    rotate_embedding,
    rotate_outputs,
    set_linear,
    unit_norm,
)

__all__ = [
    "DECODER_LAYOUTS",
    "DecoderLayout",
    "Gemma4Layout",
    "LayerShape",
    "fold_rotate_inputs",
    "keep_input_channels",
    "keep_output_channels",
    "layout_for",
    "rotate_embedding",
    "rotate_outputs",
    "set_linear",
    "unit_norm",
]
