"""ONNX Runtime: export a model with its techniques as one ONNX graph with KV-cache inputs.

Install with ``pip install economize[onnx]``::

    from economize.backends.onnx import load_onnx, to_onnx

    export = to_onnx(model, "out/", tokenizer=tokenizer)
    tokens = load_onnx(export).generate(input_ids, max_new_tokens=32)

The runtime follows the design of the serving backends: :data:`ONNX` holds one adapter per technique (registry
``economize.onnx_techniques``) with a support level (``native``, ``densified`` or ``unsupported`` with the reason),
and :func:`economize.backend_support` reads it under the name ``"onnx"``. Importing this package imports neither
``onnx`` nor ``onnxruntime``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import OnnxCausalLM, check_onnx, load_onnx, plain_model, to_onnx
    from .graph import matmul_nbits, nbits_reason, pack_bits, trace_onnx
    from .runtime import ONNX, ONNX_TECHNIQUES, OnnxQuant

__all__ = [
    "ONNX",
    "ONNX_TECHNIQUES",
    "OnnxCausalLM",
    "OnnxQuant",
    "check_onnx",
    "load_onnx",
    "matmul_nbits",
    "nbits_reason",
    "pack_bits",
    "plain_model",
    "to_onnx",
    "trace_onnx",
]

_HOME = {
    "OnnxCausalLM": "engine", "check_onnx": "engine", "load_onnx": "engine", "plain_model": "engine",
    "to_onnx": "engine",
    "matmul_nbits": "graph", "nbits_reason": "graph", "pack_bits": "graph", "trace_onnx": "graph",
    "ONNX": "runtime", "ONNX_TECHNIQUES": "runtime", "OnnxQuant": "runtime",
}


def __getattr__(name: str) -> object:
    """Import the public names on first use."""
    if name not in _HOME:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{_HOME[name]}", __name__), name)
