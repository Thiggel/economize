"""TensorRT-LLM (NVIDIA GPUs): export a model with its techniques as a TensorRT-LLM checkpoint.

Export only: the checkpoint is written on any platform with ``safetensors``; ``trtllm-build`` turns it into an
engine on a machine with TensorRT-LLM and an NVIDIA GPU::

    from economize.backends.trtllm import to_trtllm

    export = to_trtllm(model, "out/", tokenizer=tokenizer)
    print(" ".join(export.command))      # trtllm-build --checkpoint_dir out --output_dir out/engine ...
    print(export.settings)               # speculative_config / kv_cache_config for TensorRT-LLM's LLM API

The runtime follows the design of the serving backends: :data:`TRTLLM` holds one adapter per technique (registry
``economize.trtllm_techniques``) with a support level (``native``, ``densified`` or ``unsupported`` with the
reason), and :func:`economize.backend_support` reads it under the name ``"trtllm"``. Architectures are registered
in :data:`TRTLLM_ARCHS` (``economize.trtllm_archs``). Importing this package does not import ``tensorrt_llm``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .arch import TRTLLM_ARCHS, TrtArch, trt_arch
    from .engine import check_trtllm, to_trtllm, write_trtllm
    from .runtime import (
        TRTLLM,
        TRTLLM_TECHNIQUES,
        TrtDraftModel,
        TrtKvFp8,
        TrtLookup,
        TrtMedusa,
        TrtQuant,
        TrtStreaming,
    )
    from .writer import LINEARS, QUANT_ALGO, TRT_GROUP_SIZES, pack_int4_rows, trt_config, trt_reason, trt_tensors

__all__ = [
    "LINEARS",
    "QUANT_ALGO",
    "TRTLLM",
    "TRTLLM_ARCHS",
    "TRTLLM_TECHNIQUES",
    "TRT_GROUP_SIZES",
    "TrtArch",
    "TrtDraftModel",
    "TrtKvFp8",
    "TrtLookup",
    "TrtMedusa",
    "TrtQuant",
    "TrtStreaming",
    "check_trtllm",
    "pack_int4_rows",
    "to_trtllm",
    "trt_arch",
    "trt_config",
    "trt_reason",
    "trt_tensors",
    "write_trtllm",
]

_HOME = {
    "TRTLLM_ARCHS": "arch", "TrtArch": "arch", "trt_arch": "arch",
    "check_trtllm": "engine", "to_trtllm": "engine", "write_trtllm": "engine",
    "TRTLLM": "runtime", "TRTLLM_TECHNIQUES": "runtime", "TrtDraftModel": "runtime", "TrtKvFp8": "runtime",
    "TrtLookup": "runtime", "TrtMedusa": "runtime", "TrtQuant": "runtime", "TrtStreaming": "runtime",
    "LINEARS": "writer", "QUANT_ALGO": "writer", "TRT_GROUP_SIZES": "writer", "pack_int4_rows": "writer",
    "trt_config": "writer", "trt_reason": "writer", "trt_tensors": "writer",
}


def __getattr__(name: str) -> object:
    """Import the public names on first use."""
    if name not in _HOME:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{_HOME[name]}", __name__), name)
