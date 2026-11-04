"""The composition root: it connects the lower layers to implementations that live in higher ones.

A layer imports only layers below it (``docs/ARCHITECTURE.md``, Section 2.1). Where a lower layer needs behaviour
from a higher one, it exposes a registry and the package root registers the implementation here, importing the
higher module on first use so that ``import economize`` stays light.
"""

from __future__ import annotations

from typing import Any

from .core.compatibility import BACKEND_SUPPORT, BackendSupport
from .core.config import TechniqueConfig
from .core.model import GENERATE_IMPLEMENTATIONS
from .core.technique import Technique

__all__: list[str] = []


@GENERATE_IMPLEMENTATIONS.register("transformers")
def _transformers_generate(model: Any, inputs: Any = None, **kwargs: Any) -> Any:
    """:func:`economize.integrations.generate.generate`, imported on the first ``EfficientModel.generate`` call."""
    from .integrations.generate import generate

    return generate(model, inputs, **kwargs)


@BACKEND_SUPPORT.register("vllm")
def _vllm_support(technique: type[Technique], config: TechniqueConfig | None) -> BackendSupport:
    """The vLLM adapters (:mod:`economize.backends.vllm.serving`, imported on first use; vLLM itself is not)."""
    from .backends.support import serving_support
    from .backends.vllm.serving import VLLM_TECHNIQUES

    return serving_support("vLLM", VLLM_TECHNIQUES, technique, config)


@BACKEND_SUPPORT.register("sglang")
def _sglang_support(technique: type[Technique], config: TechniqueConfig | None) -> BackendSupport:
    """The SGLang adapters (:mod:`economize.backends.sglang.serving`, imported on first use; SGLang itself is not)."""
    from .backends.sglang.serving import SGLANG_TECHNIQUES
    from .backends.support import serving_support

    return serving_support("SGLang", SGLANG_TECHNIQUES, technique, config)


@BACKEND_SUPPORT.register("gguf")
def _gguf_support(technique: type[Technique], config: TechniqueConfig | None) -> BackendSupport:
    """The llama.cpp export adapters (:mod:`economize.backends.gguf.runtime`; imports neither gguf nor llama.cpp)."""
    from .backends.gguf.runtime import GGUF

    return GGUF.support(technique, config)


@BACKEND_SUPPORT.register("onnx")
def _onnx_support(technique: type[Technique], config: TechniqueConfig | None) -> BackendSupport:
    """The ONNX Runtime export adapters (:mod:`economize.backends.onnx.runtime`; imports neither onnx nor ORT)."""
    from .backends.onnx.runtime import ONNX

    return ONNX.support(technique, config)


@BACKEND_SUPPORT.register("mlx")
def _mlx_support(technique: type[Technique], config: TechniqueConfig | None) -> BackendSupport:
    """The MLX export adapters (:mod:`economize.backends.mlx.runtime`; does not import mlx)."""
    from .backends.mlx.runtime import MLX

    return MLX.support(technique, config)


@BACKEND_SUPPORT.register("trtllm")
def _trtllm_support(technique: type[Technique], config: TechniqueConfig | None) -> BackendSupport:
    """The TensorRT-LLM export adapters (:mod:`economize.backends.trtllm.runtime`; does not import tensorrt_llm)."""
    from .backends.trtllm.runtime import TRTLLM

    return TRTLLM.support(technique, config)
