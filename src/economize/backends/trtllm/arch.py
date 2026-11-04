"""The Hugging Face architectures the TensorRT-LLM checkpoint writer converts, by ``config.model_type``.

An entry names TensorRT-LLM's model class (the checkpoint's ``architecture``) and the extra fields of its
``config.json``. Every registered architecture uses the Llama block layout (pre-norm RMSNorm, rotary attention
with optional query/key/value biases, SwiGLU MLP), which the writer maps to TensorRT-LLM's tensor names. A new
architecture is added by registering a :class:`TrtArch` in :data:`TRTLLM_ARCHS` (or through the entry-point group
``economize.trtllm_archs``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ...errors import ConfigError
from ...registry import Registry

__all__ = ["TRTLLM_ARCHS", "TrtArch", "trt_arch"]


@dataclass(frozen=True)
class TrtArch:
    """How one Hugging Face architecture is written as a TensorRT-LLM checkpoint.

    Attributes
    ----------
    architecture : str
        TensorRT-LLM's model class (``config.json["architecture"]``).
    fields : Mapping[str, object]
        Extra fields of ``config.json`` (e.g. ``qwen_type``).
    """

    architecture: str
    fields: Mapping[str, object] = field(default_factory=dict)


#: ``config.model_type -> TrtArch``.
TRTLLM_ARCHS: Registry[TrtArch] = Registry("TensorRT-LLM architecture", "economize.trtllm_archs")
TRTLLM_ARCHS.register("llama", TrtArch("LlamaForCausalLM"))
TRTLLM_ARCHS.register("mistral", TrtArch("LlamaForCausalLM"))
TRTLLM_ARCHS.register("qwen2", TrtArch("QWenForCausalLM", {"qwen_type": "qwen2"}))


def trt_arch(model_type: str) -> TrtArch:
    """The TensorRT-LLM architecture of a Hugging Face ``model_type``.

    Parameters
    ----------
    model_type : str
        ``config.model_type``.

    Returns
    -------
    TrtArch
        The registered entry.

    Raises
    ------
    ConfigError
        No entry is registered for ``model_type``.
    """
    if model_type not in TRTLLM_ARCHS:
        raise ConfigError(f"the TensorRT-LLM writer has no architecture for model_type={model_type!r} (registered: "
                          f"{sorted(TRTLLM_ARCHS.names())}); register a TrtArch in "
                          "economize.backends.trtllm.TRTLLM_ARCHS")
    return TRTLLM_ARCHS.get(model_type)
