"""The Hugging Face architectures the GGUF writer converts, by ``config.model_type``.

An entry says which llama.cpp architecture runs the model, whether llama.cpp's rotary layout needs the query and
key rows permuted (Hugging Face's Llama checkpoints rotate halves, llama.cpp's ``llama`` architecture rotates
interleaved pairs; Qwen2 and Qwen3 rotate halves in both), and the pre-tokenizer name of its BPE vocabulary. A new
architecture is added by registering a :class:`GgufArch` in :data:`GGUF_ARCHS` (or through the entry-point group
``economize.gguf_archs``); llama.cpp must know the architecture, and gguf-py's tensor-name map must cover its
Hugging Face tensor names.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...errors import ConfigError
from ...registry import Registry

__all__ = ["GGUF_ARCHS", "GgufArch", "gguf_arch"]


@dataclass(frozen=True)
class GgufArch:
    """How one Hugging Face architecture is written to GGUF.

    Attributes
    ----------
    name : str
        The llama.cpp architecture (``general.architecture``), a name of ``gguf.MODEL_ARCH_NAMES``.
    permute_qk : bool
        Permute the rows of the query and key maps from Hugging Face's rotate-half layout to llama.cpp's
        interleaved one (``convert_hf_to_gguf.py``'s ``LlamaModel.permute``).
    pre_tokenizer : str
        ``tokenizer.ggml.pre`` of a BPE vocabulary (llama.cpp selects its pre-tokenizer regex by this name).
    """

    name: str
    permute_qk: bool = False
    pre_tokenizer: str = "default"


#: ``config.model_type -> GgufArch``.
GGUF_ARCHS: Registry[GgufArch] = Registry("GGUF architecture", "economize.gguf_archs")
GGUF_ARCHS.register("qwen2", GgufArch("qwen2", pre_tokenizer="qwen2"))
GGUF_ARCHS.register("qwen3", GgufArch("qwen3", pre_tokenizer="qwen2"))
GGUF_ARCHS.register("llama", GgufArch("llama", permute_qk=True, pre_tokenizer="llama-bpe"))
GGUF_ARCHS.register("mistral", GgufArch("llama", permute_qk=True, pre_tokenizer="llama-bpe"))


def gguf_arch(model_type: str) -> GgufArch:
    """The GGUF architecture of a Hugging Face ``model_type``.

    Parameters
    ----------
    model_type : str
        ``config.model_type``.

    Returns
    -------
    GgufArch
        The registered entry.

    Raises
    ------
    ConfigError
        No entry is registered.
    """
    if model_type not in GGUF_ARCHS:
        raise ConfigError(f"no GGUF architecture is registered for model_type {model_type!r} (registered: "
                          f"{sorted(GGUF_ARCHS.names())}); register a GgufArch in economize.backends.gguf.GGUF_ARCHS, "
                          "or convert the dense model with llama.cpp's convert_hf_to_gguf.py")
    return GGUF_ARCHS.get(model_type)
