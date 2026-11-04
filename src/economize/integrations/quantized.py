"""Base models loaded through a ``transformers`` quantizer (bitsandbytes, torchao, compressed-tensors, GPTQ, AWQ, ...).

economize runs on such bases without special handling: the techniques read hidden states and call the base model's
own (quantized) linear maps, and the exact path is the quantized model itself, so lossless decoding reproduces
the quantized model's greedy output. What changes is the teacher: a drafter trained on the full-precision model
is accepted less often on a quantized one. This module therefore describes a base model's quantization, which
checkpoints record in ``economize_config.json`` next to the weight fingerprint, with a warning when a checkpoint is
loaded onto a differently quantized base. :func:`describe_quantization` lives in :mod:`economize.core.hub`, which writes
that record (the core does not import the integrations).

    from economize.integrations.quantized import describe_quantization
    base = AutoModelForCausalLM.from_pretrained(name, quantization_config=BitsAndBytesConfig(load_in_4bit=True))
    describe_quantization(base)   # {"method": "bitsandbytes", "quantizer": "Bnb4BitHfQuantizer", ...}

Exporting the weights of economize's own ``quant`` technique for vLLM lives in
:mod:`economize.integrations.compressed_tensors`.
"""

from __future__ import annotations

from typing import Any

from ..core.hub import describe_quantization  # lives in the core, which records it in checkpoints

__all__ = ["describe_quantization", "is_quantized", "same_quantization"]


def is_quantized(model: Any) -> bool:
    """Whether ``model``'s base was loaded through a ``transformers`` quantizer."""
    return describe_quantization(model) is not None


def same_quantization(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    """Whether two :func:`describe_quantization` results describe the same quantization (method and config; the
    quantizer class and whether the weights were pre-quantized do not matter)."""
    if a is None or b is None:
        return a is b
    return a.get("method") == b.get("method") and a.get("config") == b.get("config")

