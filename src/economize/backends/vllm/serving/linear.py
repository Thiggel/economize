"""The ``quantization_config`` of economize's linear method on vLLM, which ``teal`` and ``lowrank`` share.

vLLM builds every linear map of a model with one quantization method, so the techniques that change how a dense
linear map computes write one section each into the same config (see :mod:`economize.backends.vllm.linear`). An
adapter adds its section with :meth:`~economize.backends.vllm.serving.base.PlanBuilder.add_linear_section`;
:func:`economize.backends.vllm.serving.base.to_vllm` writes the config right after the base checkpoint, before the
post-writers, so a calibrated fp8 KV cache adds its ``kv_cache_scheme`` to it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...exporters import UnsupportedBackendError
from ..linear import QUANT_METHOD
from .kv import KV_SCHEME_METHODS

__all__ = ["write_linear_config"]

KV_SCHEME_METHODS.add(QUANT_METHOD)  # EconomizeLinearConfig serves calibrated fp8 KV scales


def _kv_only(config: Mapping[str, Any]) -> bool:
    """Whether a ``quantization_config`` quantizes the KV cache only (a compressed-tensors config without groups)."""
    return (config.get("quant_method") == "compressed-tensors" and not config.get("config_groups")
            and not config.get("sparsity_config") and not config.get("transform_config"))


def write_linear_config(path: Path, sections: Mapping[str, Mapping[str, Any]]) -> None:
    """Write the ``economize`` ``quantization_config`` with ``sections`` into the ``config.json`` at ``path``.

    A KV-only compressed-tensors config already in the file gives its ``kv_cache_scheme`` to the new config.

    Raises:
        UnsupportedBackendError: The checkpoint is quantized with another method (vLLM loads one per model).
    """
    config = json.loads(path.read_text())
    existing = config.get("quantization_config")
    linear: dict[str, Any] = {"quant_method": QUANT_METHOD, **{name: dict(body) for name, body in sections.items()}}
    if existing:
        if existing.get("quant_method") != QUANT_METHOD and not _kv_only(existing):
            raise UnsupportedBackendError(
                f"the served checkpoint is quantized with {existing.get('quant_method')!r}; vLLM loads one "
                f"quantization method per model, so {', '.join(sections)} serve dense (or densified) weights only")
        if existing.get("kv_cache_scheme") is not None:
            linear["kv_cache_scheme"] = existing["kv_cache_scheme"]
    config["quantization_config"] = linear
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
