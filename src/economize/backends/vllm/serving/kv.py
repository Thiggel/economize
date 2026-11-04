"""vLLM adapters of the KV-cache techniques: fp8 caches map onto vLLM's ``kv_cache_dtype``.

=================================  ==============================================================================
economize                          vLLM 0.30
=================================  ==============================================================================
``kv_fp8`` e4m3 / e5m2, no scale   ``kv_cache_dtype="fp8_e4m3"`` / ``"fp8_e5m2"`` (scale 1, as economize's)
``kv_fp8`` e4m3, ``scale="token"``  ``kv_cache_dtype="fp8_per_token_head"`` (one scale per token and KV head,
                                   computed in the kernel), on the ``TRITON_ATTN`` backend
``kv_fp8`` e4m3, ``scale="tensor"`` ``kv_cache_dtype="fp8_e4m3"`` with the calibrated per-layer scales in the
                                   checkpoint: ``<attention>.k_scale`` / ``.v_scale`` tensors and a
                                   compressed-tensors ``kv_cache_scheme`` (the llm-compressor fp8 KV format)
``layers=(...)``                   ``kv_cache_dtype_skip_layers``: the other layers (0-indexed) stay in the model
                                   dtype
``kv_fp8_fused``                   the same as ``kv_fp8`` (vLLM's attention reads fp8 pages directly)
=================================  ==============================================================================

vLLM has one ``kv_cache_dtype`` per engine, so a plan holds at most one KV-cache technique (``Role.KV_CACHE``).
vLLM refuses ``fp8_e5m2`` for a checkpoint with fp8 KV scales, so calibrated scales are e4m3 only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ...exporters import UnsupportedBackendError
from .base import PlanBuilder, Role, Support, VllmTechnique, register_vllm_technique

__all__ = [
    "KV_CACHE_SCHEME",
    "KV_SCHEME_METHODS",
    "TOKEN_SCALE_ATTENTION",
    "FP8KVVllm",
    "FusedFP8KVVllm",
    "attention_modules",
    "kv_cache_settings",
]

#: The compressed-tensors ``kv_cache_scheme`` of one static fp8 (e4m3) scale per layer for keys and for values.
KV_CACHE_SCHEME: dict[str, Any] = {
    "num_bits": 8,
    "type": "float",
    "strategy": "tensor",
    "dynamic": False,
    "symmetric": True,
}

#: The attention backends that read a ``fp8_per_token_head`` cache: vLLM's Triton attention, and economize's
#: StreamingLLM backend (``CUSTOM``), which is Triton attention with another mask.
TOKEN_SCALE_ATTENTION: tuple[str, ...] = ("TRITON_ATTN", "CUSTOM")

#: A compressed-tensors ``quantization_config`` that quantizes only the KV cache (no linear layer is quantized).
_KV_ONLY_CONFIG: dict[str, Any] = {
    "quant_method": "compressed-tensors",
    "format": "dense",
    "config_groups": {},
    "ignore": [],
    "quantization_status": "compressed",
}


#: The ``quant_method`` values whose vLLM config reads a ``kv_cache_scheme``; a plugin quantization method that
#: serves the calibrated scales adds its name (e.g. :mod:`economize.backends.vllm.serving.teal`).
KV_SCHEME_METHODS: set[str] = {"compressed-tensors"}


def kv_cache_settings(fmt: str, scale: str, layers: tuple[int, ...], num_layers: int) -> dict[str, object]:
    """The vLLM engine arguments of an fp8 KV cache (``scale="token"`` also needs one of the attention backends
    :data:`TOKEN_SCALE_ATTENTION`, which the adapter requires).

    Args:
        fmt: ``"e4m3"`` or ``"e5m2"``.
        scale: ``"none"`` (unscaled casts), ``"token"`` (one absmax scale per token and head) or ``"tensor"``
            (static per-layer scales, which the checkpoint holds).
        layers: 1-indexed layers whose cache is fp8 (empty: all).
        num_layers: Decoder layers of the model.

    Raises:
        UnsupportedBackendError: A format and scale vLLM has no cache dtype for.
    """
    if scale == "none":
        engine: dict[str, object] = {"kv_cache_dtype": f"fp8_{fmt}"}
    elif scale == "token" and fmt == "e4m3":
        engine = {"kv_cache_dtype": "fp8_per_token_head"}
    elif scale == "tensor" and fmt == "e4m3":
        engine = {"kv_cache_dtype": "fp8_e4m3"}
    elif scale == "tensor":
        raise UnsupportedBackendError("vLLM refuses an fp8_e5m2 KV cache for a checkpoint with fp8 KV scales: "
                                      "calibrate the scales with format='e4m3'")
    else:
        raise UnsupportedBackendError(f"vLLM has no {fmt} KV cache with scale={scale!r}: per-token-head fp8 is "
                                      "e4m3 only (use format='e4m3', or scale='none')")
    if layers:
        kept = set(layers)
        skipped = [str(i - 1) for i in range(1, num_layers + 1) if i not in kept]
        if skipped:
            engine["kv_cache_dtype_skip_layers"] = skipped
    return engine


def attention_modules(base: nn.Module, layers: nn.ModuleList) -> list[str]:
    """The checkpoint name of each decoder layer's attention module (``...layers.<i>.self_attn``).

    The attention module is the layer's ``self_attn`` child, or else its first submodule with ``k_proj`` and
    ``v_proj``. vLLM loads ``<attention>.k_scale`` into the scale of that layer's attention.

    Raises:
        UnsupportedBackendError: A layer has no recognizable attention module.
    """
    names = {id(module): name for name, module in base.named_modules()}
    found = []
    for index, layer in enumerate(layers):
        prefix = names.get(id(layer))
        child: str | None = "self_attn"
        if getattr(layer, "self_attn", None) is None:
            child = next((name for name, module in layer.named_modules()
                          if hasattr(module, "k_proj") and hasattr(module, "v_proj")), None)
        if prefix is None or child is None:
            raise UnsupportedBackendError(f"decoder layer {index} has no attention module to hold its KV scales")
        found.append(f"{prefix}.{child}")
    return found


def _add_kv_scheme(builder: PlanBuilder, folder: Path) -> None:
    """Add the fp8 ``kv_cache_scheme`` to the written ``config.json`` (a KV-only quantization config if none)."""
    path = folder / "config.json"
    config = json.loads(path.read_text())
    quantization = config.get("quantization_config") or dict(_KV_ONLY_CONFIG)
    if quantization.get("quant_method") not in KV_SCHEME_METHODS:
        raise UnsupportedBackendError(f"the base checkpoint is quantized with {quantization.get('quant_method')!r}; "
                                      "calibrated fp8 KV scales need a compressed-tensors (or unquantized) base")
    quantization["kv_cache_scheme"] = dict(KV_CACHE_SCHEME)
    config["quantization_config"] = quantization
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


@register_vllm_technique
class FP8KVVllm(VllmTechnique):
    """``kv_fp8``: an fp8 KV cache (vLLM's ``kv_cache_dtype``)."""

    technique = "kv_fp8"
    role = Role.KV_CACHE
    support = Support.NATIVE
    reason = "kv_cache_dtype fp8_e4m3 / fp8_e5m2 / fp8_per_token_head, kv_cache_dtype_skip_layers for a layer subset"

    def _settings(self, technique: nn.Module, builder: PlanBuilder) -> dict[str, object]:
        config = technique.config
        return kv_cache_settings(config.format, config.scale, tuple(config.layers),
                                 int(builder.model.adapter.num_layers))

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        self._settings(technique, builder)

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        for key, value in self._settings(technique, builder).items():
            builder.set_engine(key, value, owner=self.technique)
        if technique.config.scale == "token":
            builder.require_attention(*TOKEN_SCALE_ATTENTION, owner=self.technique)
            builder.note(self.technique, "vLLM computes the per-token scales in its Triton attention kernel; the "
                                         "rounding of the scale may differ from economize's in the last bit")
        if technique.config.scale == "tensor":
            self._export_scales(technique, builder)

    def _export_scales(self, technique: nn.Module, builder: PlanBuilder) -> None:
        keys, values = technique.static_scales()
        # Every layer gets its scales (a complete checkpoint for any loader); the layers outside ``layers`` are in
        # kv_cache_dtype_skip_layers, so vLLM keeps their cache in the model dtype and never reads their scales.
        for index, name in enumerate(attention_modules(builder.base, builder.model.adapter.layers)):
            builder.extra_state[f"{name}.k_scale"] = keys[index].reshape(1).to(torch.float32)
            builder.extra_state[f"{name}.v_scale"] = values[index].reshape(1).to(torch.float32)
        builder.post_writers.append(_add_kv_scheme)
        builder.note(self.technique, "calibrated per-layer k_scale / v_scale in the checkpoint (compressed-tensors "
                                     "kv_cache_scheme); a drafter's own attention layers use unit scales")


@register_vllm_technique
class FusedFP8KVVllm(FP8KVVllm):
    """``kv_fp8_fused``: the fp8 cache read inside attention, which is how vLLM always reads an fp8 cache."""

    technique = "kv_fp8_fused"
