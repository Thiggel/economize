"""KV-cache techniques on SGLang: an unscaled fp8 cache for every layer (``kv_cache_dtype``)."""

from __future__ import annotations

from typing import Any

from torch import nn

from ...exporters import UnsupportedBackendError
from .base import Role, SglangPlanBuilder, SglangTechnique, Support, register_sglang_technique

__all__ = ["FP8KVSglang", "FusedFP8KVSglang", "kv_cache_dtype"]


def kv_cache_dtype(fmt: str, scale: str, layers: tuple[int, ...], num_layers: int) -> str:
    """SGLang's ``kv_cache_dtype`` of an fp8 cache.

    Args:
        fmt: ``"e4m3"`` or ``"e5m2"``.
        scale: ``"none"`` (unscaled casts); ``"token"`` and ``"tensor"`` are refused.
        layers: 1-indexed layers whose cache is fp8 (empty: all).
        num_layers: Decoder layers of the model.

    Raises:
        UnsupportedBackendError: A scale or layer subset SGLang has no cache setting for.
    """
    if scale == "token":
        raise UnsupportedBackendError("SGLang has no per-token-head fp8 KV cache: use scale='none' (unscaled casts)")
    if scale == "tensor":
        raise UnsupportedBackendError(
            "SGLang does not read the compressed-tensors kv_cache_scheme with calibrated per-layer k/v scales: use "
            "scale='none' (unscaled casts) or serve on vLLM"
        )
    if scale != "none":
        raise UnsupportedBackendError(f"SGLang has no fp8 KV cache with scale={scale!r}")
    if layers and set(layers) != set(range(1, num_layers + 1)):
        raise UnsupportedBackendError("SGLang applies kv_cache_dtype to every layer (it has no skip-layer list): use "
                                      "layers=() for an fp8 cache in all layers")
    return f"fp8_{fmt}"


@register_sglang_technique
class FP8KVSglang(SglangTechnique):
    """``kv_fp8``: SGLang's fp8 KV cache (``kv_cache_dtype=fp8_e4m3`` or ``fp8_e5m2``)."""

    technique = "kv_fp8"
    role = Role.KV_CACHE
    support = Support.NATIVE
    reason = ("kv_cache_dtype fp8_e4m3 / fp8_e5m2 with unscaled casts in all layers (no per-token scales, "
              "calibrated scales or layer subsets)")

    def _dtype(self, technique: nn.Module, builder: SglangPlanBuilder) -> str:
        config: Any = technique.config
        return kv_cache_dtype(str(config.format), str(config.scale), tuple(config.layers),
                              int(builder.model.adapter.num_layers))

    def check(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        self._dtype(technique, builder)

    def contribute(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        builder.set_engine("kv_cache_dtype", self._dtype(technique, builder), owner=self.technique)
        builder.note(self.technique, "SGLang casts K and V to fp8 when it writes the cache and back when attention "
                                     "reads it (its attention backends also read the fp8 cache directly)")


@register_sglang_technique
class FusedFP8KVSglang(FP8KVSglang):
    """``kv_fp8_fused``: the fp8 cache read inside attention, which SGLang's fp8 cache does as well."""

    technique = "kv_fp8_fused"
