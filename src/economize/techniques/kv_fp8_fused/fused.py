"""``kv_fp8_fused``: :mod:`economize.techniques.kv_quant.fp8` with the upcast moved into the attention kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from ...attention.registry import ATTENTION_KERNELS, accepts_fp8_kv
from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import register_technique
from ...errors import ConfigError, IncompatibleTechniquesError
from ...runtime import KVCache
from .._kv import CacheWrapper, layer_indices
from ..kv_quant.fp8 import FP8Cache, FP8KVConfig, FP8KVTechnique

__all__ = ["FusedFP8Cache", "FusedFP8KVConfig", "FusedFP8KVTechnique"]


@dataclass(frozen=True)
class FusedFP8KVConfig(FP8KVConfig):
    """FP8 KV cache whose bytes the attention kernel reads directly.

    Attributes:
        format: ``"e4m3"`` or ``"e5m2"`` (as for :class:`~economize.techniques.kv_quant.FP8KVConfig`).
        scale: Must be ``"none"``.
        layers: 1-indexed layers whose cache is stored in fp8 (empty: every layer).
    """

    technique: ClassVar[str] = "kv_fp8_fused"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scale != "none":
            raise ConfigError("kv_fp8_fused supports scale='none' only; use kv_fp8 for per-token scales")

    def validate(self, adapter: Any) -> None:
        super().validate(adapter)
        implementation = str(getattr(adapter.config, "_attn_implementation", None) or "eager")
        if not accepts_fp8_kv(implementation):
            fused = [name for name in ATTENTION_KERNELS.names() if accepts_fp8_kv(name)]
            raise IncompatibleTechniquesError(
                f"kv_fp8_fused needs an attention kernel that reads fp8 K/V, the model runs {implementation!r}; "
                f"load it with attn_implementation in {fused} (after economize.attention.install()) or use kv_fp8"
            )


class FusedFP8Cache(FP8Cache):
    """An :class:`~economize.techniques.kv_quant.FP8Cache` whose :meth:`update` (what attention consumes) returns
    the stored buffers as fp8 views. :meth:`read`, used outside attention, still returns the model's dtype."""

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.covers(layer_idx):
            return super().update(key_states, value_states, layer_idx, *args, **kwargs)
        key_states, value_states = self.transform_update(key_states, value_states, layer_idx)
        keys, values = self.inner.update(key_states, value_states, layer_idx, *args, **kwargs)
        return keys.view(self.fp8), values.view(self.fp8)


@register_technique
class FusedFP8KVTechnique(FP8KVTechnique):
    """KV cache in 8-bit floats, decoded inside the attention kernel (see :class:`FusedFP8KVConfig`)."""

    name: ClassVar[str] = FusedFP8KVConfig.technique
    config_class: ClassVar[type[FusedFP8KVConfig]] = FusedFP8KVConfig  # type: ignore[assignment]
    info = TechniqueInfo(family=Family.KV_QUANTIZATION, saves=Saving.KV_MEMORY | Saving.LATENCY, lossless=False)
    graph_safe: ClassVar[bool] = True  # scale="none": layers get fp8 views of the fixed buffers, no side state
    config: FusedFP8KVConfig

    def make_wrapper(self, cache: KVCache) -> CacheWrapper:
        return FusedFP8Cache(cache, self.config, layer_indices(self.config.layers, self.adapter))
