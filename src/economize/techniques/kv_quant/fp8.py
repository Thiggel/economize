"""FP8 KV cache: every K/V entry stored as an 8-bit float, as in vLLM's and TensorRT-LLM's ``fp8`` cache dtype.

Entries are cast to ``float8_e4m3fn`` (default) or ``float8_e5m2`` when written and cast back when attention reads
them. ``scale="none"`` casts directly (vLLM's default with unit scales); ``scale="token"`` divides every token's
head vector by its absolute maximum over the representable range first, which keeps outliers representable at the
cost of one fp32 scale per token and head; ``scale="tensor"`` divides every entry of a layer's keys (and of its
values) by one static scale, calibrated as the layer's absolute maximum on calibration data over the format's
maximum. The static scales are the ``k_scale`` / ``v_scale`` of an llm-compressor fp8 KV checkpoint, which is how
:func:`economize.backends.vllm.to_vllm` exports them. The inner cache stores the raw bytes (``uint8``), so the
memory held is the memory an fp8 cache needs. Drafts, tree compaction and dropped rows are handled by the inner
cache, with the scales following the same slots.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from ...core.calibration import calibration_source, token_batches
from ...core.config import TechniqueConfig
from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import AttachContext, register_technique
from ...errors import ConfigError
from ...runtime import KVCache
from ...validation import check_choice, check_layers
from .._kv import CacheWrapper, KVCacheTechnique, SlotStore, check_positional, layer_indices

__all__ = [
    "FP8_FORMATS",
    "FP8_SCALES",
    "FP8Cache",
    "FP8KVConfig",
    "FP8KVTechnique",
    "calibrate_kv_absmax",
    "fp8_round_trip",
]

_log = logging.getLogger(__name__)

#: Name -> torch dtype of the supported 8-bit float formats.
FP8_FORMATS: dict[str, torch.dtype] = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
FP8_SCALES = ("none", "token", "tensor")


@dataclass(frozen=True)
class FP8KVConfig(TechniqueConfig):
    """FP8 KV cache.

    Attributes:
        format: ``"e4m3"`` (``float8_e4m3fn``, 3 mantissa bits) or ``"e5m2"`` (wider range, 2 mantissa bits).
        scale: ``"none"`` (cast directly, saturating at the format's maximum), ``"token"`` (per-token, per-head
            absmax scale stored in fp32) or ``"tensor"`` (one static scale per layer for its keys and one for its
            values, calibrated at attach time: pass calibration batches to :func:`economize.apply`).
        layers: 1-indexed layers whose cache is stored in fp8 (empty: every layer).
    """

    technique: ClassVar[str] = "kv_fp8"
    format: str = "e4m3"
    scale: str = "none"
    layers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        check_choice("format", self.format, tuple(FP8_FORMATS))
        check_choice("scale", self.scale, FP8_SCALES)
        check_layers(self.layers)

    def validate(self, adapter: Any) -> None:
        check_layers(self.layers, adapter)
        check_positional(adapter, self.technique)


def fp8_encode(x: torch.Tensor, dtype: torch.dtype, scale: torch.Tensor | None) -> torch.Tensor:
    """``x / scale`` (``scale`` per token and head, or a scalar) cast (saturating) to the fp8 ``dtype`` and viewed
    as ``uint8``."""
    limit = torch.finfo(dtype).max
    work = x.float() if scale is None else x.float() / scale
    return work.clamp(-limit, limit).to(dtype).view(torch.uint8)


def fp8_decode(raw: torch.Tensor, dtype: torch.dtype, scale: torch.Tensor | None, out: torch.dtype) -> torch.Tensor:
    """Inverse of :func:`fp8_encode` (in ``out`` precision)."""
    work = raw.view(dtype).float()
    return (work if scale is None else work * scale).to(out)


def token_scale(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Per-token, per-head scale ``[B, H, S, 1]`` mapping each head vector's absmax to the format's maximum."""
    return (x.float().abs().amax(-1, keepdim=True) / torch.finfo(dtype).max).clamp_min(1e-12)


@torch.no_grad()
def calibrate_kv_absmax(adapter: Any, batches: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-layer absolute maxima of the keys and of the values the model writes into its cache on ``batches``.

    The model runs each batch with a ``transformers`` ``DynamicCache`` whose ``update`` records the states it
    receives, so the maxima are those of the cached entries (after the rotary embedding and any key norm). A layer
    that never writes the cache (a layer reading another layer's cache) keeps 0.

    Returns:
        ``(keys, values)``, float32 tensors of shape ``[num_layers]`` on the CPU.
    """
    from transformers import DynamicCache

    keys = torch.zeros(adapter.num_layers)
    values = torch.zeros(adapter.num_layers)

    class Recording(DynamicCache):
        def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args: Any,
                   **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            keys[layer_idx] = torch.maximum(keys[layer_idx], key_states.detach().abs().amax().float().cpu())
            values[layer_idx] = torch.maximum(values[layer_idx], value_states.detach().abs().amax().float().cpu())
            return super().update(key_states, value_states, layer_idx, *args, **kwargs)

    model = adapter.model
    for batch in batches:
        model(input_ids=batch.to(adapter.device), past_key_values=Recording(config=model.config), use_cache=True)
    return keys, values


def static_scales(absmax: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Static scales mapping each layer's absolute maximum to the format's maximum (1 where nothing was seen)."""
    scales = absmax.float() / torch.finfo(dtype).max
    return torch.where(scales > 0, scales, torch.ones_like(scales))


def fp8_round_trip(x: torch.Tensor, format: str = "e4m3", scale: str = "none") -> torch.Tensor:
    """``x`` stored and read back through an fp8 cache (for tests and error studies)."""
    dtype = FP8_FORMATS[format]
    s = token_scale(x, dtype) if scale == "token" else None
    return fp8_decode(fp8_encode(x, dtype, s), dtype, s, x.dtype)


class FP8Cache(CacheWrapper):
    """The cache of one generation stored in fp8 (see the module docstring)."""

    def __init__(
        self,
        inner: KVCache,
        config: FP8KVConfig,
        layers: frozenset[int] | None,
        static: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        super().__init__(inner)
        self.config = config
        self.static = static  # per-layer key and value scales of scale="tensor"
        self.layers = layers
        self.fp8 = FP8_FORMATS[config.format]
        self.dtypes: dict[int, torch.dtype] = {}
        self.key_scales, self.value_scales = SlotStore(), SlotStore()

    def transform_update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.covers(layer_idx):
            return key_states, value_states
        self.dtypes.setdefault(layer_idx, key_states.dtype)
        encoded = []
        for kind, (states, store) in enumerate(((key_states, self.key_scales), (value_states, self.value_scales))):
            scale = self._static(kind, layer_idx, states)
            if self.config.scale == "token":
                scale = token_scale(states, self.fp8)
                full = (self.inner.batch_size, states.shape[1], self.num_slots, 1)
                store.ensure(layer_idx, full, dtype=torch.float32, device=states.device, fill=1.0)
                store.write(layer_idx, self.active_slots, scale, self.active_rows)
            encoded.append(fp8_encode(states, self.fp8, scale))
        return encoded[0], encoded[1]

    def transform_read(
        self, keys: torch.Tensor, values: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.covers(layer_idx):
            return keys, values
        out = self.dtypes[layer_idx]
        if self.config.scale == "token":
            key_scale = self.key_scales.view(layer_idx, self.active_rows)
            value_scale = self.value_scales.view(layer_idx, self.active_rows)
        else:
            key_scale, value_scale = self._static(0, layer_idx, keys), self._static(1, layer_idx, values)
        return fp8_decode(keys, self.fp8, key_scale, out), fp8_decode(values, self.fp8, value_scale, out)

    def _static(self, kind: int, layer_idx: int, like: torch.Tensor) -> torch.Tensor | None:
        """The static scale of layer ``layer_idx``'s keys (``kind=0``) or values (1), on ``like``'s device."""
        if self.static is None:
            return None
        return self.static[kind][layer_idx].to(like.device)

    def pass_begin(self, frontier: list[int | None]) -> None:
        self.account(frontier, self._layer_bytes)

    def _layer_bytes(
        self, layer_idx: int, keys: torch.Tensor, committed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        heads = keys.shape[1]
        stored = self.dense_bytes(keys, committed)  # the fp8 entries
        if self.config.scale == "token":
            stored = stored + 2.0 * committed * heads * 4  # one float32 scale per token, head, K and V
        return stored, self.dense_bytes(keys, committed, self.dtypes[layer_idx])

    def on_gather(self, rows: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> None:
        self.key_scales.gather(rows, src, dst)
        self.value_scales.gather(rows, src, dst)

    def on_reorder(self, rows: torch.Tensor) -> None:
        self.key_scales.reorder(rows)
        self.value_scales.reorder(rows)


@register_technique
class FP8KVTechnique(KVCacheTechnique):
    """KV cache in 8-bit floats (see :class:`FP8KVConfig`). Training-free; no parameters."""

    name: ClassVar[str] = FP8KVConfig.technique
    config_class: ClassVar[type[FP8KVConfig]] = FP8KVConfig
    info = TechniqueInfo(family=Family.KV_QUANTIZATION, saves=Saving.KV_MEMORY, lossless=False)
    order: ClassVar[int] = 11
    config: FP8KVConfig

    def __init__(self, config: FP8KVConfig, adapter: Any) -> None:
        super().__init__(config, adapter)
        if config.scale == "tensor":  # saved with the technique, so a loaded recipe skips the calibration
            self.register_buffer("key_scale", torch.full((adapter.num_layers,), float("nan")))
            self.register_buffer("value_scale", torch.full((adapter.num_layers,), float("nan")))

    @property
    def calibrated(self) -> bool:
        """Whether the static scales of ``scale="tensor"`` are set (always true for the other scales)."""
        if self.config.scale != "tensor":
            return True
        return bool(torch.isfinite(self.key_scale).all() and torch.isfinite(self.value_scale).all())

    def attach(self, model: Any, adapter: Any, context: AttachContext | None = None) -> None:
        super().attach(model, adapter, context)
        if not self.calibrated:
            self.calibrate(token_batches(calibration_source(context, self.name)))

    def calibrate(self, batches: Sequence[torch.Tensor]) -> None:
        """Set the static scales from the absolute maxima of the cached keys and values on ``batches``."""
        keys, values = calibrate_kv_absmax(self.adapter, batches)
        unseen = [i + 1 for i in range(len(keys)) if keys[i] == 0 and values[i] == 0]
        if unseen:
            _log.warning("kv_fp8: layers %s never wrote the cache on the calibration data; their scale stays 1",
                         unseen)
        dtype = FP8_FORMATS[self.config.format]
        self.key_scale.copy_(static_scales(keys, dtype))
        self.value_scale.copy_(static_scales(values, dtype))

    def static_scales(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The per-layer ``(key, value)`` scales of ``scale="tensor"`` (``None`` for the other scales)."""
        if self.config.scale != "tensor":
            return None
        if not self.calibrated:
            raise ConfigError("kv_fp8 with scale='tensor' is not calibrated: attach it with calibration data")
        return self.key_scale.detach().float().cpu(), self.value_scale.detach().float().cpu()

    def make_wrapper(self, cache: KVCache) -> CacheWrapper:
        return FP8Cache(cache, self.config, layer_indices(self.config.layers, self.adapter), self.static_scales())
