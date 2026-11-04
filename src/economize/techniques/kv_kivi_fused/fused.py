"""``kivi_fused``: :mod:`economize.techniques.kv_quant.kivi` with packed codes the attention kernel reads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from ...attention.kivi import KIVIPacked, pack_codes
from ...attention.packed import attach_packed
from ...attention.registry import ATTENTION_KERNELS, accepts_packed_kv
from ...core.metadata import Family, Saving, Simulation, TechniqueInfo
from ...core.technique import register_technique
from ...errors import IncompatibleTechniquesError
from ...runtime import KVCache
from .._kv import CacheWrapper, innermost, layer_indices
from ..kv_quant.kivi import _FP16_MAX, KIVICache, KIVIConfig, KIVITechnique, quantize

__all__ = ["FusedKIVICache", "FusedKIVIConfig", "FusedKIVITechnique"]

FORMAT = "kivi"
_TENSORS = ("key_codes", "key_scale", "key_zero", "value_codes", "value_scale", "value_zero")


@dataclass(frozen=True)
class FusedKIVIConfig(KIVIConfig):
    """KIVI KV-cache quantization whose packed codes the attention kernel reads.

    Attributes:
        bits: Bits per quantized element (2, 4 or 8).
        group_size: Tokens per key group (per channel) and channels per value group (per token).
        residual_length: Most recent tokens kept in full precision (a multiple of ``group_size``).
        layers: 1-indexed layers whose cache is quantized (empty: every layer).
    """

    technique: ClassVar[str] = "kivi_fused"

    def validate(self, adapter: Any) -> None:
        super().validate(adapter)
        implementation = str(getattr(adapter.config, "_attn_implementation", None) or "eager")
        if not accepts_packed_kv(implementation, FORMAT):
            fused = [name for name in ATTENTION_KERNELS.names() if accepts_packed_kv(name, FORMAT)]
            raise IncompatibleTechniquesError(
                f"kivi_fused needs an attention kernel that reads KIVI codes, the model runs {implementation!r}; "
                f"load it with attn_implementation in {fused} (after economize.attention.install()) or use kivi"
            )


class _LayerStore:
    """Packed codes, scales and zero points of one layer, shaped like its ``[B, H, slots, D]`` buffers."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor, bits: int, group: int) -> None:
        batch, heads, slots, dim = keys.shape
        dim_v = values.shape[3]
        per_byte = 8 // bits
        self.value_group = group if dim_v % group == 0 else dim_v
        device = keys.device
        self.key_codes = torch.zeros(batch, heads, slots, dim // per_byte, dtype=torch.uint8, device=device)
        self.key_scale = torch.zeros(batch, heads, -(-slots // group), dim, dtype=torch.float16, device=device)
        self.key_zero = torch.zeros_like(self.key_scale)
        self.value_codes = torch.zeros(batch, heads, slots, dim_v // per_byte, dtype=torch.uint8, device=device)
        value_groups = dim_v // self.value_group
        self.value_scale = torch.zeros(batch, heads, slots, value_groups, dtype=torch.float16, device=device)
        self.value_zero = torch.zeros_like(self.value_scale)
        self.bind(keys)

    def bind(self, keys: torch.Tensor) -> None:
        """Remember the key buffer the codes belong to (after an allocation or a reorder of the cache)."""
        self.pointer = keys.data_ptr()
        self.shape = tuple(keys.shape)

    def reorder(self, rows: torch.Tensor, keys: torch.Tensor) -> None:
        for name in _TENSORS:
            setattr(self, name, getattr(self, name).index_select(0, rows.to(self.key_codes.device)))
        self.bind(keys)

    def copy_rows(self, source: _LayerStore, rows: torch.Tensor, keys: torch.Tensor) -> None:
        """Overwrite this store in place with the ``rows`` of ``source`` (this store has ``len(rows)`` rows)."""
        for name in _TENSORS:
            selected = getattr(source, name).index_select(0, rows.to(self.key_codes.device))
            getattr(self, name).copy_(selected)
        self.bind(keys)

    @classmethod
    def selected(cls, source: _LayerStore, rows: torch.Tensor, keys: torch.Tensor) -> _LayerStore:
        """A new store holding the ``rows`` of ``source``."""
        store = cls.__new__(cls)
        store.value_group = source.value_group
        for name in _TENSORS:
            setattr(store, name, getattr(source, name).index_select(0, rows.to(source.key_codes.device)))
        store.bind(keys)
        return store


class FusedKIVICache(KIVICache):
    """A :class:`~economize.techniques.kv_quant.KIVICache` that also keeps the packed codes of every quantized
    entry and attaches them (:func:`economize.attention.attach_packed`) to the key buffer attention receives.

    The attachment is made only for the whole batch (a :meth:`selecting` pass gets copies of the selected rows and
    no attachment) and only on the buffer the codes were made for, so a kernel never reads codes that do not match
    the dense values. A scale or zero point beyond the fp16 range stops the attachment for the rest of the
    generation; attention then reads the dense (equally quantized) buffers.

    Over the static cache of CUDA graphs the wrapper lives as long as the cache and keeps what a recorded kernel
    reads at fixed addresses: one store set and one pair of device counts per batch size (like the cache's own
    buffer sets; the counts once per device of the model), all created at pass boundaries, the counts updated in
    place. It then attaches the codes on every pass, also before anything is quantized (the kernel reads the dense
    entries of rows whose counts are 0), and reports whether it attaches as its graph state.
    """

    def __init__(self, inner: KVCache, config: KIVIConfig, layers: frozenset[int] | None) -> None:
        super().__init__(inner, config, layers)
        self._static = self.wraps_static_cache
        self._sets: dict[int, dict[int, _LayerStore]] = {}  # static cache: store set per batch size
        self._stores: dict[int, _LayerStore] = self._sets.setdefault(inner.batch_size, {})
        self._overflow: list[torch.Tensor] = []
        self._packable = True
        # per batch size and device (the layers of a model split over several GPUs read counts on their own GPU)
        self._counts: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

    def pass_begin(self, frontier: list[int | None]) -> None:
        super().pass_begin(frontier)
        if self._overflow:
            device = self._overflow[0].device  # layers of a model split over several GPUs flag on their own GPU
            if bool(torch.stack([flag.to(device) for flag in self._overflow]).any()):
                self._packable = False
            self._overflow = []
        if self._static and not self._stores:  # every store before a graph records over the cache
            for layer_idx in self.positional_buffers():
                if self.covers(layer_idx):
                    self._store(layer_idx)
        self._sync_counts()

    def own_graph_state(self) -> Any:
        return (self._packable, bool(self._stores))

    def on_restart(self) -> None:
        super().on_restart()
        self._stores = self._sets.setdefault(self.batch_size, {})
        self._overflow = []
        self._packable = True
        self._sync_counts()

    def _quantize_block(
        self, layer_idx: int, kind: str, rows: torch.Tensor, start: int, block: torch.Tensor
    ) -> torch.Tensor:
        store = self._store(layer_idx)
        n, heads, tokens, dim = block.shape
        g, bits = self.config.group_size, self.config.bits
        stop = start + tokens
        if kind == "k":  # per channel, groups of g tokens (start and tokens are multiples of R, hence of g)
            codes, scale, low = quantize(block.reshape(n, heads, tokens // g, g, dim), bits, dim=3)
            dense = (codes * scale + low).reshape(block.shape).to(block.dtype)
            store.key_codes[rows, :, start:stop] = pack_codes(codes.reshape(block.shape), bits)
            store.key_scale[rows, :, start // g : stop // g] = scale.squeeze(3).half()
            store.key_zero[rows, :, start // g : stop // g] = low.squeeze(3).half()
        else:  # per token, groups of channels
            group = store.value_group
            codes, scale, low = quantize(block.reshape(n, heads, tokens, dim // group, group), bits, dim=4)
            dense = (codes * scale + low).reshape(block.shape).to(block.dtype)
            store.value_codes[rows, :, start:stop] = pack_codes(codes.reshape(block.shape), bits)
            store.value_scale[rows, :, start:stop] = scale.squeeze(4).half()
            store.value_zero[rows, :, start:stop] = low.squeeze(4).half()
        self._overflow.append((scale.abs() > _FP16_MAX).any() | (low.abs() > _FP16_MAX).any())
        return dense

    def _store(self, layer_idx: int) -> _LayerStore:
        store = self._stores.get(layer_idx)
        if store is None:
            keys, values = innermost(self.inner).read(layer_idx)
            store = self._stores[layer_idx] = _LayerStore(keys, values, self.config.bits, self.config.group_size)
        return store

    def _sync_counts(self) -> None:
        """Copy the quantized prefixes into the device counts of the current batch size, on every device that has
        them. Over a static cache with stores the counts are created here for the devices of all stores, so that no
        pass inside a graph allocates them."""
        batch = len(self.keys_quantized)
        counts = [pair for (size, _), pair in self._counts.items() if size == batch]
        if counts:
            keys = torch.tensor(self.keys_quantized, dtype=torch.int32)
            values = torch.tensor(self.values_quantized, dtype=torch.int32)
            for kq, vq in counts:
                kq.copy_(keys)
                vq.copy_(values)
        elif self._static and self._stores:
            for device in {store.key_codes.device for store in self._stores.values()}:
                self._device_counts(device)

    def _device_counts(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (len(self.keys_quantized), device)
        counts = self._counts.get(key)
        if counts is None:
            counts = self._counts[key] = (
                torch.tensor(self.keys_quantized, dtype=torch.int32, device=device),
                torch.tensor(self.values_quantized, dtype=torch.int32, device=device),
            )
        return counts

    def transform_read(
        self, keys: torch.Tensor, values: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        store = self._stores.get(layer_idx)
        if (
            store is None
            or not self._packable
            or self.active_rows is not None
            or keys.data_ptr() != store.pointer
            or tuple(keys.shape) != store.shape
            or not (self._static or any(self.keys_quantized) or any(self.values_quantized))
        ):
            return keys, values
        kq, vq = self._device_counts(keys.device)
        packed = KIVIPacked(
            bits=self.config.bits,
            key_group=self.config.group_size,
            value_group=store.value_group,
            key_codes=store.key_codes,
            key_scale=store.key_scale,
            key_zero=store.key_zero,
            value_codes=store.value_codes,
            value_scale=store.value_scale,
            value_zero=store.value_zero,
            keys_quantized=kq,
            values_quantized=vq,
        )
        return attach_packed(keys, packed), values

    def on_reorder(self, rows: torch.Tensor) -> None:
        super().on_reorder(rows)
        base = innermost(self.inner)
        if not self._static:
            for layer_idx, store in self._stores.items():
                store.reorder(rows, base.read(layer_idx)[0])
            self._counts.clear()
            return
        target = self._sets.setdefault(len(rows), {})  # the cache moved to its buffer set of this size
        for layer_idx, store in self._stores.items():
            keys = base.read(layer_idx)[0]
            if layer_idx in target:
                target[layer_idx].copy_rows(store, rows, keys)
            else:
                target[layer_idx] = _LayerStore.selected(store, rows, keys)
        self._stores = target
        self._sync_counts()


@register_technique
class FusedKIVITechnique(KIVITechnique):
    """KIVI quantization of the KV cache, attended over as packed codes (see :class:`FusedKIVIConfig`)."""

    name: ClassVar[str] = FusedKIVIConfig.technique
    packed_format: ClassVar[str | None] = FORMAT
    config_class: ClassVar[type[FusedKIVIConfig]] = FusedKIVIConfig
    info = TechniqueInfo(
        family=Family.KV_QUANTIZATION,
        saves=Saving.KV_MEMORY | Saving.LATENCY,
        lossless=False,
        simulated=Simulation(Saving.KV_MEMORY, "keeps the dense cache buffers next to the codes"),
    )
    graph_safe: ClassVar[bool] = True
    config: FusedKIVIConfig

    def make_wrapper(self, cache: KVCache) -> CacheWrapper:
        return FusedKIVICache(cache, self.config, layer_indices(self.config.layers, self.adapter))
