"""``kv_evict_fused``: one-shot prompt eviction whose kept entries are stored compacted for the attention kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from ...attention.gathered import GatheredKV
from ...attention.packed import attach_packed, packed_of
from ...attention.registry import ATTENTION_KERNELS, accepts_packed_kv
from ...core.metadata import Exclusion, Family, Saving, Simulation, TechniqueInfo
from ...core.technique import register_technique
from ...errors import ConfigError, IncompatibleTechniquesError
from ...runtime import KVCache, current_phase
from .._kv import CacheWrapper, SlotStore, innermost
from ..kv_evict.policies import EVICTION_METHODS
from ..kv_evict.technique import EvictionCache, KVEvictConfig, KVEvictTechnique

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["FusedEvictionCache", "FusedKVEvictConfig", "FusedKVEvictTechnique"]

FORMAT = "gathered"


@dataclass(frozen=True)
class FusedKVEvictConfig(KVEvictConfig):
    """KV eviction at prefill (``"snapkv"``, ``"pyramidkv"`` or ``"h2o_prefill"``) whose kept prompt entries
    attention reads compacted.

    The fields are those of :class:`~economize.techniques.kv_evict.KVEvictConfig`; ``method`` must name a method
    that selects once (:attr:`~economize.techniques.kv_evict.EvictionMethod.selects_once`).
    """

    technique: ClassVar[str] = "kv_evict_fused"
    method: str = "snapkv"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not EVICTION_METHODS.get(self.method).selects_once:
            once = [name for name in EVICTION_METHODS.names() if EVICTION_METHODS.get(name).selects_once]
            raise ConfigError(
                f"kv_evict_fused compacts a selection made once after the prompt, method {self.method!r} keeps "
                f"evicting while decoding; use one of {once} or kv_evict"
            )

    def validate(self, adapter: ArchAdapter) -> None:
        super().validate(adapter)
        implementation = str(getattr(adapter.config, "_attn_implementation", None) or "eager")
        if not accepts_packed_kv(implementation, FORMAT):
            fused = [name for name in ATTENTION_KERNELS.names() if accepts_packed_kv(name, FORMAT)]
            raise IncompatibleTechniquesError(
                f"kv_evict_fused needs an attention kernel that reads a compacted cache, the model runs "
                f"{implementation!r}; load it with attn_implementation in {fused} (after "
                f"economize.attention.install()) or use kv_evict"
            )


_CAPACITY_STEP = 128  # static compact buffers grow in steps of this many slots


def _kept(keys: torch.Tensor, keep: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
    """``[B, H, slots]`` kept prompt entries (below ``start``, ``[B]``)."""
    slots = min(keep.shape[-1], keys.shape[2])
    position = torch.arange(slots, device=keys.device)
    return keep[..., :slots].to(keys.device) & (position < start[:, None, None])


def _gather(buffer: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return torch.gather(buffer, 2, index[..., None].expand(-1, -1, -1, buffer.shape[3]))


class _Compact:
    """The kept prompt entries of one layer, gathered once after the prompt pass.

    An eager cache builds one sized to its selection (:meth:`gathered`). Over the static cache of CUDA graphs the
    buffers have a fixed ``capacity`` and are refilled in place after every prompt pass (:meth:`buffers`,
    :meth:`fill`); their kernel input then names no shortening of the key axis (``dropped = 0``), since the kernel
    skips the blocks beyond each row's kept entries on the device.
    """

    def __init__(self, key: torch.Tensor, value: torch.Tensor, index: torch.Tensor, count: torch.Tensor,
                 start: torch.Tensor, keys: torch.Tensor, static: bool) -> None:
        self.key, self.value, self.index, self.count, self.start = key, value, index, count, start
        self.static = static
        self.bind(keys)

    @classmethod
    def gathered(cls, keys: torch.Tensor, values: torch.Tensor, keep: torch.Tensor, start: torch.Tensor) -> _Compact:
        kept = _kept(keys, keep, start)
        count = kept.sum(-1, dtype=torch.int32)
        capacity = max(1, int(count.max()))
        index = (~kept).to(torch.int8).argsort(dim=-1, stable=True)[..., :capacity]
        return cls(_gather(keys, index).contiguous(), _gather(values, index).contiguous(),
                   index.to(torch.int32).contiguous(), count.contiguous(), start.to(torch.int32).contiguous(),
                   keys, static=False)

    @classmethod
    def buffers(cls, keys: torch.Tensor, values: torch.Tensor, capacity: int) -> _Compact:
        batch, heads = keys.shape[:2]
        device = keys.device
        return cls(keys.new_zeros(batch, heads, capacity, keys.shape[3]),
                   values.new_zeros(batch, heads, capacity, values.shape[3]),
                   torch.zeros(batch, heads, capacity, dtype=torch.int32, device=device),
                   torch.zeros(batch, heads, dtype=torch.int32, device=device),
                   torch.zeros(batch, dtype=torch.int32, device=device), keys, static=True)

    @property
    def capacity(self) -> int:
        return self.key.shape[2]

    def fill(self, keys: torch.Tensor, values: torch.Tensor, kept: torch.Tensor, start: torch.Tensor) -> None:
        """Overwrite the buffers in place with the ``kept`` entries (see :func:`_kept`) of ``keys`` and ``values``;
        no row and head may keep more than :attr:`capacity`."""
        order = (~kept).to(torch.int8).argsort(dim=-1, stable=True)
        index = order[..., : self.capacity]
        if index.shape[-1] < self.capacity:  # fewer slots than the capacity: pad with the last slot (never read)
            index = torch.cat([index, index[..., -1:].expand(-1, -1, self.capacity - index.shape[-1])], -1)
        self.index.copy_(index)
        self.key.copy_(_gather(keys, index))
        self.value.copy_(_gather(values, index))
        self.count.copy_(kept.sum(-1, dtype=torch.int32))
        self.start.copy_(start)
        self.bind(keys)

    @property
    def dropped(self) -> int:
        """Prompt slots every row and head skips (the shortening of the kernel's key axis; 0 for static buffers)."""
        return 0 if self.static else int((self.start[:, None] - self.count).min())

    @property
    def saves(self) -> bool:
        """Whether some row and head skips a prompt slot (rows with nothing evicted read their prompt compacted)."""
        return bool((self.start[:, None] > self.count).any())

    def bind(self, keys: torch.Tensor) -> None:
        self.pointer = keys.data_ptr()
        self.shape = tuple(keys.shape)
        self.packed = GatheredKV(self.key, self.value, self.index, self.count, self.start, self.dropped)

    def reorder(self, rows: torch.Tensor, keys: torch.Tensor) -> None:
        rows = rows.to(self.key.device)
        for name in _BUFFERS:
            setattr(self, name, getattr(self, name).index_select(0, rows).contiguous())
        self.bind(keys)

    def copy_rows(self, source: _Compact, rows: torch.Tensor, keys: torch.Tensor) -> None:
        """Overwrite these static buffers in place with the ``rows`` of ``source`` (``len(rows)`` rows)."""
        rows = rows.to(self.key.device)
        for name in _BUFFERS:
            getattr(self, name).copy_(getattr(source, name).index_select(0, rows))
        self.bind(keys)

    @classmethod
    def selected(cls, source: _Compact, rows: torch.Tensor, keys: torch.Tensor) -> _Compact:
        """New static buffers holding the ``rows`` of ``source``."""
        rows = rows.to(source.key.device)
        key, value, index, count, start = (getattr(source, n).index_select(0, rows).contiguous() for n in _BUFFERS)
        return cls(key, value, index, count, start, keys, static=True)


_BUFFERS = ("key", "value", "index", "count", "start")


class FusedEvictionCache(EvictionCache):
    """An :class:`~economize.techniques.kv_evict.EvictionCache` that, after the prompt pass evicted, copies the
    kept prompt entries of every layer into compact buffers and attaches them
    (:func:`economize.attention.attach_packed`) to the key buffer attention receives.

    The dense buffers and the keep mask are left as they are, so every fallback (``sdpa`` for long queries) reads
    the masked dense cache, and the kernel still applies the mask to every entry it reads. Attachments are made only
    for the whole batch and only on the buffer the copies were made from. A pass that writes below a row's prompt
    end (a new prompt) drops the copies.

    Over the static cache of CUDA graphs the wrapper lives as long as the cache and keeps what a recorded kernel
    reads at fixed addresses: one set of keep masks and one set of compact buffers per batch size (like the cache's
    own buffer sets), refilled in place. Every prompt pass then fills the compact buffers, also when it evicted
    nothing, and attention always applies the keep masks, so the recorded decoding passes do not depend on the
    prompt. The graph state is the compact capacity (it grows in steps of 128 slots when a selection outgrows it)
    and whether a pass wrote below a prompt end, which stops the attachment until the next generation.
    """

    def __init__(self, inner: KVCache, owner: KVEvictTechnique) -> None:
        super().__init__(inner, owner)
        self._static = self.wraps_static_cache
        self._compact: dict[int, _Compact] = {}
        self._prompt_end: list[int] = []
        self._keep_sets: dict[int, SlotStore] = {inner.batch_size: self.keep}  # static cache: per batch size
        self._compact_sets: dict[int, dict[int, _Compact]] = {}  # static cache: per batch size
        self._capacity = 0
        self._filled = False  # static cache: the compact buffers hold this generation's prompt
        self._stale = False  # static cache: a pass wrote below a prompt end

    @property
    def masks_attention(self) -> bool:
        return self.evicted or self._static

    def own_graph_state(self) -> Any:
        return (self._capacity, self._stale) if self._static else None

    def pass_begin(self, frontier: list[int | None]) -> None:
        prefill = current_phase() == "prefill"
        if (self._compact if not self._static else self._filled) and (
            prefill
            or any(f is not None and 0 <= f < end for f, end in zip(frontier, self._prompt_end, strict=False))
        ):
            if self._static:
                self._filled = False
                self._stale = self._stale or not prefill
            else:
                self._compact = {}
        super().pass_begin(frontier)

    def pass_end(self) -> None:
        info = self._info
        super().pass_end()
        if info is None or not info.prefill:
            return
        end = info.end.clamp(min=0)
        if self._static:
            self._prompt_end = end.tolist()
            self._fill(end)
            return
        if not self.evicted:
            return
        base = innermost(self.inner)
        self._prompt_end = end.tolist()
        for layer_idx, keep in self.keep.tensors.items():
            keys, values = base.read(layer_idx)
            compact = _Compact.gathered(keys, values, keep, end.to(keys.device))
            if compact.saves:
                self._compact[layer_idx] = compact

    def _fill(self, end: torch.Tensor) -> None:
        """Refill the static compact buffers of the current batch size after a prompt pass."""
        if self._stale or not self.keep.tensors:
            return
        base = innermost(self.inner)
        selections = {}
        for layer_idx, keep in self.keep.tensors.items():
            keys, values = base.read(layer_idx)
            start = end.to(device=keys.device, dtype=torch.int32)
            selections[layer_idx] = (keys, values, _kept(keys, keep, start), start)
        need = max(int(kept.sum(-1).max()) for _, _, kept, _ in selections.values())
        if need > self._capacity:  # grow every set (a new graph state: the graphs are recorded again)
            slots = min(kept.shape[-1] for _, _, kept, _ in selections.values())
            self._capacity = max(1, min(slots, -(-need // _CAPACITY_STEP) * _CAPACITY_STEP))
            self._compact_sets = {}
        compacts = self._compact = self._compact_sets.setdefault(self.batch_size, {})
        for layer_idx, (keys, values, kept, start) in selections.items():
            compact = compacts.get(layer_idx)
            if compact is None:
                compact = compacts[layer_idx] = _Compact.buffers(keys, values, self._capacity)
            compact.fill(keys, values, kept, start)
        self._filled = True

    def transform_read(
        self, keys: torch.Tensor, values: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compact = self._compact.get(layer_idx) if self._filled or not self._static else None
        if (
            compact is None
            or self.active_rows is not None
            or keys.data_ptr() != compact.pointer
            or tuple(keys.shape) != compact.shape
            or packed_of(keys) is not None
        ):
            return keys, values
        return attach_packed(keys, compact.packed), values

    def on_restart(self) -> None:
        super().on_restart()
        self.keep = self._keep_sets.setdefault(self.batch_size, SlotStore())
        for keep in self.keep.tensors.values():
            keep.fill_(True)
        self._compact = self._compact_sets.get(self.batch_size, {})
        self._prompt_end = []
        self._filled = self._stale = False

    def _reorder_keep(self, rows: torch.Tensor) -> None:
        if not self._static:
            super()._reorder_keep(rows)
            return
        target = self._keep_sets.setdefault(len(rows), SlotStore())  # the cache moved to its buffer set of this size
        for layer_idx, keep in self.keep.tensors.items():
            selected = keep.index_select(0, rows.to(keep.device))
            existing = target.tensors.get(layer_idx)
            if existing is not None and existing.shape == selected.shape:
                existing.copy_(selected)
            else:
                target.tensors[layer_idx] = selected
        self.keep = target

    def on_reorder(self, rows: torch.Tensor) -> None:
        super().on_reorder(rows)
        base = innermost(self.inner)
        self._prompt_end = [self._prompt_end[r] for r in rows.tolist()] if self._prompt_end else []
        if not self._static:
            for layer_idx, compact in self._compact.items():
                compact.reorder(rows, base.read(layer_idx)[0])
            return
        if not self._compact:
            return
        target = self._compact_sets.setdefault(len(rows), {})
        for layer_idx, compact in self._compact.items():
            keys = base.read(layer_idx)[0]
            existing = target.get(layer_idx)  # the same set (and object) when the batch size stays
            if existing is not None and existing.capacity == compact.capacity:
                existing.copy_rows(compact, rows, keys)
            else:
                target[layer_idx] = _Compact.selected(compact, rows, keys)
        self._compact = target


@register_technique
class FusedKVEvictTechnique(KVEvictTechnique):
    """One-shot KV eviction attended over as a compacted cache (see :class:`FusedKVEvictConfig`)."""

    name: ClassVar[str] = FusedKVEvictConfig.technique
    packed_format: ClassVar[str | None] = FORMAT
    config_class: ClassVar[type[FusedKVEvictConfig]] = FusedKVEvictConfig
    info = TechniqueInfo(
        family=Family.KV_EVICTION,
        saves=Saving.KV_MEMORY | Saving.LATENCY,
        lossless=False,
        excludes=(
            Exclusion(
                Family.KV_QUANTIZATION,
                "the quantizer rewrites the dense prompt entries, which the compact copies miss; use kv_evict with it",
            ),
        ),
        simulated=Simulation(Saving.KV_MEMORY, "compacts the kept entries but keeps the dense buffers"),
    )
    graph_safe: ClassVar[bool] = True  # see FusedEvictionCache: fixed buffers per batch size over a static cache
    config: FusedKVEvictConfig

    def make_wrapper(self, cache: KVCache) -> CacheWrapper:
        return FusedEvictionCache(cache, self)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "compacted": True}
