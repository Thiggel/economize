"""KIVI: asymmetric 2/4-bit KV-cache quantization, keys per channel and values per token (Liu et al., ICML 2024).

Keys have a few channels of large magnitude, so KIVI quantizes them per channel: every group of ``group_size``
consecutive tokens gets one scale and zero point per channel. Values are quantized per token, in groups of
``group_size`` channels. The most recent tokens stay in full precision: keys are quantized in whole blocks of
``residual_length`` tokens once a block is complete, values once a token is more than ``residual_length``
tokens old. With ``l`` committed tokens, ``floor(l / R) * R`` keys and ``max(0, l - R)`` values are quantized.

The prompt is quantized at the end of its pass, later entries at the start of each exact pass, and only committed
ones (see :mod:`.wrapper`); draft K/V written between exact passes stay in full precision, so draft rollback and
tree compaction need no extra work. The byte accounting counts ``bits / 8`` bytes per quantized element plus an
fp16 scale and zero point per group, as the paper's kernels store them (fp32, 4 bytes each, once a scale or zero point
of the simulated storage exceeded the fp16 range and stayed in fp32). A head dimension that is not a multiple of
``group_size`` quantizes each value token as one group of the whole head dimension (:func:`value_group`, with a
warning); the accounting counts one scale and zero point per value token then.

How the cache holds the entries is a storage of :data:`~.storage.KIVI_STORAGES`. The default ``"packed"`` storage
(:class:`~.packed.PackedKIVICache`) keeps the quantized entries as packed codes with fp16 scales and zero points and
the recent ones in a short full-precision ring, so the memory really shrinks. ``"simulated"`` (:class:`KIVICache`)
quantizes in the cache's own dtype and writes the dequantized values back into the dense buffers: the same numerics
with the full dense memory, graph-safe, and the storage used when another KV technique wraps the same cache.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, overload

import torch
from torch import nn

from ...core.config import TechniqueConfig
from ...core.metadata import Family, Saving, Simulation, TechniqueInfo
from ...core.technique import AttachContext, register_technique
from ...errors import ConfigError
from ...runtime import KVCache, current_phase
from ...validation import check_choice, check_layers, check_positive
from .._kv import CacheWrapper, KVCacheTechnique, check_positional, innermost
from .storage import KIVI_STORAGES

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = [
    "KIVI_BITS", "KIVICache", "KIVIConfig", "KIVITechnique", "fake_quantize", "kivi_bytes", "quantize", "value_group",
]

KIVI_BITS = (2, 4, 8)


@dataclass(frozen=True)
class KIVIConfig(TechniqueConfig):
    """KIVI KV-cache quantization.

    Attributes:
        bits: Bits per quantized element (2 or 4 in the paper; 8 for reference).
        group_size: Tokens per key group (per channel) and channels per value group (per token). The paper uses 32.
        residual_length: Most recent tokens kept in full precision (``R``, a multiple of ``group_size``; 128 in
            the paper). A residual at least as long as the generation quantizes nothing, which is exact.
        layers: 1-indexed layers whose cache is quantized (empty: every layer).
        storage: How the cache stores its entries (:data:`~.storage.KIVI_STORAGES`): ``"packed"`` keeps packed
            codes and a full-precision ring and no dense buffer, so the memory really shrinks; ``"simulated"``
            quantizes in place inside the dense buffers. ``None`` (the default) picks ``"packed"`` and falls back
            to ``"simulated"`` where packed storage cannot run (another KV-cache technique, shared layer caches,
            trainable layer copies); an explicit ``"packed"`` raises there instead.
        headroom: Ring slots of the packed storage beyond ``residual_length``: room for the uncommitted tokens of
            one cycle (draft steps and a verify pass). Passes of at most this many queries read the packed codes
            directly in an attention kernel that supports them.
    """

    technique: ClassVar[str] = "kivi"
    #: Whether the configuration may use another storage than ``"simulated"``.
    packable: ClassVar[bool] = True
    bits: int = 2
    group_size: int = 32
    residual_length: int = 128
    layers: tuple[int, ...] = ()
    storage: str | None = None
    headroom: int = 256

    def __post_init__(self) -> None:
        check_choice("bits", self.bits, KIVI_BITS)
        check_positive("group_size", self.group_size)
        check_positive("residual_length", self.residual_length)
        if self.residual_length % self.group_size:
            raise ConfigError(
                f"residual_length ({self.residual_length}) must be a multiple of group_size ({self.group_size})"
            )
        check_layers(self.layers)
        check_positive("headroom", self.headroom)
        if self.storage is not None:
            check_choice("storage", self.storage, tuple(KIVI_STORAGES.names()))
            if not self.packable and self.storage != "simulated":
                raise ConfigError(f"{self.technique} supports only storage='simulated', got {self.storage!r}")

    @property
    def kivi_storage(self) -> str:
        """The storage in use when nothing blocks it: :attr:`storage`, else ``"packed"`` (``"simulated"`` for a
        configuration that is not :attr:`packable`)."""
        if self.storage is not None:
            return self.storage
        return "packed" if self.packable else "simulated"

    def validate(self, adapter: Any) -> None:
        check_layers(self.layers, adapter)
        check_positional(adapter, self.technique)


def fake_quantize(x: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    """Asymmetric min-max quantization of ``x`` along ``dim`` with ``2**bits`` levels, dequantized.

    Scale and zero point are rounded to fp16, the format in which the KIVI kernels store them. A scale or zero point
    beyond the fp16 range (``|K/V|`` above 65504, which bf16 and fp32 caches can hold) stays in fp32 instead of
    overflowing to ``inf``; :class:`KIVICache` then counts 4 bytes per scale and zero point.
    """
    codes, scale, low = quantize(x, bits, dim)
    return (codes * scale + low).to(x.dtype)


def quantize(x: torch.Tensor, bits: int, dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The integer codes (as fp32), scale and zero point of :func:`fake_quantize` (scale and zero point keep
    ``dim`` with size 1); ``codes * scale + low`` is the dequantized value."""
    work = x.float()
    low = work.amin(dim, keepdim=True)
    high = work.amax(dim, keepdim=True)
    levels = 2**bits - 1
    scale = _fp16_rounded((high - low) / levels)
    low = _fp16_rounded(low)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = ((work - low) / safe).round().clamp(0, levels)
    return codes, scale, low


_FP16_MAX = torch.finfo(torch.float16).max


def _fp16_rounded(x: torch.Tensor) -> torch.Tensor:
    """``x`` (fp32) rounded to fp16 where fp16 represents it, unchanged where it would overflow."""
    return torch.where(x.abs() <= _FP16_MAX, x.half().float(), x)


def value_group(dim: int, group_size: int) -> int:
    """Channels per value group for a head dimension ``dim``: ``group_size``, or the whole ``dim`` (one group per
    token, coarser than the paper's grouping) with a :class:`UserWarning` when ``dim`` is not a multiple of it."""
    if dim % group_size == 0:
        return group_size
    warnings.warn(
        f"KIVI: head dim {dim} is not a multiple of group_size {group_size}; values are quantized with one scale "
        f"and zero point per token over all {dim} channels",
        UserWarning,
        stacklevel=3,
    )
    return dim


@overload
def kivi_bytes(tokens: int, keys_quantized: int, values_quantized: int, heads: int, dim: int, element: int,
               config: KIVIConfig, scale_bytes: int = 2) -> float: ...
@overload
def kivi_bytes(tokens: torch.Tensor, keys_quantized: torch.Tensor, values_quantized: torch.Tensor, heads: int,
               dim: int, element: int, config: KIVIConfig, scale_bytes: int = 2) -> torch.Tensor: ...
def kivi_bytes(tokens: Any, keys_quantized: Any, values_quantized: Any, heads: int, dim: int, element: int,
               config: KIVIConfig, scale_bytes: int = 2) -> Any:
    """Bytes of one layer's K and V for ``tokens`` entries of which the first ``keys_quantized`` keys and
    ``values_quantized`` values are quantized (integers, or per-row tensors computed elementwise), with
    ``scale_bytes`` bytes per scale and per zero point (2 for fp16, 4 for the fp32 fallback)."""
    g = config.group_size
    value_groups = dim // g if dim % g == 0 else 1
    per_bit = config.bits / 8
    pair = 2 * scale_bytes  # one scale and one zero point per group
    keys = keys_quantized * heads * dim * per_bit + (keys_quantized // g) * heads * dim * pair
    values = values_quantized * heads * dim * per_bit + values_quantized * heads * value_groups * pair
    full = ((tokens - keys_quantized) + (tokens - values_quantized)) * heads * dim * element
    return keys + values + full


class KIVICache(CacheWrapper):
    """The cache of one generation under KIVI (see the module docstring)."""

    def __init__(self, inner: KVCache, config: KIVIConfig, layers: frozenset[int] | None) -> None:
        super().__init__(inner)
        self.config = config
        self.layers = layers
        self.keys_quantized = [0] * inner.batch_size  # per row: quantized key prefix (a multiple of R)
        self.values_quantized = [0] * inner.batch_size  # per row: quantized value prefix
        self._prompt_ends: list[int | None] | None = None  # per row, during the prompt pass: its end
        self.scale_bytes = 2  # bytes per scale and zero point: 4 once one stayed in fp32 (beyond the fp16 range)
        self._wide: list[torch.Tensor] = []  # per quantized block of the simulated storage: [] bool, fp32 fallback

    def pass_begin(self, frontier: list[int | None]) -> None:
        self._quantize_committed(frontier)
        self._check_wide()
        self.account(frontier, self._layer_bytes)
        self._prompt_ends = self._pass_ends() if current_phase() == "prefill" else None

    def pass_end(self) -> None:
        # The prompt is committed once its pass has written it: quantize it now, as KIVI does at prefill, so that
        # its one-time cost lands in the prefill and the first decode pass finds nothing left to quantize.
        if self._prompt_ends is not None:
            ends, self._prompt_ends = self._prompt_ends, None
            self._quantize_committed(ends)

    def _check_wide(self) -> None:
        """Switch the accounting to fp32 scales once a quantized block kept one (one host sync per pass at most)."""
        if self._wide:
            device = self._wide[0].device
            if bool(torch.stack([flag.to(device) for flag in self._wide]).any()):
                self.scale_bytes = 4
            self._wide = []

    def _pass_ends(self) -> list[int | None]:
        """One past the last slot every row writes in the current (eager) pass."""
        slots = self.active_slots
        if slots.dim() == 1:
            slots = slots[None].expand(self.inner.active_batch_size, -1)
        real = slots != self.trash_slot
        ends = torch.where(real, slots + 1, torch.zeros_like(slots)).amax(1).tolist()
        return [int(e) or None for e in ends]

    def _quantize_committed(self, committed: list[int | None]) -> None:
        """Quantize what KIVI quantizes once row ``b`` holds ``committed[b]`` committed entries."""
        jobs = self._plan(committed)
        for (kind, _, stop), rows in jobs.items():
            counts = self.keys_quantized if kind == "k" else self.values_quantized
            for b in rows:
                counts[b] = stop
        self._apply_jobs(jobs)

    def _plan(self, committed: list[int | None]) -> dict[tuple[str, int, int], list[int]]:
        """The blocks :meth:`_quantize_committed` quantizes: ``(kind, start, stop)`` (``kind`` ``"k"`` or ``"v"``)
        mapped to the rows that quantize slots ``start:stop``."""
        r = self.config.residual_length
        jobs: dict[tuple[str, int, int], list[int]] = {}
        for b, c in enumerate(committed):
            if c is None:
                continue
            k_to, v_to = (c // r) * r, max(0, c - r)
            if k_to > self.keys_quantized[b]:
                jobs.setdefault(("k", self.keys_quantized[b], k_to), []).append(b)
            if v_to > self.values_quantized[b]:
                jobs.setdefault(("v", self.values_quantized[b], v_to), []).append(b)
        return jobs

    def _apply_jobs(self, jobs: dict[tuple[str, int, int], list[int]]) -> None:
        """Quantize the blocks of :meth:`_plan` in place in the dense buffers of every covered layer."""
        base = innermost(self.inner)
        layers = [i for i in self.positional_buffers() if self.covers(i)]
        for (kind, start, stop), rows in jobs.items():
            for layer_idx in layers:
                keys, values = base.read(layer_idx)
                buffer = keys if kind == "k" else values
                index = torch.tensor(rows, device=buffer.device)
                block = buffer[index, :, start:stop]
                buffer[index, :, start:stop] = self._quantize_block(layer_idx, kind, index, start, block)

    def _quantize_block(
        self, layer_idx: int, kind: str, rows: torch.Tensor, start: int, block: torch.Tensor
    ) -> torch.Tensor:
        """The dequantized ``block`` (``[rows, H, T, D]``, slots ``start:start+T`` of ``rows`` of a layer's key or
        value buffer) that replaces it in place; a subclass may also keep the codes."""
        return self._quantize(block, kind)

    def _quantize(self, block: torch.Tensor, kind: str) -> torch.Tensor:
        rows, heads, tokens, dim = block.shape
        g = self.config.group_size
        if kind == "k":  # per channel, groups of g tokens
            grouped, axis = block.reshape(rows, heads, tokens // g, g, dim), 3
        else:  # per token, groups of g channels
            group = value_group(dim, g)
            grouped, axis = block.reshape(rows, heads, tokens, dim // group, group), 4
        codes, scale, low = quantize(grouped, self.config.bits, dim=axis)
        self._wide.append((scale.abs() > _FP16_MAX).any() | (low.abs() > _FP16_MAX).any())
        return (codes * scale + low).to(block.dtype).reshape(block.shape)

    def _layer_bytes(
        self, layer_idx: int, keys: torch.Tensor, committed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        heads, dim = keys.shape[1], keys.shape[3]
        quantized_keys = torch.tensor(self.keys_quantized, dtype=torch.float64)
        quantized_values = torch.tensor(self.values_quantized, dtype=torch.float64)
        stored = kivi_bytes(
            committed, quantized_keys, quantized_values, heads, dim, keys.element_size(), self.config, self.scale_bytes
        )
        return stored, self.dense_bytes(keys, committed)

    def on_reorder(self, rows: torch.Tensor) -> None:
        kept = rows.tolist()
        self.keys_quantized = [self.keys_quantized[r] for r in kept]
        self.values_quantized = [self.values_quantized[r] for r in kept]

    def on_restart(self) -> None:
        self._prompt_ends = None
        self.scale_bytes = 2
        self._wide = []
        self.keys_quantized = [0] * self.batch_size
        self.values_quantized = [0] * self.batch_size


@register_technique
class KIVITechnique(KVCacheTechnique):
    """KIVI: the KV cache in low-bit codes, keys quantized per channel and values per token.

    See :class:`KIVIConfig`. Training-free; no parameters. :attr:`storage` is the storage in use; under
    ``"packed"`` decoding runs eagerly (no CUDA graphs).
    """

    name: ClassVar[str] = KIVIConfig.technique
    config_class: ClassVar[type[KIVIConfig]] = KIVIConfig
    info = TechniqueInfo(
        family=Family.KV_QUANTIZATION,
        saves=Saving.KV_MEMORY,
        lossless=False,
        simulated=Simulation(Saving.KV_MEMORY, "writes the dequantized entries back into the dense cache"),
    )
    order: ClassVar[int] = 10
    graph_safe: ClassVar[bool] = True  # simulated: quantizes in place at the pass boundaries
    config: KIVIConfig

    def __init__(self, config: KIVIConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)
        self.storage = config.kivi_storage
        self._sync_graph_safe()

    def _sync_graph_safe(self) -> None:
        # The packed storage hands layers fresh tensors on every pass, so it decodes eagerly.
        self.graph_safe = type(self).graph_safe and self.storage == "simulated"  # type: ignore[misc]

    @classmethod
    def simulation_for(cls, config: TechniqueConfig | None = None) -> Simulation | None:
        resolved = config if config is not None else cls.config_class
        if not getattr(resolved, "packable", False):
            return super().simulation_for(config)
        if isinstance(resolved, type):
            resolved = resolved()
        assert isinstance(resolved, KIVIConfig)
        return cls._storage_simulation(resolved.kivi_storage)

    def simulation(self) -> Simulation | None:
        if not self.config.packable:
            return super().simulation()
        return self._storage_simulation(self.storage)

    @classmethod
    def _storage_simulation(cls, storage: str) -> Simulation | None:
        simulation = KIVI_STORAGES.get(storage).simulation
        return simulation.within(cls.info.saves) if simulation is not None else None

    def reads_packed(self) -> bool:
        """Whether the model's attention implementation reads the ``"kivi"`` packed format."""
        from ...attention.registry import accepts_packed_kv

        implementation = str(getattr(self.adapter.config, "_attn_implementation", None) or "eager")
        return accepts_packed_kv(implementation, "kivi")

    def make_wrapper(self, cache: KVCache) -> CacheWrapper:
        storage = self.storage
        if isinstance(cache, CacheWrapper) and storage != "simulated":  # another wrapper owns the buffers
            storage = "simulated"
        return KIVI_STORAGES.get(storage).factory(cache, self)

    def attach(self, model: nn.Module, adapter: ArchAdapter, context: AttachContext | None = None) -> None:
        if self.storage != "simulated":
            reason = self._storage_blocker(adapter, context or AttachContext())
            if reason is not None and self.config.storage is not None:
                raise ConfigError(f"storage={self.storage!r} cannot run here: {reason}; use storage='simulated'")
            if reason is not None:
                self.storage = "simulated"
                self._sync_graph_safe()

    def _storage_blocker(self, adapter: ArchAdapter, context: AttachContext) -> str | None:
        """Why the storage cannot run on this model and recipe (``None`` if it can)."""
        others = [t.name for t in context.techniques if t is not self and isinstance(t, KVCacheTechnique)]
        if others:
            return f"another KV-cache technique ({', '.join(others)}) wraps the same cache"
        if any(adapter.kv_source(index) != index for index in range(1, adapter.num_layers + 1)):
            return "layers share caches"
        if context.layer_copies():
            return "the recipe runs trainable layer copies, which write the cache outside its layers"
        return None
