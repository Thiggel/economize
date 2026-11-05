"""KIVI: asymmetric 2/4-bit KV-cache quantization, keys per channel and values per token (Liu et al., ICML 2024).

Keys have a few channels of large magnitude, so KIVI quantizes them per channel: every group of ``group_size``
consecutive tokens gets one scale and zero point per channel. Values are quantized per token, in groups of
``group_size`` channels. The most recent tokens stay in full precision: keys are quantized in whole blocks of
``residual_length`` tokens once a block is complete, values once a token is more than ``residual_length``
tokens old. With ``l`` committed tokens, ``floor(l / R) * R`` keys and ``max(0, l - R)`` values are quantized.

economize simulates the quantization in the cache's own dtype (quantize, then dequantize in place), which reproduces
the numerics of the integer kernels without them. The prompt is quantized at the end of its pass, later entries at
the start of each exact pass, and only committed ones (see :mod:`.wrapper`); draft K/V written between exact passes
stay in full precision, so draft rollback and tree compaction need no extra work. The byte accounting counts
``bits / 8`` bytes per quantized element plus an fp16 scale and zero point per group, as the paper's kernels store
them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, overload

import torch

from ...core.config import TechniqueConfig
from ...core.metadata import Family, Saving, Simulation, TechniqueInfo
from ...core.technique import register_technique
from ...errors import ConfigError
from ...runtime import KVCache, current_phase
from ...validation import check_choice, check_layers, check_positive
from .._kv import CacheWrapper, KVCacheTechnique, check_positional, innermost, layer_indices

__all__ = ["KIVI_BITS", "KIVICache", "KIVIConfig", "KIVITechnique", "fake_quantize", "kivi_bytes", "quantize"]

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
    """

    technique: ClassVar[str] = "kivi"
    bits: int = 2
    group_size: int = 32
    residual_length: int = 128
    layers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        check_choice("bits", self.bits, KIVI_BITS)
        check_positive("group_size", self.group_size)
        check_positive("residual_length", self.residual_length)
        if self.residual_length % self.group_size:
            raise ConfigError(
                f"residual_length ({self.residual_length}) must be a multiple of group_size ({self.group_size})"
            )
        check_layers(self.layers)

    def validate(self, adapter: Any) -> None:
        check_layers(self.layers, adapter)
        check_positional(adapter, self.technique)


def fake_quantize(x: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    """Asymmetric min-max quantization of ``x`` along ``dim`` with ``2**bits`` levels, dequantized.

    Scale and zero point are rounded to fp16, the format in which the KIVI kernels store them. A scale or zero point
    beyond the fp16 range (``|K/V|`` above 65504, which bf16 and fp32 caches can hold) stays in fp32 instead of
    overflowing to ``inf``; the byte accounting still counts 2 bytes for it.
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


@overload
def kivi_bytes(tokens: int, keys_quantized: int, values_quantized: int, heads: int, dim: int, element: int,
               config: KIVIConfig) -> float: ...
@overload
def kivi_bytes(tokens: torch.Tensor, keys_quantized: torch.Tensor, values_quantized: torch.Tensor, heads: int,
               dim: int, element: int, config: KIVIConfig) -> torch.Tensor: ...
def kivi_bytes(tokens: Any, keys_quantized: Any, values_quantized: Any, heads: int, dim: int, element: int,
               config: KIVIConfig) -> Any:
    """Bytes of one layer's K and V for ``tokens`` entries of which the first ``keys_quantized`` keys and
    ``values_quantized`` values are quantized (integers, or per-row tensors computed elementwise)."""
    g = config.group_size
    value_groups = dim // g if dim % g == 0 else 1
    per_bit = config.bits / 8
    keys = keys_quantized * heads * dim * per_bit + (keys_quantized // g) * heads * dim * 4
    values = values_quantized * heads * dim * per_bit + values_quantized * heads * value_groups * 4
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

    def pass_begin(self, frontier: list[int | None]) -> None:
        self._quantize_committed(frontier)
        self.account(frontier, self._layer_bytes)
        self._prompt_ends = self._pass_ends() if current_phase() == "prefill" else None

    def pass_end(self) -> None:
        # The prompt is committed once its pass has written it: quantize it now, as KIVI does at prefill, so that
        # its one-time cost lands in the prefill and the first decode pass finds nothing left to quantize.
        if self._prompt_ends is not None:
            ends, self._prompt_ends = self._prompt_ends, None
            self._quantize_committed(ends)

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
        base = innermost(self.inner)
        r = self.config.residual_length
        jobs: dict[tuple[str, int, int], list[int]] = {}
        for b, c in enumerate(committed):
            if c is None:
                continue
            k_to, v_to = (c // r) * r, max(0, c - r)
            if k_to > self.keys_quantized[b]:
                jobs.setdefault(("k", self.keys_quantized[b], k_to), []).append(b)
                self.keys_quantized[b] = k_to
            if v_to > self.values_quantized[b]:
                jobs.setdefault(("v", self.values_quantized[b], v_to), []).append(b)
                self.values_quantized[b] = v_to
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
            grouped = block.reshape(rows, heads, tokens // g, g, dim)
            return fake_quantize(grouped, self.config.bits, dim=3).reshape(block.shape)
        group = g if dim % g == 0 else dim  # per token, groups of g channels
        grouped = block.reshape(rows, heads, tokens, dim // group, group)
        return fake_quantize(grouped, self.config.bits, dim=4).reshape(block.shape)

    def _layer_bytes(
        self, layer_idx: int, keys: torch.Tensor, committed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        heads, dim = keys.shape[1], keys.shape[3]
        quantized_keys = torch.tensor(self.keys_quantized, dtype=torch.float64)
        quantized_values = torch.tensor(self.values_quantized, dtype=torch.float64)
        stored = kivi_bytes(committed, quantized_keys, quantized_values, heads, dim, keys.element_size(), self.config)
        return stored, self.dense_bytes(keys, committed)

    def on_reorder(self, rows: torch.Tensor) -> None:
        kept = rows.tolist()
        self.keys_quantized = [self.keys_quantized[r] for r in kept]
        self.values_quantized = [self.values_quantized[r] for r in kept]

    def on_restart(self) -> None:
        self._prompt_ends = None
        self.keys_quantized = [0] * self.batch_size
        self.values_quantized = [0] * self.batch_size


@register_technique
class KIVITechnique(KVCacheTechnique):
    """KIVI: the KV cache in low-bit codes, keys quantized per channel and values per token.

    See :class:`KIVIConfig`. Training-free; no parameters."""

    name: ClassVar[str] = KIVIConfig.technique
    config_class: ClassVar[type[KIVIConfig]] = KIVIConfig
    info = TechniqueInfo(
        family=Family.KV_QUANTIZATION,
        saves=Saving.KV_MEMORY,
        lossless=False,
        simulated=Simulation(Saving.KV_MEMORY, "writes the dequantized entries back into the dense cache"),
    )
    order: ClassVar[int] = 10
    graph_safe: ClassVar[bool] = True  # quantizes in place at the pass boundaries; layers read the dense buffers
    config: KIVIConfig

    def make_wrapper(self, cache: KVCache) -> CacheWrapper:
        return KIVICache(cache, self.config, layer_indices(self.config.layers, self.adapter))
