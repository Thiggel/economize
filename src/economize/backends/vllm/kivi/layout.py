"""KIVI's paged cell layout for vLLM, and its torch reference (no vLLM, no triton).

vLLM stores one *cell* per (page, KV head, token) of every attention layer: ``2 * head_size`` elements of the cache
dtype for dense K/V, or ``state_content_bytes`` bytes when a backend declares them. KIVI keeps the dense K/V of a
token (vLLM's Triton kernels and the full-precision residual read them) and appends what its packed form needs::

    [ K: D elements | V: D elements | value scales: nvg fp16 | value zeros: nvg fp16 |
      key aux: ka fp16 | key codes: D/pb bytes | value codes: D/pb bytes | key flag | value flag ]  (16-byte aligned)

* Values are quantized per token over groups of ``vg`` channels, so a token's cell holds its own ``nvg = D / vg``
  scales and zero points.
* Keys are quantized per channel over groups of ``g`` tokens, so a group of ``g`` tokens shares ``2 * D`` fp16
  values (``D`` scales, then ``D`` zero points). They are spread over the group's cells, ``ka = ceil(2D / g)`` per
  cell: entry ``e`` of the group that starts at token ``s0`` lives in the cell of token ``s0 + e // ka`` at aux slot
  ``e % ka``. vLLM's page size is a multiple of ``g``, so a group never spans two pages.
* Codes are packed along the channel axis like :func:`economize.attention.kivi.pack_codes` (``pb = 8 // bits``
  codes per byte).
* The flags say what the cell holds: 0 = full precision (vLLM's cache write clears them), 1 = quantized and packed,
  2 = quantized with a scale or zero point beyond the fp16 range (the dense entry holds the fake-quantized value, the
  codes are unused). Every quantized cell's dense entry holds the dequantized value too, so vLLM's own kernels and
  the packed kernel attend over the same numbers.

:class:`KIVISettings` is the engine's configuration (``additional_config["economize_kivi"]``), written by
:class:`economize.backends.vllm.serving.kivi.KIVIVllm`. :func:`reference_quantize` and :func:`write_cells` /
:func:`read_cells` are the torch definition the kernels of :mod:`.kernels` are tested against.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from ....attention.kivi import pack_codes
from ....errors import ConfigError
from ....techniques.kv_quant.kivi import quantize
from ..engine_settings import EngineSettings

__all__ = ["CONFIG_KEY", "FLAG_DENSE", "FLAG_OVERFLOW", "FLAG_PACKED", "SETTINGS", "CellLayout", "KIVISettings",
           "cell_layout", "read_cells", "reference_quantize", "settings_of", "write_cells"]

#: The ``additional_config`` key of an engine that serves KIVI.
CONFIG_KEY = "economize_kivi"
FLAG_DENSE, FLAG_PACKED, FLAG_OVERFLOW = 0, 1, 2
_FP16_MAX = float(torch.finfo(torch.float16).max)
#: vLLM's Triton kernels need pages of a multiple of 16 tokens.
_TRITON_BLOCK = 16


@dataclass(frozen=True)
class KIVISettings:
    """KIVI's configuration of one engine.

    Attributes:
        bits: Bits per quantized element (2, 4 or 8).
        group_size: Tokens per key group and channels per value group (``g``).
        residual_length: Most recent tokens kept in full precision (``R``, a multiple of ``g``).
        layers: 1-indexed decoder layers that quantize; empty means all.
        num_layers: Decoder layers of the target model (layers at or beyond it belong to a drafter and stay dense).
        fused: Attend over the packed codes (``kivi_fused``) instead of the dense fake-quantized entries (``kivi``).
    """

    bits: int
    group_size: int
    residual_length: int
    layers: frozenset[int] = frozenset()
    num_layers: int = 0
    fused: bool = False

    def __post_init__(self) -> None:
        if self.bits not in (2, 4, 8):
            raise ConfigError(f"KIVI quantizes to 2, 4 or 8 bits, not {self.bits}")
        g = self.group_size
        if g < 1 or g & (g - 1):
            raise ConfigError(f"economize's vLLM KIVI backend needs a power-of-two group_size, not {g}")
        if self.residual_length < g or self.residual_length % g:
            raise ConfigError(f"residual_length ({self.residual_length}) must be a positive multiple of group_size "
                              f"({g})")

    @property
    def per_byte(self) -> int:
        """Codes per byte."""
        return 8 // self.bits

    @property
    def block_multiple(self) -> int:
        """vLLM page sizes must be a multiple of this (Triton's 16 tokens and whole key groups)."""
        return math.lcm(_TRITON_BLOCK, self.group_size)

    def covers(self, layer0: int) -> bool:
        """Whether the 0-indexed layer ``layer0`` quantizes (a target layer inside :attr:`layers`)."""
        if self.num_layers and layer0 >= self.num_layers:
            return False
        return not self.layers or layer0 + 1 in self.layers

    def as_config(self) -> dict[str, Any]:
        """The ``additional_config`` entry (JSON-serializable)."""
        return {"bits": self.bits, "group_size": self.group_size, "residual_length": self.residual_length,
                "layers": sorted(self.layers), "num_layers": self.num_layers, "fused": self.fused}


def _parse(entry: Mapping[str, Any], vllm_config: Any) -> KIVISettings:
    return KIVISettings(bits=int(entry["bits"]), group_size=int(entry["group_size"]),
                        residual_length=int(entry["residual_length"]),
                        layers=frozenset(int(layer) for layer in entry.get("layers") or ()),
                        num_layers=int(entry.get("num_layers") or 0), fused=bool(entry.get("fused")))


#: The KIVI settings of the engine (see :class:`~economize.backends.vllm.engine_settings.EngineSettings`).
SETTINGS: EngineSettings[KIVISettings] = EngineSettings(CONFIG_KEY, _parse)
#: The KIVI settings ``vllm_config`` (or a plain ``additional_config`` mapping) asks for, or ``None``.
settings_of = SETTINGS.of


@dataclass(frozen=True)
class CellLayout:
    """Byte offsets inside one cell (see the module docstring).

    Attributes:
        head_size: ``D`` (keys and values).
        element_size: Bytes per element of the cache dtype.
        bits: Bits per code.
        group_size: ``g``.
        value_group: Channels per value scale (``g`` when it divides ``D``, else ``D``).
        key_aux: fp16 slots of key scales and zero points per cell (``ka``).
    """

    head_size: int
    element_size: int
    bits: int
    group_size: int
    value_group: int
    key_aux: int

    @property
    def value_groups(self) -> int:
        return self.head_size // self.value_group

    @property
    def code_bytes(self) -> int:
        return self.head_size // (8 // self.bits)

    @property
    def off_v(self) -> int:
        return self.head_size * self.element_size

    @property
    def off_vs(self) -> int:
        return 2 * self.head_size * self.element_size

    @property
    def off_vz(self) -> int:
        return self.off_vs + 2 * self.value_groups

    @property
    def off_ka(self) -> int:
        return self.off_vz + 2 * self.value_groups

    @property
    def off_kc(self) -> int:
        return self.off_ka + 2 * self.key_aux

    @property
    def off_vc(self) -> int:
        return self.off_kc + self.code_bytes

    @property
    def off_flag(self) -> int:
        return self.off_vc + self.code_bytes

    @property
    def size(self) -> int:
        """Bytes per cell (16-byte aligned, hence a whole number of elements of any cache dtype)."""
        return -(-(self.off_flag + 2) // 16) * 16

    @property
    def dense_size(self) -> int:
        """Bytes of the cell without KIVI (vLLM's ``2 * head_size`` elements)."""
        return self.off_vs


def cell_layout(head_size: int, dtype: torch.dtype, settings: KIVISettings) -> CellLayout:
    """The cell of a layer with ``head_size`` channels in ``dtype`` under ``settings``.

    Raises:
        ConfigError: The head size does not fit KIVI's packing.
    """
    g, pb = settings.group_size, settings.per_byte
    if head_size < 16 or head_size & (head_size - 1):
        raise ConfigError(f"economize's vLLM KIVI backend needs a power-of-two head size of at least 16, not "
                          f"{head_size}")
    if head_size % pb:
        raise ConfigError(f"{head_size} channels do not pack into whole bytes at {settings.bits} bits")
    value_group = g if head_size % g == 0 else head_size
    key_aux = -(-2 * head_size // g)
    return CellLayout(head_size, torch.empty((), dtype=dtype).element_size(), settings.bits, g, value_group, key_aux)


def reference_quantize(x: torch.Tensor, bits: int, dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Codes (int32), scale and zero point (fp32, ``dim`` kept with size 1) of KIVI's min-max quantization along
    ``dim``: :func:`economize.techniques.kv_quant.kivi.quantize` with integer codes."""
    codes, scale, low = quantize(x, bits, dim)
    return codes.to(torch.int32), scale, low


def write_cells(keys: torch.Tensor, values: torch.Tensor, keys_quantized: int, values_quantized: int,
                layout: CellLayout) -> torch.Tensor:
    """The cells (``uint8`` ``[H, S, layout.size]``) of one sequence whose first ``keys_quantized`` keys and
    ``values_quantized`` values are quantized, from its full-precision ``keys`` / ``values`` ``[H, S, D]``.

    The definition the quantization kernel implements; ``keys_quantized`` must be a multiple of the group size."""
    heads, tokens, dim = keys.shape
    g, vg, bits = layout.group_size, layout.value_group, layout.bits
    if keys_quantized % g:
        raise ConfigError(f"{keys_quantized} quantized keys are not whole groups of {g}")
    cells = torch.zeros(heads, tokens, layout.size, dtype=torch.uint8, device=keys.device)
    dense_k, dense_v = keys.clone(), values.clone()
    if keys_quantized:
        block = keys[:, :keys_quantized].reshape(heads, keys_quantized // g, g, dim)
        codes, scale, low = reference_quantize(block, bits, dim=2)
        dense_k[:, :keys_quantized] = (codes.float() * scale + low).to(keys.dtype).reshape(heads, -1, dim)
        cells[:, :keys_quantized, layout.off_kc:layout.off_vc] = pack_codes(codes.reshape(heads, -1, dim), bits)
        over = (scale.abs() > _FP16_MAX) | (low.abs() > _FP16_MAX)  # [H, G, 1, D]
        flags = torch.where(over.any(-1, keepdim=True), FLAG_OVERFLOW, FLAG_PACKED).expand(heads, -1, g, 1)
        cells[:, :keys_quantized, layout.off_flag] = flags.reshape(heads, -1).to(torch.uint8)
        aux = torch.cat([scale.squeeze(2), low.squeeze(2)], dim=-1).half()  # [H, G, 2D]
        aux = torch.nn.functional.pad(aux, (0, g * layout.key_aux - 2 * dim)).reshape(heads, -1, layout.key_aux)
        cells[:, :keys_quantized, layout.off_ka:layout.off_kc] = aux.contiguous().view(torch.uint8)
    if values_quantized:
        block = values[:, :values_quantized].reshape(heads, values_quantized, dim // vg, vg)
        codes, scale, low = reference_quantize(block, bits, dim=3)
        dense_v[:, :values_quantized] = (codes.float() * scale + low).to(values.dtype).reshape(heads, -1, dim)
        cells[:, :values_quantized, layout.off_vc:layout.off_flag] = pack_codes(codes.reshape(heads, -1, dim), bits)
        over = ((scale.abs() > _FP16_MAX) | (low.abs() > _FP16_MAX)).reshape(heads, values_quantized, -1)
        cells[:, :values_quantized, layout.off_flag + 1] = torch.where(over.any(-1), FLAG_OVERFLOW,
                                                                       FLAG_PACKED).to(torch.uint8)
        cells[:, :values_quantized, layout.off_vs:layout.off_vz] = scale.reshape(heads, -1, dim // vg).half() \
            .contiguous().view(torch.uint8)
        cells[:, :values_quantized, layout.off_vz:layout.off_ka] = low.reshape(heads, -1, dim // vg).half() \
            .contiguous().view(torch.uint8)
    cells[:, :, :layout.off_v] = dense_k.contiguous().view(torch.uint8)
    cells[:, :, layout.off_v:layout.off_vs] = dense_v.contiguous().view(torch.uint8)
    return cells


def read_cells(cells: torch.Tensor, layout: CellLayout, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """The keys and values ``[H, S, D]`` the cells hold: dequantized codes where a flag is 1, dense entries
    elsewhere (equal values, see the module docstring)."""
    heads, tokens, _ = cells.shape
    dim, g, vg, bits = layout.head_size, layout.group_size, layout.value_group, layout.bits
    keys = cells[:, :, :layout.off_v].contiguous().view(dtype).clone()
    values = cells[:, :, layout.off_v:layout.off_vs].contiguous().view(dtype).clone()
    shifts = torch.arange(8 // bits, dtype=torch.int32, device=cells.device) * bits

    def unpack(raw: torch.Tensor) -> torch.Tensor:
        return ((raw.to(torch.int32)[..., None] >> shifts) & (2**bits - 1)).reshape(*raw.shape[:-1], dim).float()

    k_packed = cells[:, :, layout.off_flag] == FLAG_PACKED
    groups = tokens // g
    if groups:
        aux = cells[:, :groups * g, layout.off_ka:layout.off_kc].contiguous().view(torch.float16)
        aux = aux.reshape(heads, groups, g * layout.key_aux)[..., :2 * dim].float()
        scale = aux[..., :dim].repeat_interleave(g, dim=1)
        zero = aux[..., dim:].repeat_interleave(g, dim=1)
        codes = unpack(cells[:, :groups * g, layout.off_kc:layout.off_vc])
        deq = (codes * scale + zero).to(dtype)
        keys[:, :groups * g] = torch.where(k_packed[:, :groups * g, None], deq, keys[:, :groups * g])
    v_packed = cells[:, :, layout.off_flag + 1] == FLAG_PACKED
    vscale = cells[:, :, layout.off_vs:layout.off_vz].contiguous().view(torch.float16).float()
    vzero = cells[:, :, layout.off_vz:layout.off_ka].contiguous().view(torch.float16).float()
    codes = unpack(cells[:, :, layout.off_vc:layout.off_flag])
    deq = (codes * vscale.repeat_interleave(vg, dim=-1) + vzero.repeat_interleave(vg, dim=-1)).to(dtype)
    values = torch.where(v_packed[..., None], deq, values)
    return keys, values
