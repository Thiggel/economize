"""Quantization grids (number formats) and the group-wise quantized weight they produce.

A :class:`QuantGrid` maps a block of weights to codes and back. Its parameters are fitted per row of the block, so
calling it on ``W[:, g*G:(g+1)*G]`` gives group-wise quantization with group size ``G`` along the input dimension,
the layout of GPTQ, AWQ and every int4 kernel. Formats are registered in :data:`QUANT_FORMATS`; a new format is one
registered factory ``(symmetric: bool) -> QuantGrid``.

* :class:`IntGrid`: uniform ``b``-bit integers with the min-max range of the GPTQ reference
  (https://github.com/IST-DASLab/gptq, ``quant.py``): the range always contains zero, asymmetric grids use an integer
  zero point ``round(-min / scale)``, symmetric grids put zero at ``(2^b) / 2``. Round to nearest is off by at most
  ``scale / 2`` for every weight of the fitted block.
* :class:`Fp8Grid`: OCP float8 e4m3 with one absmax scale ``amax / 448`` per row and group (the weight-only fp8 of
  vLLM and torchao). The relative error of a normal value is at most ``2^-4``.
* The 4-bit floating-point microscaling formats ``nvfp4`` and ``mxfp4`` (:mod:`.fp4`).

A format may fix its group size (:meth:`QuantGrid.check_group_size`) and may have parameters shared by a whole
tensor (NVFP4's global scale), which :meth:`QuantGrid.bind` fits; :func:`bind_shards` fits them once per map that
inference engines fuse (:data:`FUSED_SHARDS`), as vLLM requires one global scale per fused ``qkv_proj``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ...errors import ConfigError
from ...registry import Registry

__all__ = [
    "FP8_E4M3_MAX",
    "FUSED_SHARDS",
    "QUANT_FORMATS",
    "Fp8Grid",
    "IntGrid",
    "QParams",
    "QuantGrid",
    "QuantizedWeight",
    "bind_shards",
    "make_grid",
    "quantize_rtn",
]

FP8_E4M3_MAX = 448.0


@dataclass
class QParams:
    """Per-row parameters of one block: ``scale [rows, 1]`` and, for affine grids, ``zero [rows, 1]``."""

    scale: Tensor
    zero: Tensor | None = None


class QuantGrid(ABC):
    """A number format: fit per-row parameters, encode to codes, decode back to floats."""

    #: Registered format name (``"int4"``, ``"fp8"``).
    name: str
    #: Storage bits per weight, without scales.
    bits: int
    symmetric: bool

    @property
    @abstractmethod
    def code_dtype(self) -> torch.dtype:
        """Storage dtype of the codes."""

    @abstractmethod
    def fit(self, w: Tensor) -> QParams:
        """Parameters for every row of ``w [rows, cols]`` (float32)."""

    @abstractmethod
    def encode(self, w: Tensor, params: QParams) -> Tensor:
        """Codes of ``w [rows, cols]`` (``uint8`` integers or ``float8_e4m3fn`` values)."""

    @abstractmethod
    def decode(self, codes: Tensor, params: QParams) -> Tensor:
        """Float32 values of ``codes``."""

    def fake(self, w: Tensor, params: QParams | None = None) -> Tensor:
        """Round ``w`` to the grid (quantize and dequantize) in float32."""
        params = self.fit(w) if params is None else params
        return self.decode(self.encode(w, params), params)

    @property
    def overhead_bits(self) -> int:
        """Bits stored per group besides the codes (an fp16 scale, plus an fp16 zero for affine grids)."""
        return 16 if self.symmetric else 32

    def check_group_size(self, group_size: int | None) -> None:
        """Raise if the format cannot be stored with ``group_size`` (default: any group size works).

        Args:
            group_size: Input columns per scale, ``None`` for one scale per output channel.

        Raises:
            ConfigError: The format fixes another group size.
        """

    @property
    def requires_binding(self) -> bool:
        """Whether the grid has tensor-level parameters that :meth:`bind` has not fitted yet."""
        return False

    def bind(self, weights: Sequence[Tensor]) -> QuantGrid:
        """The grid with its tensor-level parameters fitted to ``weights`` (the shards of one fused map share them).

        Args:
            weights: ``[out_i, in]`` weights quantized with the returned grid.

        Returns:
            The bound grid (``self`` for formats without tensor-level parameters).
        """
        return self


@dataclass(frozen=True)
class IntGrid(QuantGrid):
    """Uniform ``bits``-bit integer grid (codes ``0 .. 2^bits - 1``), min-max fitted as in the GPTQ reference."""

    bits: int
    symmetric: bool = False

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 8:
            raise ConfigError(f"integer grids support 2 to 8 bits, got {self.bits}")

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"int{self.bits}"

    @property
    def maxq(self) -> int:
        return 2**self.bits - 1

    @property
    def code_dtype(self) -> torch.dtype:
        return torch.uint8

    def fit(self, w: Tensor) -> QParams:
        w = w.float()
        zero_ = torch.zeros(w.shape[0], device=w.device)
        xmin = torch.minimum(w.min(1).values, zero_)
        xmax = torch.maximum(w.max(1).values, zero_)
        if self.symmetric:
            # The reference sets xmin = -xmax only for rows with a negative entry; a row of nonnegative weights would
            # then clip at half its maximum. Every row is made symmetric here.
            xmax = torch.maximum(xmin.abs(), xmax)
            xmin = -xmax
        degenerate = (xmin == 0) & (xmax == 0)
        xmin = torch.where(degenerate, -torch.ones_like(xmin), xmin)
        xmax = torch.where(degenerate, torch.ones_like(xmax), xmax)
        scale = (xmax - xmin) / self.maxq
        zero = torch.full_like(scale, (self.maxq + 1) / 2) if self.symmetric else torch.round(-xmin / scale)
        return QParams(scale.unsqueeze(1), zero.unsqueeze(1))

    def encode(self, w: Tensor, params: QParams) -> Tensor:
        assert params.zero is not None
        q = torch.clamp(torch.round(w.float() / params.scale) + params.zero, 0, self.maxq)
        return q.to(torch.uint8)

    def decode(self, codes: Tensor, params: QParams) -> Tensor:
        assert params.zero is not None
        return params.scale * (codes.float() - params.zero)


@dataclass(frozen=True)
class Fp8Grid(QuantGrid):
    """float8 e4m3 (max 448) with an absmax scale per row: ``codes = fp8(w / scale)``, ``scale = amax / 448``."""

    symmetric: bool = True
    bits: int = 8
    name: str = "fp8"

    @property
    def code_dtype(self) -> torch.dtype:
        return torch.float8_e4m3fn

    def fit(self, w: Tensor) -> QParams:
        amax = w.float().abs().amax(1, keepdim=True)
        return QParams(torch.where(amax > 0, amax / FP8_E4M3_MAX, torch.ones_like(amax)))

    def encode(self, w: Tensor, params: QParams) -> Tensor:
        return (w.float() / params.scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)

    def decode(self, codes: Tensor, params: QParams) -> Tensor:
        return codes.float() * params.scale

    @property
    def overhead_bits(self) -> int:
        return 32  # fp32 scale, as in torchao and vLLM


GridFactory = Callable[[bool], QuantGrid]

#: Registered number formats: name -> ``factory(symmetric) -> QuantGrid``.
QUANT_FORMATS: Registry[GridFactory] = Registry("quantization format", "economize.quant_formats")


def _int_format(bits: int) -> GridFactory:
    """The factory of the ``bits``-bit integer grid."""

    def factory(symmetric: bool) -> QuantGrid:
        return IntGrid(bits, symmetric)

    return factory


for _bits in (2, 3, 4, 8):
    QUANT_FORMATS.register(f"int{_bits}", _int_format(_bits))
QUANT_FORMATS.register("fp8", lambda symmetric: Fp8Grid())


def make_grid(format: str, symmetric: bool = False) -> QuantGrid:
    """The grid of a registered format."""
    return QUANT_FORMATS.get(format)(symmetric)


@dataclass
class QuantizedWeight:
    """A group-wise quantized ``[out, in]`` weight.

    Attributes:
        grid: The number format.
        codes: ``[out, in]`` codes (``uint8`` or ``float8_e4m3fn``), in the original column order.
        scale: ``[out, groups]`` float32 scales.
        zero: ``[out, groups]`` float32 zero points, ``None`` for scale-only grids. Integer-valued except for
            methods with real-valued zero points (HQQ, see :meth:`~.methods.QuantMethod.float_zeros`).
        group_size: Input columns per group (``in`` for per-channel quantization).
        g_idx: ``[in]`` group of every input column when the groups are not contiguous (GPTQ's ``act_order``
            without ``static_groups``), else ``None`` (column ``j`` belongs to group ``j // group_size``). Only
            backends with :attr:`~.backends.QuantBackend.supports_g_idx` store such weights.
    """

    grid: QuantGrid
    codes: Tensor
    scale: Tensor
    zero: Tensor | None
    group_size: int
    g_idx: Tensor | None = None

    def dequantize(self, dtype: torch.dtype = torch.float32) -> Tensor:
        if self.g_idx is None:
            scale = self.scale.repeat_interleave(self.group_size, dim=1)
            zero = None if self.zero is None else self.zero.repeat_interleave(self.group_size, dim=1)
        else:
            g_idx = self.g_idx.to(self.scale.device)
            scale = self.scale[:, g_idx]
            zero = None if self.zero is None else self.zero[:, g_idx]
        return self.grid.decode(self.codes, QParams(scale, zero)).to(dtype)

    @property
    def bits_per_weight(self) -> float:
        return self.grid.bits + self.grid.overhead_bits / self.group_size


def quantize_rtn(weight: Tensor, grid: QuantGrid, group_size: int | None = None) -> QuantizedWeight:
    """Round-to-nearest group-wise quantization of ``weight [out, in]`` (the baseline of GPTQ and AWQ).

    An unbound grid (:attr:`QuantGrid.requires_binding`) is bound to ``weight`` alone first.
    """
    w = weight.detach().float()
    if grid.requires_binding:
        grid = grid.bind([w])
    size = group_size or w.shape[1]
    if w.shape[1] % size:
        raise ConfigError(f"group size {size} does not divide {w.shape[1]} input features")
    codes = torch.empty(w.shape, dtype=grid.code_dtype, device=w.device)  # slices, as torch.cat lacks fp8 on CPU
    scales, zeros = [], []
    for start in range(0, w.shape[1], size):
        block = w[:, start : start + size]
        params = grid.fit(block)
        codes[:, start : start + size] = grid.encode(block, params)
        scales.append(params.scale)
        zeros.append(params.zero)
    zero = None if zeros[0] is None else torch.cat(zeros, 1)  # type: ignore[arg-type]
    return QuantizedWeight(grid, codes, torch.cat(scales, 1), zero, size)


#: Linear maps that inference engines fuse into one GEMM, by their names inside one parent module (vLLM's
#: ``qkv_proj`` and ``gate_up_proj``). A grid with tensor-level parameters fits them once per fused map
#: (:func:`bind_shards`); an architecture with other fused maps appends its tuple.
FUSED_SHARDS: list[tuple[str, ...]] = [("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj")]


def _weight(module: nn.Module) -> Tensor:
    weight = module.weight
    assert isinstance(weight, Tensor)
    return weight.detach()


def bind_shards(modules: Mapping[str, nn.Module], grid: QuantGrid) -> dict[str, QuantGrid]:
    """The grid of every module, bound once per fused map of :data:`FUSED_SHARDS` and once per other module.

    Args:
        modules: Linear maps by name inside their decoder layer (``"self_attn.q_proj"``).
        grid: The configured grid.

    Returns:
        The grid each module is quantized with (``grid`` itself for formats without tensor-level parameters).
    """
    if not grid.requires_binding:
        return dict.fromkeys(modules, grid)
    members: dict[tuple[str, int], list[str]] = defaultdict(list)
    for name in modules:
        parent, _, leaf = name.rpartition(".")
        fused = next((i for i, shards in enumerate(FUSED_SHARDS) if leaf in shards), None)
        members[(parent, fused) if fused is not None else (name, -1)].append(name)
    bound: dict[str, QuantGrid] = {}
    for names in members.values():
        shared = grid.bind([_weight(modules[name]) for name in names])
        bound.update(dict.fromkeys(names, shared))
    return bound
