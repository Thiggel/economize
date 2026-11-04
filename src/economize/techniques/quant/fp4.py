"""4-bit floating-point microscaling formats: NVFP4 and MXFP4 (FP4 E2M1 codes with a scale per small group).

Both store each weight as an OCP FP4 E2M1 code (sign, 2 exponent bits, 1 mantissa bit: the values
``0, 0.5, 1, 1.5, 2, 3, 4, 6`` and their negatives) and one scale per group of contiguous input columns. They differ
in the group size and in how the scale is stored:

* ``"mxfp4"`` (OCP Microscaling Formats v1.0): groups of 32, one E8M0 scale (a power of two) per group, 4.25 bits
  per weight.
* ``"nvfp4"`` (NVIDIA Blackwell): groups of 16, one float8 e4m3 scale per group times one float32 global scale per
  tensor, 4.5 bits per weight. The fused shards of one GEMM (``q/k/v``, ``gate/up``) share the global scale
  (:func:`~.grid.bind_shards`).

The scales and the rounding follow compressed-tensors (the format vLLM loads), which the tests check bit for bit:

* MXFP4: ``scale = 2^(floor(log2(p)) - 2)`` with ``p`` the group's absolute maximum rounded to a power of two
  (up when its mantissa is at least 1.75, as compressed-tensors' ``round_to_power_2``); values beyond ``6 * scale``
  clip.
* NVFP4: ``global = 448 * 6 / amax(tensor)``; the stored group scale is ``fp8_e4m3(amax(group) / 6 * global)``
  (0.125 for an all-zero group) and the effective scale is ``group / global``.
* E2M1 rounding is to nearest with compressed-tensors' tie rule (``cast_to_fp4``): ties at 0.25, 1.25, 2.5 and 5
  round down, ties at 0.75, 1.75 and 3.5 round up.

Codes are ``uint8`` values ``0..15``: bits 0-2 index :data:`E2M1_VALUES`, bit 3 is the sign. Zero is always code 0
(a negative value that rounds to zero loses its sign), which is what compressed-tensors' compressor writes for the
dequantized weight economize exports; both zeros decode to the same weight.

Hardware: FP4 tensor-core GEMMs (W4A4) exist only on Blackwell (sm100+). On A100 / H100 vLLM serves these
checkpoints with Marlin's W4A16 kernels (``MarlinNvFp4LinearKernel``, ``MarlinMxFp4LinearKernel``), which save weight
memory but run at int4-Marlin speed, so no FP4 timing is measured there.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor

from ...errors import ConfigError
from .grid import FP8_E4M3_MAX, QUANT_FORMATS, QParams, QuantGrid

__all__ = [
    "E2M1_VALUES",
    "FP4_E2M1_MAX",
    "Fp4Grid",
    "Mxfp4Grid",
    "Nvfp4Grid",
    "e2m1_round",
    "mx_scale",
    "nvfp4_global_scale",
]

#: The non-negative FP4 E2M1 values, indexed by the low three bits of a code.
E2M1_VALUES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_E2M1_MAX = 6.0
_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)  # midpoints between consecutive E2M1 values
_TIES_UP = (0.75, 1.75, 3.5)
_NVFP4_ZERO_SCALE = 0.125  # compressed-tensors' replacement of a zero fp8 group scale
_MX_EXPONENT_BIAS = 127
_MX_ELEMENT_OFFSET = 2  # floor(log2(6)): the largest E2M1 exponent


def e2m1_round(x: Tensor) -> Tensor:
    """The index (0..7) of the E2M1 value nearest to ``|x|``, with compressed-tensors' tie rule; ``|x| > 6`` clips.

    Args:
        x: Values already divided by their scale.

    Returns:
        ``int64`` indices into :data:`E2M1_VALUES`, shaped like ``x``.
    """
    a = x.float().abs()
    index = torch.bucketize(a, torch.tensor(_BOUNDS, device=a.device), right=False)
    ties = torch.zeros_like(a, dtype=torch.bool)
    for tie in _TIES_UP:
        ties |= a == tie
    return index + ties.long()


def _round_up_power_of_two(x: Tensor) -> Tensor:
    """compressed-tensors' ``round_to_power_2`` in float32: the power of two at or below ``x``, or the next one when
    the mantissa of ``x`` is at least 1.75 (``x >= 0``)."""
    bits = x.float().contiguous().view(torch.int32)
    rounded = (bits + (1 << 21)) & (0x1FF << 23)  # add a quarter of the mantissa range, keep sign and exponent
    return rounded.view(torch.float32)


def mx_scale(amax: Tensor) -> Tensor:
    """The E8M0 group scale of MXFP4 for group maxima ``amax``.

    Args:
        amax: Non-negative group absolute maxima.

    Returns:
        float32 powers of two ``2^(e - 127)`` with ``e`` the stored biased exponent in ``0..255``.
    """
    exponent = torch.floor(torch.log2(_round_up_power_of_two(amax))) - _MX_ELEMENT_OFFSET + _MX_EXPONENT_BIAS
    exponent = exponent.clamp(0, 255)  # log2(0) = -inf gives the smallest scale, 2^-127
    return torch.exp2(exponent - _MX_EXPONENT_BIAS)


def nvfp4_global_scale(weights: Sequence[Tensor]) -> float:
    """NVFP4's per-tensor scale ``448 * 6 / amax`` over the shards ``weights`` (1.0 for an all-zero tensor).

    Args:
        weights: The shards that share the global scale.

    Returns:
        The global scale (a float32 value), as compressed-tensors' ``generate_gparam`` computes it.
    """
    amax = torch.stack([w.detach().float().abs().amax() for w in weights]).amax().cpu()
    amax = amax.clamp(min=torch.finfo(torch.float32).tiny)
    scale = torch.nan_to_num(FP8_E4M3_MAX * FP4_E2M1_MAX / amax, nan=1.0, posinf=1.0, neginf=1.0)
    return float(scale)


@dataclass(frozen=True)
class Fp4Grid(QuantGrid):
    """FP4 E2M1 codes with a scale per row and group; subclasses choose the group size and the scale."""

    #: The only group size the format defines.
    group: ClassVar[int]
    bits: int = 4
    #: E2M1 is sign-magnitude: the format has no zero point whatever the flag says.
    symmetric: bool = True

    @property
    def code_dtype(self) -> torch.dtype:
        return torch.uint8

    def check_group_size(self, group_size: int | None) -> None:
        """Raise unless ``group_size`` is the format's group size.

        Raises:
            ConfigError: Another group size, or per-channel scales.
        """
        if group_size != self.group:
            raise ConfigError(f"format {self.name!r} quantizes groups of {self.group} weights: set "
                              f"group_size={self.group}, got {group_size}")

    def encode(self, w: Tensor, params: QParams) -> Tensor:
        x = w.float() / params.scale
        index = e2m1_round(x)
        negative = torch.signbit(x) & (index > 0)  # one zero code: -0 is stored as +0
        return (index | (negative.long() << 3)).to(torch.uint8)

    def decode(self, codes: Tensor, params: QParams) -> Tensor:
        table = torch.tensor(E2M1_VALUES, device=codes.device)
        index = codes.long()
        sign = 1.0 - 2.0 * (index >> 3).float()
        return table[index & 7] * sign * params.scale


@dataclass(frozen=True)
class Mxfp4Grid(Fp4Grid):
    """MXFP4: groups of 32 weights with an E8M0 (power-of-two) scale each."""

    group: ClassVar[int] = 32
    name: str = "mxfp4"

    def fit(self, w: Tensor) -> QParams:
        return QParams(mx_scale(w.float().abs().amax(1, keepdim=True)))

    @property
    def overhead_bits(self) -> int:
        return 8  # one E8M0 exponent per group


@dataclass(frozen=True)
class Nvfp4Grid(Fp4Grid):
    """NVFP4: groups of 16 weights with a float8 e4m3 scale each, times one per-tensor global scale.

    Attributes:
        global_scale: The per-tensor scale (``None`` until :meth:`bind`; an unbound grid fits each block with the
            block's own global scale, which AWQ's search uses).
    """

    group: ClassVar[int] = 16
    name: str = "nvfp4"
    global_scale: float | None = None

    @property
    def requires_binding(self) -> bool:
        return self.global_scale is None

    def bind(self, weights: Sequence[Tensor]) -> Nvfp4Grid:
        return Nvfp4Grid(global_scale=nvfp4_global_scale(weights))

    def local_scale(self, scale: Tensor) -> Tensor:
        """The stored float8 e4m3 group scale (as float32) of effective scales ``scale``.

        Args:
            scale: Effective scales ``group / global`` of this bound grid.

        Returns:
            float32 values on the e4m3 grid.
        """
        assert self.global_scale is not None
        product = scale.float() * torch.tensor(self.global_scale, dtype=torch.float32)
        return product.to(torch.float8_e4m3fn).float()

    def fit(self, w: Tensor) -> QParams:
        w = w.float()
        global_scale = self.global_scale if self.global_scale is not None else nvfp4_global_scale([w])
        gs = torch.tensor(global_scale, dtype=torch.float32, device=w.device)
        local = gs * (w.abs().amax(1, keepdim=True) / FP4_E2M1_MAX)
        local = local.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn).float()
        local = torch.where(local == 0, torch.full_like(local, _NVFP4_ZERO_SCALE), local)
        return QParams(local / gs)

    @property
    def overhead_bits(self) -> int:
        return 8  # one e4m3 scale per group (the per-tensor fp32 global scale is negligible)


QUANT_FORMATS.register("mxfp4", lambda symmetric: Mxfp4Grid())
QUANT_FORMATS.register("nvfp4", lambda symmetric: Nvfp4Grid())
