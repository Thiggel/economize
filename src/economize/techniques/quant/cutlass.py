"""The ``"cutlass"`` backend: vLLM's CUTLASS int8 x int8 GEMM for W8A8 (SmoothQuant) models.

With ``activations="int8"`` a linear map quantizes its input per token (``dynamic_scaled_int8_quant``, scale
``max|x| / 127``, round half to even, the ``"int8"`` scheme of :mod:`.activations`) and multiplies the int8 codes by
per-channel symmetric int8 weights with ``cutlass_scaled_mm``, which accumulates in int32 and applies both scales and
the bias in its epilogue. This is the kernel vLLM's ``CompressedTensorsW8A8Int8`` layers run on Ampere and Hopper. The
torchao int8 tensor computes the same product with ``torch._int_mm`` but reads its activation scale back to the host
while dispatching, so it cannot be captured in a CUDA graph; nothing here leaves the GPU.

  ===========  ======================================================================================
  stored       per-channel symmetric int8 codes ``[k, n]`` (column-major view of ``[n, k]``) and fp32
               scales ``[1, n]``; ``n`` and ``k`` zero-padded to multiples of 16
  activations  ``"int8"`` only (the kernel has no 16-bit operand)
  GPUs         compute capability 7.5 or newer (vLLM's c2x kernels on sm75-89, c3x on sm90)
  ===========  ======================================================================================

The stored weight is a :class:`CutlassInt8Weight`, the :class:`~.marlin.MarlinWeight` tensor subclass with this
layout, so ``F.linear`` dispatch, ``.to(dtype)``, CUDA graph capture and readback behave as for Marlin.
``tests/techniques/test_quant_cutlass.py`` checks the product against the ``"fake"`` backend.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...errors import ConfigError, UnsupportedModelError
from . import activations
from .backends import QUANT_BACKENDS, QuantBackend
from .config import QuantSpec
from .constraints import KernelConstraints
from .grid import QuantizedWeight
from .marlin import MarlinLayout, MarlinWeight, _load_ops, _round_up

__all__ = ["CUTLASS_INT8", "CutlassBackend", "CutlassInt8Weight", "cutlass_available", "cutlass_int8_weight"]

#: What the CUTLASS W8A8 kernel stores: int8 codes with one scale per output channel and no zero points.
CUTLASS_INT8 = KernelConstraints("CUTLASS int8", group_sizes=(None,), symmetric="always", min_capability=(7, 5))
_MULTIPLE = 16  # cutlass_scaled_mm tiles n and k in multiples of 16


@functools.cache
def _cutlass_reason() -> str | None:
    """Why the CUTLASS int8 ops are unavailable, or ``None`` (it does not check the GPU)."""
    reason = _load_ops()
    if reason is not None:
        return reason.replace("'marlin'", "'cutlass'").replace("Marlin", "CUTLASS")
    if not (hasattr(torch.ops._C, "cutlass_scaled_mm") and hasattr(torch.ops._C, "dynamic_scaled_int8_quant")):
        return "this vllm build has no CUTLASS int8 kernels (cutlass_scaled_mm / dynamic_scaled_int8_quant)"
    return None


def cutlass_available() -> bool:
    """Whether vLLM's CUTLASS int8 kernels can be loaded (it does not check the GPU)."""
    return _cutlass_reason() is None


class CutlassInt8Weight(MarlinWeight):
    """A ``[out, in]`` int8 weight for ``cutlass_scaled_mm``; ``F.linear`` quantizes the input per token first.

    ``qweight`` is ``[padded_k, padded_n]`` int8 (column-major), ``scales`` ``[1, padded_n]`` fp32.
    """

    def __repr__(self) -> str:  # type: ignore[override]
        return f"CutlassInt8Weight(int8 x int8, shape={tuple(self.shape)}, dtype={self.dtype})"

    def linear(self, x: Tensor, bias: Tensor | None = None) -> Tensor:
        """``x @ W.T + bias`` with per-token int8 activations and the CUTLASS GEMM (any leading dimensions)."""
        lay = self.packing
        out_dtype = x.dtype
        a = x.reshape(-1, x.shape[-1])
        if a.dtype != lay.compute_dtype:
            a = a.to(lay.compute_dtype)
        if lay.padded_k != lay.size_k:  # zero inputs leave the absmax and the product unchanged
            a = F.pad(a, (0, lay.padded_k - lay.size_k))
        a = a.contiguous()
        if a.shape[0] == 0:
            out = a.new_empty(0, lay.size_n)
        else:
            codes = torch.empty_like(a, dtype=torch.int8)
            token_scales = torch.empty(a.shape[0], 1, dtype=torch.float32, device=a.device)
            torch.ops._C.dynamic_scaled_int8_quant(codes, a, token_scales, None)
            out = torch.empty(a.shape[0], lay.padded_n, dtype=lay.compute_dtype, device=a.device)
            torch.ops._C.cutlass_scaled_mm(out, codes, self.qweight, token_scales, self.scales, None)
            if lay.padded_n != lay.size_n:
                out = out[:, : lay.size_n]
        out = out.to(out_dtype)
        if bias is not None:
            out = out + bias
        return out.reshape(*x.shape[:-1], lay.size_n)

    @torch.no_grad()
    def dequantize(self, chunk: int = 2048) -> Tensor:
        """The dense ``[out, in]`` weight in the reported dtype (codes times scales, rounded once)."""
        lay = self.packing
        dense = self.qweight[: lay.size_k, : lay.size_n].T.float() * self.scales[0, : lay.size_n, None]
        return dense.to(self.dtype).contiguous()


@torch.no_grad()
def cutlass_int8_weight(
    q: QuantizedWeight, dtype: torch.dtype, device: torch.device | None = None
) -> CutlassInt8Weight:
    """Convert per-channel symmetric int8 codes to a :class:`CutlassInt8Weight` reporting ``dtype``.

    Raises:
        ConfigError: The ops are missing, or the codes are not per-channel symmetric int8.
    """
    reason = _cutlass_reason()
    if reason is not None:
        raise ConfigError(reason)
    device = device or q.codes.device
    size_n, size_k = q.codes.shape
    if q.grid.name != "int8" or q.zero is None or not bool((q.zero == 128).all()):
        raise ConfigError("the cutlass backend stores symmetric int8 codes (zero point 128) only")
    if q.group_size < size_k:
        raise ConfigError("the cutlass backend stores per-channel scales only (group_size=None)")
    compute = dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    padded_n, padded_k = _round_up(size_n, _MULTIPLE), _round_up(size_k, _MULTIPLE)
    layout = MarlinLayout(size_n, size_k, padded_n, padded_k, "int8", 8, -1, compute)

    codes = (q.codes.to(device=device, dtype=torch.int16) - 128).to(torch.int8)
    codes = F.pad(codes, (0, padded_k - size_k, 0, padded_n - size_n))  # padded outputs get scale 0
    qweight = codes.t()  # [k, n], column-major, as vLLM's CutlassScaledMMLinearKernel stores it
    scale = F.pad(q.scale.to(device=device, dtype=torch.float32)[:, 0], (0, padded_n - size_n))
    empty = torch.empty(0, dtype=torch.int32, device=device)
    return CutlassInt8Weight(qweight, scale[None].contiguous(), None, empty, layout, dtype)


@QUANT_BACKENDS.register("cutlass")
class CutlassBackend(QuantBackend):
    """vLLM's CUTLASS W8A8 int8 GEMM (``pip install vllm``; tested with vLLM 0.30 on H100). Needs
    ``activations="int8"``."""

    options = frozenset({"activations"})

    def check_config(self, config: QuantSpec) -> None:
        if config.format != "int8":
            raise ConfigError(f"the cutlass backend stores format 'int8' with int8 activations, got {config.format!r}")
        if config.activations != "int8":
            raise ConfigError(
                "the cutlass backend multiplies int8 activations by int8 weights: set activations='int8' "
                "(for weight-only int8 use backend='marlin' or 'machete')"
            )
        reason = CUTLASS_INT8.layout_error(config.group_size, config.grid.symmetric)
        if reason is not None:
            raise ConfigError(reason)

    def check(self, modules: Mapping[str, nn.Linear], config: QuantSpec) -> None:
        reason = _cutlass_reason()
        if reason is not None:
            raise ConfigError(reason)
        for name, module in modules.items():
            reason = CUTLASS_INT8.device_error(name, module.weight)
            if reason is not None:
                raise UnsupportedModelError(reason)

    def apply(self, module: nn.Linear, qweight: QuantizedWeight, config: QuantSpec) -> None:
        weight = module.weight
        module.weight = nn.Parameter(cutlass_int8_weight(qweight, weight.dtype, weight.device), requires_grad=False)
        activations.mark(module, "int8")  # the weight quantizes its inputs itself
