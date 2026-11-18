"""The ``"triton"`` backend: a weight-only int8 / fp8 GEMM in Triton that decodes the codes in registers.

torchao runs weight-only int8 and weight-only fp8 by dequantizing the whole weight on every eager
call, which makes the eager CUDA-graph decoders slower than the dense model (0.50x for int8 and 0.14x for fp8 in the
B8 reproduction). This kernel reads the 8-bit codes once, converts each tile to the activation dtype inside the
GEMM and applies the scales to the fp32 partial sums, so a decode step reads half the bytes of the dense weight. It
needs only Triton (shipped with PyTorch on Linux), no vLLM.

  ===========  ======================================================================================
  stored       8-bit codes ``[n, k]`` (signed for symmetric int8, the bytes of e4m3 for fp8), fp32 scales
               ``[n, groups]`` and, for asymmetric int8, int16 zero points ``[n, groups]``
  formats      ``int8`` (per channel or groups of 32-256, with or without zero points) and weight-only
               ``fp8`` (per channel or groups of 32-256)
  rows         at most 64 rows per call run the kernel (decoding, verification, short chunks); longer
               inputs (prefill) dequantize the weight and call the dense GEMM
  GPUs         compute capability 8.0 or newer, bfloat16 or float16 weights
  ===========  ======================================================================================

The stored weight is a :class:`TritonWeight`, the :class:`~.marlin.MarlinWeight` tensor subclass with this layout,
so ``F.linear`` dispatch, ``.to(dtype)`` and CUDA graph capture behave as for Marlin. What the kernel accepts per
format is a :class:`KernelConstraints` in :data:`TRITON_FORMATS`. ``tests/techniques/test_quant_triton.py`` checks
the product against the ``"fake"`` backend.
"""

from __future__ import annotations

import functools
import importlib.util
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...errors import ConfigError
from ...registry import Registry
from .backends import QUANT_BACKENDS, QuantBackend
from .config import QuantSpec
from .constraints import KernelConstraints, check_devices, check_layout, format_listing
from .grid import QuantizedWeight
from .marlin import MarlinLayout, MarlinWeight

__all__ = ["TRITON_FORMATS", "TritonBackend", "TritonWeight", "triton_available", "triton_weight"]

_GROUPS = (None, 32, 64, 128, 256)
_GPUS = dict(dtypes=(torch.bfloat16, torch.float16), min_capability=(8, 0))

#: What the Triton weight-only GEMM accepts, by format; a format missing here is not stored by this backend.
TRITON_FORMATS: Registry[KernelConstraints] = Registry("Triton weight-only GEMM format")
TRITON_FORMATS.register("int8", KernelConstraints("Triton int8", group_sizes=_GROUPS, **_GPUS))  # type: ignore[arg-type]
TRITON_FORMATS.register("fp8", KernelConstraints("Triton fp8", group_sizes=_GROUPS, **_GPUS))  # type: ignore[arg-type]

#: Inputs with more rows than this dequantize the weight and call the dense GEMM (the kernel reads the codes once
#: per 16 rows, which pays off only while the weight read dominates).
MAX_KERNEL_ROWS = 64
_BLOCK_M, _BLOCK_N, _BLOCK_K = 16, 64, 128
_FP8_SCALE = 256.0  # e4m3 bits shifted into fp16 read as the value times 2^-8


@functools.cache
def _triton_reason() -> str | None:
    """Why the Triton kernel cannot be compiled, or ``None`` (it does not check the GPU)."""
    if importlib.util.find_spec("triton") is None:
        return "backend 'triton' needs the triton package (installed with PyTorch's CUDA wheels on Linux)"
    return None


def triton_available() -> bool:
    """Whether triton is installed (it does not check the GPU)."""
    return _triton_reason() is None


class TritonWeight(MarlinWeight):
    """A ``[out, in]`` 8-bit weight for the Triton weight-only GEMM.

    ``qweight`` is ``[n, k]``: int8 ``code - 128`` for symmetric int8 (``packing.type_name == "int8"``), uint8 codes
    with int16 ``zeros [n, groups]`` for asymmetric int8 (``"uint8"``), the bytes of e4m3 codes for fp8
    (``"float8_e4m3fn"``). ``scales`` is ``[n, groups]`` fp32 (times 256 for fp8); ``packing.group_size`` is ``-1``
    for per-channel scales.
    """

    def __repr__(self) -> str:  # type: ignore[override]
        lay = self.packing
        group = "channel" if lay.group_size < 0 else lay.group_size
        return f"TritonWeight({lay.type_name}, shape={tuple(self.shape)}, group={group}, dtype={self.dtype})"

    @property
    def is_fp8(self) -> bool:
        return self.packing.type_name == "float8_e4m3fn"

    def linear(self, x: Tensor, bias: Tensor | None = None) -> Tensor:
        """``x @ W.T + bias`` (any leading dimensions): the Triton kernel for short inputs, else a dense GEMM."""
        lay = self.packing
        out_dtype = x.dtype
        a = x.reshape(-1, x.shape[-1])
        if a.dtype != lay.compute_dtype:
            a = a.to(lay.compute_dtype)
        rows = a.shape[0]
        if rows > MAX_KERNEL_ROWS:
            out = F.linear(a, self._dense(lay.compute_dtype))
        else:
            out = a.new_empty(rows, lay.size_n)
            if rows:
                self._launch(a.contiguous(), out)
        out = out.to(out_dtype)
        if bias is not None:
            out = out + bias
        return out.reshape(*x.shape[:-1], lay.size_n)

    def _launch(self, a: Tensor, out: Tensor) -> None:
        import triton

        from ._triton_gemm_kernels import _weight_only_gemm

        lay = self.packing
        per_channel = lay.group_size < 0
        group = lay.size_k if per_channel else lay.group_size
        zeros = self.zeros if self.zeros is not None else self.scales
        grid = (triton.cdiv(lay.size_n, _BLOCK_N), triton.cdiv(a.shape[0], _BLOCK_M))
        _weight_only_gemm[grid](
            a, self.qweight, self.scales, zeros, out, a.shape[0], lay.size_n, lay.size_k, group,
            a.stride(0), self.qweight.stride(0), self.scales.stride(0), zeros.stride(0), out.stride(0),
            FP8=self.is_fp8, HAS_ZERO=self.zeros is not None, PER_CHANNEL=per_channel,
            BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K if per_channel else min(_BLOCK_K, group),
            num_warps=4, num_stages=3,
        )  # fmt: skip

    def _dense(self, dtype: torch.dtype) -> Tensor:
        lay = self.packing
        group = lay.size_k if lay.group_size < 0 else lay.group_size
        fp8 = self.is_fp8
        codes = self.qweight.view(torch.float8_e4m3fn).float() / _FP8_SCALE if fp8 else self.qweight.float()
        codes = codes.view(lay.size_n, -1, group)
        if self.zeros is not None:
            codes = codes - self.zeros.float()[..., None]
        return (codes * self.scales[..., None]).view(lay.size_n, lay.size_k).to(dtype)

    @torch.no_grad()
    def dequantize(self, chunk: int = 2048) -> Tensor:
        """The dense ``[out, in]`` weight in the reported dtype (codes times scales in fp32, rounded once)."""
        return self._dense(self.dtype).contiguous()


@torch.no_grad()
def triton_weight(q: QuantizedWeight, dtype: torch.dtype, device: torch.device | None = None) -> TritonWeight:
    """Convert int8 or fp8 codes to a :class:`TritonWeight` reporting ``dtype``.

    Raises:
        ConfigError: Triton is missing, the format is not stored, the zero points are not integers or the groups
            are not contiguous.
    """
    reason = _triton_reason()
    if reason is not None:
        raise ConfigError(reason)
    if q.grid.name not in TRITON_FORMATS:
        raise ConfigError(f"the triton backend stores {format_listing(TRITON_FORMATS.names())}, got {q.grid.name!r}")
    if q.g_idx is not None:
        raise ConfigError("the triton backend stores contiguous groups only (no act_order g_idx)")
    device = device or q.codes.device
    size_n, size_k = q.codes.shape
    per_channel = q.group_size >= size_k
    compute = dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    fp8 = q.codes.dtype == torch.float8_e4m3fn
    symmetric = not fp8 and bool((q.zero == 128).all())  # type: ignore[union-attr]
    type_name = "float8_e4m3fn" if fp8 else "int8" if symmetric else "uint8"
    layout = MarlinLayout(size_n, size_k, size_n, size_k, type_name, 8, -1 if per_channel else q.group_size, compute)
    codes = q.codes.to(device)
    if fp8:
        codes = codes.contiguous().view(torch.uint8)
    elif symmetric:  # signed codes need no zero point in the kernel
        codes = (codes.to(torch.int16) - 128).to(torch.int8)
    scales = q.scale.to(device=device, dtype=torch.float32) * (_FP8_SCALE if fp8 else 1.0)
    zeros = None
    if not fp8 and not symmetric:
        zero = q.zero.to(device=device, dtype=torch.float32)  # type: ignore[union-attr]
        if not bool((zero == zero.round()).all()):
            raise ConfigError("the triton backend stores integer zero points only (not HQQ's real-valued ones)")
        zeros = zero.to(torch.int16).contiguous()
    empty = torch.empty(0, dtype=torch.int32, device=device)
    return TritonWeight(codes.contiguous(), scales.contiguous(), zeros, empty, layout, dtype)


@QUANT_BACKENDS.register("triton")
class TritonBackend(QuantBackend):
    """A weight-only int8 / fp8 GEMM in Triton (no vLLM needed; sm80+, bf16 or fp16), the eager-mode fast path for
    the layouts that torchao runs by dequantizing the weight on every call."""

    def check_config(self, config: QuantSpec) -> None:
        if config.format not in TRITON_FORMATS:
            raise ConfigError(
                f"the triton backend stores {format_listing(TRITON_FORMATS.names())}, got format {config.format!r}"
            )
        check_layout(TRITON_FORMATS, config)

    def unavailable(self, config: QuantSpec) -> str | None:
        return _triton_reason()

    def check(self, modules: Mapping[str, nn.Linear], config: QuantSpec) -> None:
        reason = _triton_reason()
        if reason is not None:
            raise ConfigError(reason)
        check_devices(TRITON_FORMATS, modules, config)

    def apply(self, module: nn.Linear, qweight: QuantizedWeight, config: QuantSpec) -> None:
        weight = module.weight
        module.weight = nn.Parameter(triton_weight(qweight, weight.dtype, weight.device), requires_grad=False)
