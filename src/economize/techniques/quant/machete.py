"""The ``"machete"`` backend: vLLM's Machete mixed-precision GEMM for weight-only int4 and int8 on Hopper (sm90).

Machete (Wilkinson et al., 2024, the CUTLASS 3 successor of Marlin in vLLM) multiplies 16-bit activations by 4- or
8-bit weights with Hopper's warp-group MMA and TMA loads, which Marlin (written for Ampere's ``mma.sync``) does not
use. vLLM selects it over Marlin for GPTQ, AWQ and compressed-tensors checkpoints on H100. The kernels come from the
vLLM wheel (``torch.ops._C.machete_prepack_B`` and ``machete_mm``); nothing is compiled here.

  ==========  ==========================  =========================================================================
  format      Machete weight type         codes
  ==========  ==========================  =========================================================================
  ``int4``    ``uint4b8`` / ``uint4``     symmetric (zero 8) or with integer zero points (the kernel adds
                                          ``-scale * zero`` per group)
  ``int8``    ``uint8b128``               symmetric (zero 128) only
  ==========  ==========================  =========================================================================

The kernel adds zero points as a per-group shift ``-scale * zero`` in the compute dtype. bf16 has 8 significant bits,
so for 8-bit zero points (up to 255) the rounded shift moves every weight of a group by up to half a quantization step
(about 1% of the groups of a random matrix land a whole zero point off), which the ``"fake"`` backend does not do.
Affine int8 is therefore refused; 4-bit zero points (up to 15) stay exact.

Group sizes: ``None`` (one scale per output channel), 64 or 128. Shapes whose input features miss a multiple of 64 or
whose output features miss a multiple of 128 are zero-padded at conversion, as in the Marlin backend.

The stored weight is a :class:`MacheteWeight`, the :class:`~.marlin.MarlinWeight` tensor subclass with Machete's
layout, so ``F.linear`` dispatch, ``.to(dtype)``, CUDA graph capture and :meth:`~.marlin.MarlinWeight.dequantize`
behave as for Marlin. ``tests/techniques/test_quant_machete.py`` checks every layout against the ``"fake"`` backend.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...errors import ConfigError, UnsupportedModelError
from .backends import QUANT_BACKENDS, QuantBackend
from .config import QuantConfig, QuantSpec
from .constraints import KernelConstraints, format_listing
from .grid import QuantizedWeight
from .marlin import MarlinLayout, MarlinWeight, _gptq_pack, _load_ops, _round_up, _type_id

__all__ = [
    "MACHETE_CONSTRAINTS",
    "MacheteBackend",
    "MacheteWeight",
    "machete_available",
    "machete_layout_error",
    "machete_padded_nk",
    "machete_weight",
]

#: What Machete stores, by format (bf16 / fp16 activations).
MACHETE_CONSTRAINTS = {
    "int4": KernelConstraints("Machete int4", group_sizes=(None, 64, 128)),
    "int8": KernelConstraints("Machete int8", group_sizes=(None, 64, 128), symmetric="always"),
}
_TYPES = {("int4", True): "uint4b8", ("int4", False): "uint4", ("int8", True): "uint8b128", ("int8", False): "uint8"}
_BLOCK_K, _BLOCK_N = 64, 128  # vLLM's MACHETE_PREPACKED_BLOCK_SHAPE
_CAPABILITY = (9, 0)


@functools.cache
def _machete_reason() -> str | None:
    """Why the Machete ops are unavailable, or ``None`` (it does not check the GPU)."""
    reason = _load_ops()
    if reason is not None:
        return reason.replace("'marlin'", "'machete'").replace("Marlin", "Machete")
    if not (hasattr(torch.ops._C, "machete_mm") and hasattr(torch.ops._C, "machete_prepack_B")):
        return "this vllm build has no Machete kernels (torch.ops._C.machete_mm / machete_prepack_B)"
    return None


def machete_available() -> bool:
    """Whether vLLM's Machete kernels can be loaded (it does not check the GPU)."""
    return _machete_reason() is None


def machete_layout_error(format: str, group_size: int | None, symmetric: bool) -> str | None:
    """Why Machete has no kernel for a layout, or ``None`` if it has one."""
    if format not in MACHETE_CONSTRAINTS:
        return f"the machete backend stores {format_listing(list(MACHETE_CONSTRAINTS))}, got format {format!r}"
    reason = MACHETE_CONSTRAINTS[format].layout_error(group_size, symmetric)
    if reason is not None and format == "int8" and not symmetric and "has no zero points" in reason:
        return (
            "Machete int8 has no exact zero points: the kernel adds -scale * zero in the compute dtype, which bf16 "
            "rounds by up to half a quantization step for 8-bit zero points; use symmetric=True or format='int4'"
        )
    return reason


def machete_padded_nk(size_n: int, size_k: int, group_size: int = -1) -> tuple[int, int]:
    """The smallest ``(n, k)`` that Machete's prepacked blocks tile, keeping whole groups."""
    group = group_size if group_size > 0 else 1
    k_multiple = _BLOCK_K * group // math.gcd(_BLOCK_K, group)
    return _round_up(size_n, _BLOCK_N), _round_up(size_k, k_multiple)


def _check_device(device: torch.device, what: str) -> None:
    if device.type != "cuda":
        raise UnsupportedModelError(f"Machete kernels run on CUDA GPUs; {what} is on {device}")
    if torch.cuda.get_device_capability(device) != _CAPABILITY:
        raise UnsupportedModelError(
            f"Machete runs on Hopper (compute capability 9.0) only; {what} is on {torch.cuda.get_device_name(device)}"
            " (use backend='marlin')"
        )


class MacheteWeight(MarlinWeight):
    """A ``[out, in]`` weight in Machete's prepacked layout; ``F.linear`` runs ``machete_mm``.

    ``scales`` are ``[groups, n]`` in the compute dtype; ``zeros`` (affine codes only) hold ``-scale * zero``.
    """

    def __repr__(self) -> str:  # type: ignore[override]
        lay = self.packing
        group = "channel" if lay.group_size < 0 else lay.group_size
        return f"MacheteWeight({lay.type_name}, shape={tuple(self.shape)}, group={group}, dtype={self.dtype})"

    def linear(self, x: Tensor, bias: Tensor | None = None) -> Tensor:
        """``x @ W.T + bias`` with the Machete GEMM (any leading dimensions)."""
        lay = self.packing
        out_dtype = x.dtype
        a = x.reshape(-1, x.shape[-1])
        if a.dtype != lay.compute_dtype:
            a = a.to(lay.compute_dtype)
        if lay.padded_k != lay.size_k:
            a = F.pad(a, (0, lay.padded_k - lay.size_k))
        a = a.contiguous()
        if a.shape[0] == 0:
            out = a.new_empty(0, lay.size_n)
        else:
            out = torch.ops._C.machete_mm(
                a, self.qweight, _type_id(lay.type_name), None, self.scales, self.zeros, lay.group_size,
                None, None, None,  # channel scales, token scales, schedule (the kernel's heuristic)
            )  # fmt: skip
            if lay.padded_n != lay.size_n:
                out = out[:, : lay.size_n]
        out = out.to(out_dtype)
        if bias is not None:
            out = out + bias
        return out.reshape(*x.shape[:-1], lay.size_n)


@torch.no_grad()
def machete_weight(q: QuantizedWeight, dtype: torch.dtype, device: torch.device | None = None) -> MacheteWeight:
    """Convert a :class:`QuantizedWeight` to a :class:`MacheteWeight` reporting ``dtype``.

    Raises:
        ConfigError: The ops are missing, or the format, zero points or group size have no Machete kernel.
    """
    reason = _machete_reason()
    if reason is not None:
        raise ConfigError(reason)
    device = device or q.codes.device
    size_n, size_k = q.codes.shape
    compute = dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    per_channel = q.group_size >= size_k
    group = -1 if per_channel else q.group_size
    if q.codes.dtype.is_floating_point:
        raise ConfigError("Machete stores integer codes (int4, int8); use backend='marlin' or 'torchao' for fp8")
    bits = q.grid.bits
    symmetric = q.zero is not None and bool((q.zero == 2 ** (bits - 1)).all())
    reason = machete_layout_error(q.grid.name, None if per_channel else q.group_size, symmetric)
    if reason is None and q.g_idx is not None:
        reason = "Machete is given contiguous groups only (no g_idx): quantize with static_groups=True"
    if reason is not None:
        raise ConfigError(reason)
    type_name = _TYPES[(q.grid.name, symmetric)]
    padded_n, padded_k = machete_padded_nk(size_n, size_k, group)
    layout = MarlinLayout(size_n, size_k, padded_n, padded_k, type_name, bits, group, compute)

    codes = q.codes.to(device)
    pad_n, pad_k = padded_n - size_n, padded_k - size_k
    if pad_n or pad_k:  # code 0 with scale 0 (padded outputs) or zero activations (padded inputs) adds nothing
        codes = F.pad(codes, (0, pad_k, 0, pad_n))
    packed = _gptq_pack(codes, bits)  # [k * bits / 32, n], packed along k
    from vllm.scalar_type import scalar_types

    b_type = getattr(scalar_types, type_name).id
    qweight = torch.ops._C.machete_prepack_B(packed.t().contiguous().t(), compute, b_type, compute)

    groups = 1 if per_channel else padded_k // group
    scale32 = F.pad(q.scale.to(device=device, dtype=torch.float32).T, (0, pad_n, 0, groups - q.scale.shape[1]))
    scales = scale32.to(compute).contiguous()  # [groups, n]
    zeros: Tensor | None = None
    if not symmetric:
        if q.zero is None:
            raise ConfigError(f"Machete {q.grid.name} codes need zero points; this {q.grid.name} weight has none")
        zero = F.pad(q.zero.to(device=device, dtype=torch.float32).T, (0, pad_n, 0, groups - q.zero.shape[1]))
        zeros = (-scales.float() * zero).to(compute).contiguous()  # the kernel's w = code * scale + zeros
    return MacheteWeight(qweight, scales, zeros, torch.empty(0, dtype=torch.int32, device=device), layout, dtype)


@QUANT_BACKENDS.register("machete")
class MacheteBackend(QuantBackend):
    """vLLM's Machete weight-only GEMM on Hopper (``pip install vllm``; tested with vLLM 0.30 on H100)."""

    def check_config(self, config: QuantSpec) -> None:
        reason = machete_layout_error(config.format, config.group_size, config.grid.symmetric)
        if reason is not None:
            raise ConfigError(reason)

    def check(self, modules: Mapping[str, nn.Linear], config: QuantSpec) -> None:
        reason = _machete_reason()
        if reason is not None:
            raise ConfigError(reason)
        whole_model = isinstance(config, QuantConfig)
        for name, module in modules.items():
            weight = module.weight
            if whole_model and weight.dtype not in (torch.bfloat16, torch.float16):
                raise UnsupportedModelError(f"Machete needs a bfloat16 or float16 model; {name} is {weight.dtype}")
            _check_device(weight.device, name)

    def apply(self, module: nn.Linear, qweight: QuantizedWeight, config: QuantSpec) -> None:
        weight = module.weight
        module.weight = nn.Parameter(machete_weight(qweight, weight.dtype, weight.device), requires_grad=False)
