"""The ``"marlin"`` backend: vLLM's Marlin mixed-precision GEMM for weight-only int4, int8 and fp8 (sm80+).

Marlin (Frantar et al., 2024, "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language
Models") multiplies 16-bit activations by 4- or 8-bit weights that it dequantizes in registers, so a decoding step
reads the quantized bytes only. It is the weight-only kernel vLLM runs for GPTQ, AWQ, compressed-tensors W8A16 and
fp8 checkpoints on Ampere, where torch has no fused int8 or fp8 weight-only matmul (torchao's ``Int8Tensor`` and
``Float8Tensor`` dequantize the whole weight on every eager call). The kernels come from the vLLM wheel
(``torch.ops._C.marlin_gemm`` and ``gptq_marlin_repack``); nothing is compiled here.

  ==========  ===========================  ============================================================
  format      Marlin weight type           codes
  ==========  ===========================  ============================================================
  ``int8``    ``uint8b128``                symmetric only (zero point 128), per channel or per group
  ``int4``    ``uint4b8`` / ``uint4``      symmetric (zero 8) or with integer zero points (AWQ layout)
  ``fp8``     ``float8_e4m3fn``            e4m3 codes, per channel or groups of 128 input columns
  ==========  ===========================  ============================================================

Each format is one :class:`MarlinFormat` in :data:`MARLIN_FORMATS` (its weight types and the layouts its kernel
accepts). Group sizes: ``None`` (one scale per output channel), 32, 64 or 128 (fp8: ``None`` or 128). Shapes that miss
Marlin's thread tiles (``n % 64`` and ``k % 128``, or ``n % 128`` and ``k % 64``) are zero-padded at conversion;
the activations are then padded and the outputs sliced on every call.

The stored weight is a :class:`MarlinWeight`, a tensor subclass that keeps the ``[out, in]`` shape and dtype of
the weight it replaces and routes ``F.linear`` to the kernel, so modules, LoRA hooks, ``restore_draft`` and CUDA
graph capture work unchanged (the kernel allocates its output on the current stream and never synchronizes with
the host). Any other operation on it raises; :meth:`MarlinWeight.dequantize` returns the dense weight. A weight
whose module runs in float32 (a drafter's predictor) is computed in bfloat16 and cast back.

The packing helpers below reimplement vLLM's ``marlin_utils`` preprocessing (Apache-2.0) in torch, so that only
the two custom ops are taken from vLLM. ``tests/techniques/test_quant_marlin.py`` checks every format against the
``"fake"`` backend.
"""

from __future__ import annotations

import functools
import importlib.util
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...errors import ConfigError, UnsupportedModelError
from ...registry import Registry
from .backends import QUANT_BACKENDS, QuantBackend
from .config import QuantConfig
from .constraints import KernelConstraints, format_listing
from .grid import QuantizedWeight

if TYPE_CHECKING:
    from .config import QuantSpec

__all__ = [
    "MARLIN_FORMATS",
    "MARLIN_GROUP_SIZES",
    "MarlinBackend",
    "MarlinFormat",
    "MarlinLayout",
    "MarlinWeight",
    "marlin_available",
    "marlin_layout_error",
    "marlin_padded_nk",
    "marlin_weight",
]

#: Group sizes the Marlin kernels take besides per-channel scales (``group_size=None``).
MARLIN_GROUP_SIZES = (32, 64, 128)
_FP8_GROUP_SIZES = (128,)


@dataclass(frozen=True)
class MarlinFormat:
    """One format the Marlin kernels store.

    Attributes:
        constraints: The layouts the kernel accepts (group sizes, zero points).
        symmetric_type: vLLM scalar type of symmetric codes (a name in ``vllm.scalar_type.scalar_types``).
        affine_type: vLLM scalar type of codes with integer zero points (``None``: symmetric codes only).
    """

    constraints: KernelConstraints
    symmetric_type: str
    affine_type: str | None = None

    def type_name(self, symmetric: bool) -> str:
        """The Marlin weight type of symmetric or affine codes."""
        return self.symmetric_type if symmetric or self.affine_type is None else self.affine_type


#: The formats of the Marlin backend by name (as in :data:`~.grid.QUANT_FORMATS`).
MARLIN_FORMATS: Registry[MarlinFormat] = Registry("marlin format")
MARLIN_FORMATS.register("int8", MarlinFormat(
    KernelConstraints("Marlin int8", group_sizes=(None, *MARLIN_GROUP_SIZES), symmetric="always"), "uint8b128"
))
MARLIN_FORMATS.register("int4", MarlinFormat(
    KernelConstraints("Marlin int4", group_sizes=(None, *MARLIN_GROUP_SIZES)), "uint4b8", "uint4"
))
MARLIN_FORMATS.register("fp8", MarlinFormat(
    KernelConstraints("Marlin fp8", group_sizes=(None, *_FP8_GROUP_SIZES)), "float8_e4m3fn"
))


def marlin_layout_error(format: str, group_size: int | None, symmetric: bool) -> str | None:
    """Why Marlin has no kernel for a layout, or ``None`` if it has one.

    The single check behind :meth:`MarlinBackend.check_config` (on a config) and :func:`marlin_weight` (on the codes),
    against the format's entry in :data:`MARLIN_FORMATS`.

    Args:
        format: Format name (``"int4"``, ``"int8"``, ``"fp8"``, ...).
        group_size: Input columns per group, ``None`` for per-channel scales.
        symmetric: Whether the codes have no zero point (always true for fp8).
    """
    if format not in MARLIN_FORMATS:
        return f"the marlin backend stores {format_listing(MARLIN_FORMATS.names())}, got format {format!r}"
    return MARLIN_FORMATS.get(format).constraints.layout_error(group_size, symmetric)


_TILE = 16
_FP8_MAX_EXPONENT = 4  # e4m3


# ---- the vLLM ops -----------------------------------------------------------------------------------------------


@functools.cache
def _load_ops() -> str | None:
    """Import vLLM's custom ops once; return why they are unavailable, or ``None``."""
    if importlib.util.find_spec("vllm") is None:
        return "backend 'marlin' needs the vllm package (pip install vllm), which ships the Marlin kernels"
    try:
        import vllm._custom_ops  # noqa: F401  (loading it registers the Marlin ops under torch.ops._C)
    except Exception as error:  # pragma: no cover - depends on the installed wheel and driver
        return f"vllm is installed but its custom ops do not load: {error}"
    if not (hasattr(torch.ops._C, "marlin_gemm") and hasattr(torch.ops._C, "gptq_marlin_repack")):
        return "this vllm build has no Marlin kernels (torch.ops._C.marlin_gemm / gptq_marlin_repack)"
    return None


def marlin_available() -> bool:
    """Whether vLLM's Marlin kernels can be loaded (it does not check the GPU)."""
    return _load_ops() is None


@functools.cache
def _type_id(type_name: str) -> int:
    """The id of a vLLM scalar type (``"uint4b8"``, ``"float8_e4m3fn"``, ...)."""
    from vllm.scalar_type import scalar_types

    return int(getattr(scalar_types, type_name).id)


def _check_device(device: torch.device, dtype: torch.dtype, what: str) -> None:
    if device.type != "cuda":
        raise UnsupportedModelError(f"Marlin kernels run on CUDA GPUs; {what} is on {device}")
    need = (8, 0) if dtype in (torch.bfloat16, torch.float32) else (7, 5)
    if torch.cuda.get_device_capability(device) < need:
        raise UnsupportedModelError(
            f"Marlin with {dtype} needs compute capability {need[0]}.{need[1]} or newer; {what} is on "
            f"{torch.cuda.get_device_name(device)}"
        )


# ---- layout helpers (vLLM marlin_utils, reimplemented) -----------------------------------------------------------


def _round_up(x: int, multiple: int) -> int:
    return -(-x // multiple) * multiple


def marlin_padded_nk(size_n: int, size_k: int, group_size: int = -1) -> tuple[int, int]:
    """The smallest ``(n, k)`` that fits one of Marlin's thread-tile families, keeping whole groups."""
    group = group_size if group_size > 0 else 1
    candidates = (
        (_round_up(size_n, 64), _round_up(size_k, math.lcm(128, group))),
        (_round_up(size_n, 128), _round_up(size_k, math.lcm(64, group))),
    )
    return min(candidates, key=lambda nk: (nk[0] * nk[1], nk[0] + nk[1]))


@functools.cache
def _scale_perms() -> tuple[list[int], list[int]]:
    grouped = [i + 8 * j for i in range(8) for j in range(8)]
    single = [2 * i + j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25)]
    return grouped, single


def _permute_scales(scales: Tensor, size_n: int, grouped: bool) -> Tensor:
    """``[groups, n]`` scales in the order the kernel reads them."""
    grouped_perm, single_perm = _scale_perms()
    perm = grouped_perm if grouped else single_perm
    return scales.reshape(-1, len(perm))[:, perm].reshape(-1, size_n).contiguous()


def _pack(values: Tensor, bits: int) -> Tensor:
    """Pack the last dimension of non-negative ``values`` into int32 words, value ``i`` at bits ``bits*i``."""
    per_word = 32 // bits
    v = values.to(torch.int64).reshape(*values.shape[:-1], -1, per_word)
    shifts = torch.arange(per_word, device=values.device, dtype=torch.int64) * bits
    words = (v << shifts).sum(-1)
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)


def _permute_zero_points(zero: Tensor, size_n: int, bits: int) -> Tensor:
    """``[groups, n]`` integer zero points permuted, interleaved and packed along ``n`` as Marlin reads them."""
    grouped_perm, _ = _scale_perms()
    interleave = [0, 2, 4, 6, 1, 3, 5, 7] if bits == 4 else [0, 2, 1, 3]
    z = zero.reshape(-1, len(grouped_perm))[:, grouped_perm]
    z = z.reshape(-1, len(interleave))[:, interleave].reshape(-1, size_n)
    return _pack(z, bits).contiguous()


def _gptq_pack(codes: Tensor, bits: int) -> Tensor:
    """``[n, k]`` codes (one byte each) to the GPTQ layout ``[k * bits / 32, n]`` int32, packed along ``k``."""
    if bits == 8:  # four bytes per word, little endian: the byte of column 4i+j lands at bits 8j
        return codes.contiguous().view(torch.uint8).view(torch.int32).T.contiguous()
    return _pack(codes, bits).T.contiguous()


# ---- the stored weight ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MarlinLayout:
    """Static facts about a Marlin weight (shape before and after padding, number type, compute dtype).

    Attributes:
        size_n: Output features.
        size_k: Input features.
        padded_n: Output features after tile padding.
        padded_k: Input features after tile padding.
        type_name: Marlin weight type (``"uint8b128"``, ``"uint4b8"``, ``"uint4"`` or ``"float8_e4m3fn"``).
        bits: Bits per code.
        group_size: Input columns per scale, ``-1`` for one scale per output channel.
        compute_dtype: Activation dtype of the kernel (bfloat16 or float16).
    """

    size_n: int
    size_k: int
    padded_n: int
    padded_k: int
    type_name: str
    bits: int
    group_size: int
    compute_dtype: torch.dtype

    @property
    def padded(self) -> bool:
        return (self.padded_n, self.padded_k) != (self.size_n, self.size_k)


class MarlinWeight(Tensor):
    """A ``[out, in]`` weight stored in Marlin's packed layout; ``F.linear`` runs the Marlin GEMM.

    The tensor reports the shape, dtype and device of the dense weight it replaces. Only ``F.linear``, ``detach``,
    ``alias``, ``clone``, ``.to(dtype)`` (which changes the reported dtype only) and :meth:`dequantize` are
    supported.
    """

    qweight: Tensor
    scales: Tensor
    zeros: Tensor | None
    workspace: Tensor
    packing: MarlinLayout  # not ``layout``: ``Tensor.layout`` is a read-only attribute of every tensor

    _SHARE_OPS: ClassVar[set[Any]] = {torch.ops.aten.detach.default, torch.ops.aten.alias.default}

    @staticmethod
    def __new__(
        cls,
        qweight: Tensor,
        scales: Tensor,
        zeros: Tensor | None,
        workspace: Tensor,
        layout: MarlinLayout,
        dtype: torch.dtype,
    ) -> MarlinWeight:
        return Tensor._make_wrapper_subclass(
            cls, (layout.size_n, layout.size_k), dtype=dtype, device=qweight.device, requires_grad=False
        )

    def __init__(
        self,
        qweight: Tensor,
        scales: Tensor,
        zeros: Tensor | None,
        workspace: Tensor,
        layout: MarlinLayout,
        dtype: torch.dtype,
    ) -> None:
        self.qweight, self.scales, self.zeros, self.workspace, self.packing = qweight, scales, zeros, workspace, layout

    def __repr__(self) -> str:  # type: ignore[override]
        lay = self.packing
        group = "channel" if lay.group_size < 0 else lay.group_size
        return f"MarlinWeight({lay.type_name}, shape={tuple(self.shape)}, group={group}, dtype={self.dtype})"

    def _rewrap(self, dtype: torch.dtype | None = None, clone: bool = False) -> MarlinWeight:
        def copy(t: Tensor | None) -> Tensor | None:
            return t.clone() if clone and t is not None else t

        return type(self)(  # a subclass (MacheteWeight) keeps its own linear
            copy(self.qweight),  # type: ignore[arg-type]
            copy(self.scales),  # type: ignore[arg-type]
            copy(self.zeros),
            torch.zeros_like(self.workspace) if clone else self.workspace,
            self.packing,
            dtype or self.dtype,
        )

    def data_ptr(self) -> int:
        """The address of the packed codes (used as a cache key, e.g. by ``InModelDtype``)."""
        return self.qweight.data_ptr()

    def linear(self, x: Tensor, bias: Tensor | None = None) -> Tensor:
        """``x @ W.T + bias`` with the Marlin GEMM (any leading dimensions)."""
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
            out = torch.ops._C.marlin_gemm(
                a, None, self.qweight, None, self.scales, None, None, self.zeros, self.workspace,
                _type_id(lay.type_name), a.shape[0], lay.padded_n, lay.padded_k,
                False, True, False,  # use_atomic_add, use_fp32_reduce (vLLM's default), is_zp_float
            )  # fmt: skip
            if lay.padded_n != lay.size_n:
                out = out[:, : lay.size_n]
        out = out.to(out_dtype)
        if bias is not None:
            out = out + bias
        return out.reshape(*x.shape[:-1], lay.size_n)

    @torch.no_grad()
    def dequantize(self, chunk: int = 2048) -> Tensor:
        """The dense ``[out, in]`` weight in the reported dtype, read back through the kernel itself.

        Multiplying identity rows by the weight returns each dequantized entry rounded once to the compute dtype,
        exactly what the kernel uses; it needs no copy of the codes.
        """
        lay = self.packing
        rows = []
        for start in range(0, lay.size_k, chunk):
            stop = min(start + chunk, lay.size_k)
            eye = torch.zeros(stop - start, lay.size_k, dtype=lay.compute_dtype, device=self.device)
            index = torch.arange(stop - start, device=self.device)
            eye[index, index + start] = 1
            rows.append(self.linear(eye))
        return torch.cat(rows).T.contiguous().to(self.dtype)

    @classmethod
    def __torch_function__(cls, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None) -> Any:
        kwargs = kwargs or {}
        if func is F.linear:
            x, weight = args[0], args[1]
            bias = args[2] if len(args) > 2 else kwargs.get("bias")
            if isinstance(weight, MarlinWeight) and not isinstance(x, MarlinWeight):
                return weight.linear(x, bias)
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(  # type: ignore[override]  # torch types the disabled default, not the subclass protocol
        cls, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None
    ) -> Any:
        kwargs = kwargs or {}
        self = args[0]
        if func in cls._SHARE_OPS:
            return self._rewrap()
        if func is torch.ops.aten.clone.default:
            return self._rewrap(clone=True)
        if func is torch.ops.aten._to_copy.default:
            dtype = kwargs.get("dtype", self.dtype)
            device = kwargs.get("device")
            same_device = device is None or _same_device(torch.device(device), self.device)
            if same_device and dtype.is_floating_point:
                return self._rewrap(dtype)
        raise NotImplementedError(
            f"MarlinWeight supports F.linear, detach, clone and .to(dtype) only, not {func}; "
            "call .dequantize() for a dense weight"
        )


def _same_device(a: torch.device, b: torch.device) -> bool:
    current = torch.cuda.current_device() if torch.cuda.is_available() else 0
    return a.type == b.type and (a.index if a.index is not None else current) == (
        b.index if b.index is not None else current
    )


def _workspace(device: torch.device) -> Tensor:
    # one thread-block counter per SM; it must keep its address for CUDA graphs, so each weight owns one
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms, dtype=torch.int32, device=device)


@torch.no_grad()
def marlin_weight(q: QuantizedWeight, dtype: torch.dtype, device: torch.device | None = None) -> MarlinWeight:
    """Convert a :class:`QuantizedWeight` to a :class:`MarlinWeight` reporting ``dtype``.

    Raises:
        ConfigError: The format, zero points or group size have no Marlin kernel.
    """
    reason = _load_ops()
    if reason is not None:
        raise ConfigError(reason)
    device = device or q.codes.device
    size_n, size_k = q.codes.shape
    compute = dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    per_channel = q.group_size >= size_k
    group = -1 if per_channel else q.group_size
    fp8 = q.codes.dtype == torch.float8_e4m3fn
    bits = 8 if fp8 else q.grid.bits
    symmetric = fp8 or (q.zero is not None and bool((q.zero == 2 ** (bits - 1)).all()))
    reason = marlin_layout_error(q.grid.name, None if per_channel else q.group_size, symmetric)
    if reason is None and q.g_idx is not None:
        reason = "Marlin is given contiguous groups only (no g_idx): quantize with static_groups=True"
    if reason is not None:
        raise ConfigError(reason)
    type_name = MARLIN_FORMATS.get(q.grid.name).type_name(symmetric)
    padded_n, padded_k = marlin_padded_nk(size_n, size_k, group)
    layout = MarlinLayout(size_n, size_k, padded_n, padded_k, type_name, bits, group, compute)

    codes = q.codes.to(device)
    if fp8:
        codes = codes.view(torch.uint8)
    pad_n, pad_k = padded_n - size_n, padded_k - size_k
    if pad_n or pad_k:  # code 0 with scale 0 (padded outputs) or zero activations (padded inputs) adds nothing
        codes = F.pad(codes, (0, pad_k, 0, pad_n))
    packed = _gptq_pack(codes, bits)
    qweight = torch.ops._C.gptq_marlin_repack(packed, padded_k, padded_n, bits, False)

    scale = q.scale.to(device=device, dtype=torch.float32).T  # [groups, n]
    groups = 1 if per_channel else padded_k // group
    scale = F.pad(scale, (0, pad_n, 0, groups - scale.shape[0]))
    scales = _permute_scales(scale.to(compute), padded_n, grouped=not per_channel)
    if fp8:  # the kernel converts e4m3 to the compute dtype by a shift; the exponent-bias difference goes in the scale
        target = 8 if compute == torch.bfloat16 else 5
        bias = 2 ** (target - 1) - 2 ** (_FP8_MAX_EXPONENT - 1)
        scales = scales * torch.tensor(2.0, dtype=compute, device=device) ** bias

    zeros: Tensor | None = None
    if type_name == "uint4":
        assert q.zero is not None
        zero = q.zero.to(device).round().to(torch.int32).T  # [groups, n]
        zero = F.pad(zero, (0, pad_n, 0, groups - zero.shape[0]))
        zeros = _permute_zero_points(zero, padded_n, bits)
    elif not fp8:
        zeros = torch.empty(0, dtype=torch.int32, device=device)
    return MarlinWeight(qweight, scales, zeros, _workspace(device), layout, dtype)


# ---- the backend -------------------------------------------------------------------------------------------------


@QUANT_BACKENDS.register("marlin")
class MarlinBackend(QuantBackend):
    """vLLM's Marlin weight-only GEMM (``pip install vllm``; tested with vLLM 0.30, A40 and A100)."""

    def check_config(self, config: QuantSpec) -> None:
        reason = marlin_layout_error(config.format, config.group_size, config.grid.symmetric)
        if reason is not None:
            raise ConfigError(reason)

    def check(self, modules: Mapping[str, nn.Linear], config: QuantSpec) -> None:
        reason = _load_ops()
        if reason is not None:
            raise ConfigError(reason)
        whole_model = isinstance(config, QuantConfig)  # a drafter's fp32 maps may run in bf16; a model keeps its dtype
        for name, module in modules.items():
            weight = module.weight
            if whole_model and weight.dtype not in (torch.bfloat16, torch.float16):
                raise UnsupportedModelError(f"Marlin needs a bfloat16 or float16 model; {name} is {weight.dtype}")
            _check_device(weight.device, weight.dtype, name)

    def apply(self, module: nn.Linear, qweight: QuantizedWeight, config: QuantSpec) -> None:
        weight = module.weight
        module.weight = nn.Parameter(marlin_weight(qweight, weight.dtype, weight.device), requires_grad=False)
