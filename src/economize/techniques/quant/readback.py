"""Read the codes back out of a kernel weight, so that a model quantized for one kernel can be exported for another.

The ``"marlin"``, ``"machete"``, ``"cutlass"`` and ``"torchao"`` backends replace a module's weight by a kernel tensor
and keep no :class:`~.grid.QuantizedWeight`. Exporters that write codes (the compressed-tensors checkpoint vLLM
serves, :func:`economize.integrations.compressed_tensors.export_compressed_tensors`) recover them here instead of
asking the user to quantize again. A reader returns the codes, scales and zero points the backend was given; the
scales come back rounded to the kernel's compute dtype (bfloat16 or float16), since that is what the kernel stores
(the CUTLASS weight keeps fp32 scales).

  ======================  =========================================================================================
  weight                  how the codes are recovered
  ======================  =========================================================================================
  ``MarlinWeight``        scales and zero points un-permuted from Marlin's order, codes from the kernel's own
                          dequantized weight (``code = round(w / scale) + zero``)
  ``MacheteWeight``       scales as stored, zero points from ``-scale * zero`` (int4 only: the backend refuses
                          affine int8, whose bf16 shift loses the zero point), codes as for Marlin
  ``CutlassInt8Weight``   the stored int8 codes (shifted by 128) and fp32 per-channel scales
  torchao int8/fp8        the tensor's ``qdata``, ``scale`` and ``zero_point``
  torchao int4            tinygemm's dequantized weight (``_weight_int4pack_mm`` against an identity) and its
                          packed scales and zeros
  ======================  =========================================================================================

Rounding the kernel's weight recovers the integer codes exactly: a 16-bit product ``(code - zero) * scale`` is off by
at most ``|code - zero| * 2^-9`` scale units, below one half for every 4- and 8-bit layout. The GPU tests check the
round trip for every layout of each backend.

A new kernel tensor is supported by one :class:`CodeReader` registered in :data:`CODE_READERS` (entry-point group
``economize.code_readers``)::

    @CODE_READERS.register("my-kernel")
    class MyReader(CodeReader):
        def accepts(self, weight): return type(weight).__name__ == "MyKernelWeight"
        def read(self, weight): return QuantizedWeight(...)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from ...registry import Registry
from .grid import QuantizedWeight, make_grid

__all__ = ["CODE_READERS", "CodeReader", "read_quantized_weight", "reader_for"]


class CodeReader(ABC):
    """Recovers the :class:`QuantizedWeight` stored in one kind of kernel weight."""

    @abstractmethod
    def accepts(self, weight: Tensor) -> bool:
        """Whether ``weight`` is a tensor this reader understands."""

    @abstractmethod
    def read(self, weight: Tensor) -> QuantizedWeight:
        """The codes, scales and zero points of ``weight`` (on its device, without padding)."""


#: Reader classes of kernel weights (entry-point group ``economize.code_readers``). Each accepts only its own tensor
#: type, so the first match is the only one.
CODE_READERS: Registry[type[CodeReader]] = Registry("quantized-weight reader", "economize.code_readers")


def reader_for(module_or_weight: nn.Module | Tensor) -> CodeReader | None:
    """The reader of a module's (or a tensor's) kernel weight, ``None`` for a plain or unknown weight."""
    weight = getattr(module_or_weight, "weight", None) if isinstance(module_or_weight, nn.Module) else module_or_weight
    if not isinstance(weight, Tensor) or type(weight) in (Tensor, nn.Parameter):
        return None
    for name in CODE_READERS:
        reader = CODE_READERS.get(name)()
        if reader.accepts(weight):
            return reader
    return None


def read_quantized_weight(module_or_weight: nn.Module | Tensor) -> QuantizedWeight | None:
    """The :class:`QuantizedWeight` of a module's (or a tensor's) kernel weight, ``None`` for a plain weight."""
    reader = reader_for(module_or_weight)
    if reader is None:
        return None
    weight = module_or_weight.weight if isinstance(module_or_weight, nn.Module) else module_or_weight
    return reader.read(weight)


# ---- shared arithmetic -------------------------------------------------------------------------------------------


def _expand(per_group: Tensor, group_size: int, size_k: int) -> Tensor:
    return per_group.repeat_interleave(group_size, dim=1)[:, :size_k]


def _int_codes(weight: Tensor, scale: Tensor, offset: Tensor, group_size: int, bits: int) -> Tensor:
    """``round(weight / scale) + offset`` per group, clamped to the code range (zero scale: the offset)."""
    s = _expand(scale.float(), group_size, weight.shape[1])
    o = _expand(offset.float(), group_size, weight.shape[1])
    ratio = torch.where(s != 0, weight.float() / torch.where(s != 0, s, 1.0), 0.0)
    return (ratio.round() + o).round().clamp(0, 2**bits - 1).to(torch.uint8)


def _inverse(perm: list[int]) -> list[int]:
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


def _unpack(words: Tensor, bits: int) -> Tensor:
    """Inverse of :func:`~.marlin._pack`: int32 words to their ``32 / bits`` values along the last dimension."""
    per_word = 32 // bits
    shifts = torch.arange(per_word, device=words.device, dtype=torch.int64) * bits
    values = (words.to(torch.int64).unsqueeze(-1) >> shifts) & (2**bits - 1)
    return values.reshape(*words.shape[:-1], words.shape[-1] * per_word)


# ---- Marlin and Machete ------------------------------------------------------------------------------------------

#: Marlin / Machete weight type -> (economize format, symmetric).
_KERNEL_TYPES = {
    "uint4b8": ("int4", True),
    "uint4": ("int4", False),
    "uint8b128": ("int8", True),
    "uint8": ("int8", False),
    "float8_e4m3fn": ("fp8", True),
}


def _marlin_grouping(weight: Tensor) -> tuple[int, int]:
    """``(group_size, groups)`` of the unpadded weight (per channel: one group of all inputs)."""
    lay = weight.packing  # type: ignore[attr-defined]
    if lay.group_size < 0:
        return lay.size_k, 1
    return lay.group_size, -(-lay.size_k // lay.group_size)


@CODE_READERS.register("machete")
class MacheteReader(CodeReader):
    """:class:`~.machete.MacheteWeight`: ``w = code * scale + zeros`` with ``zeros = -scale * zero`` (affine)."""

    def accepts(self, weight: Tensor) -> bool:
        from .machete import MacheteWeight

        return type(weight) is MacheteWeight

    @torch.no_grad()
    def read(self, weight: Tensor) -> QuantizedWeight:
        lay = weight.packing  # type: ignore[attr-defined]
        fmt, symmetric = _KERNEL_TYPES[lay.type_name]
        group_size, groups = _marlin_grouping(weight)
        scale = weight.scales[:groups, : lay.size_n].T.float().contiguous()  # type: ignore[attr-defined]
        dense = weight.dequantize().float()  # type: ignore[attr-defined]
        if symmetric:
            zero = torch.full_like(scale, 2 ** (lay.bits - 1))
            codes = _int_codes(dense, scale, zero, group_size, lay.bits)
        else:
            shift = weight.zeros[:groups, : lay.size_n].T.float()  # type: ignore[attr-defined]  # -scale * zero
            safe = torch.where(scale != 0, scale, 1.0)
            zero = torch.where(scale != 0, (-shift / safe).round(), 0.0).clamp(0, 2**lay.bits - 1)
            ratio = (dense - _expand(shift, group_size, lay.size_k)) / _expand(safe, group_size, lay.size_k)
            nonzero = _expand(scale, group_size, lay.size_k) != 0
            ratio = torch.where(nonzero, ratio, _expand(zero, group_size, lay.size_k))
            codes = ratio.round().clamp(0, 2**lay.bits - 1).to(torch.uint8)
        return QuantizedWeight(make_grid(fmt, symmetric), codes, scale, zero, group_size)


@CODE_READERS.register("cutlass")
class CutlassReader(CodeReader):
    """:class:`~.cutlass.CutlassInt8Weight`: symmetric int8 codes and fp32 scales, stored as they are."""

    def accepts(self, weight: Tensor) -> bool:
        from .cutlass import CutlassInt8Weight

        return type(weight) is CutlassInt8Weight

    @torch.no_grad()
    def read(self, weight: Tensor) -> QuantizedWeight:
        lay = weight.packing  # type: ignore[attr-defined]
        stored = weight.qweight[: lay.size_k, : lay.size_n].T  # type: ignore[attr-defined]
        codes = (stored.to(torch.int16) + 128).to(torch.uint8)
        scale = weight.scales[0, : lay.size_n, None].contiguous()  # type: ignore[attr-defined]
        zero = torch.full_like(scale, 128)
        return QuantizedWeight(make_grid("int8", True), codes.contiguous(), scale, zero, lay.size_k)


@CODE_READERS.register("marlin")
class MarlinReader(CodeReader):
    """:class:`~.marlin.MarlinWeight`: scales (and uint4 zero points) in Marlin's permuted order, ``w = (code -
    zero) * scale``; fp8 scales carry the exponent-bias factor of the kernel's e4m3 conversion."""

    def accepts(self, weight: Tensor) -> bool:
        from .marlin import MarlinWeight

        return type(weight) is MarlinWeight

    @torch.no_grad()
    def read(self, weight: Tensor) -> QuantizedWeight:
        from .marlin import _FP8_MAX_EXPONENT, _scale_perms

        lay = weight.packing  # type: ignore[attr-defined]
        fmt, symmetric = _KERNEL_TYPES[lay.type_name]
        group_size, groups = _marlin_grouping(weight)
        grouped_perm, single_perm = _scale_perms()
        perm = single_perm if lay.group_size < 0 else grouped_perm
        scales = weight.scales  # type: ignore[attr-defined]
        scales = scales.reshape(-1, len(perm))[:, _inverse(perm)].reshape(-1, lay.padded_n).float()
        if fmt == "fp8":
            target = 8 if lay.compute_dtype == torch.bfloat16 else 5
            scales = scales / 2.0 ** (2 ** (target - 1) - 2 ** (_FP8_MAX_EXPONENT - 1))
        scale = scales[:groups, : lay.size_n].T.contiguous()
        dense = weight.dequantize().float()  # type: ignore[attr-defined]
        if fmt == "fp8":
            s = _expand(scale, group_size, lay.size_k)
            codes = torch.where(s != 0, dense / torch.where(s != 0, s, 1.0), 0.0).to(torch.float8_e4m3fn)
            return QuantizedWeight(make_grid(fmt), codes, scale, None, group_size)
        if symmetric:
            zero = torch.full_like(scale, 2 ** (lay.bits - 1))
        else:
            interleave = [0, 2, 4, 6, 1, 3, 5, 7] if lay.bits == 4 else [0, 2, 1, 3]
            z = _unpack(weight.zeros, lay.bits)  # type: ignore[attr-defined]
            z = z.reshape(-1, len(interleave))[:, _inverse(interleave)]
            z = z.reshape(-1, len(grouped_perm))[:, _inverse(grouped_perm)].reshape(-1, lay.padded_n)
            zero = z[:groups, : lay.size_n].T.float().contiguous()
        codes = _int_codes(dense, scale, zero, group_size, lay.bits)
        return QuantizedWeight(make_grid(fmt, symmetric), codes, scale, zero, group_size)


# ---- torchao -----------------------------------------------------------------------------------------------------


def _is_torchao(weight: Tensor, name: str) -> bool:
    cls = type(weight)
    return cls.__name__ == name and cls.__module__.startswith("torchao.")


def _group_size(weight: Tensor, size_k: int) -> int:
    block = getattr(weight, "block_size", None)
    return int(block[-1]) if block else size_k


@CODE_READERS.register("torchao-int8")
class TorchaoInt8Reader(CodeReader):
    """torchao ``Int8Tensor``: signed codes ``qdata`` (economize's minus 128) and an optional signed zero point."""

    def accepts(self, weight: Tensor) -> bool:
        return _is_torchao(weight, "Int8Tensor")

    @torch.no_grad()
    def read(self, weight: Tensor) -> QuantizedWeight:
        qdata = weight.qdata  # type: ignore[attr-defined]
        size_n, size_k = qdata.shape
        group_size = _group_size(weight, size_k)
        scale = weight.scale.float().reshape(size_n, -1).contiguous()  # type: ignore[attr-defined]
        zero_point = getattr(weight, "zero_point", None)
        zero = torch.full_like(scale, 128) if zero_point is None else zero_point.float().reshape(size_n, -1) + 128
        codes = (qdata.to(torch.int16) + 128).to(torch.uint8)
        return QuantizedWeight(make_grid("int8", zero_point is None), codes, scale, zero, group_size)


@CODE_READERS.register("torchao-fp8")
class TorchaoFp8Reader(CodeReader):
    """torchao ``Float8Tensor``: e4m3 codes ``qdata`` and float32 scales."""

    def accepts(self, weight: Tensor) -> bool:
        return _is_torchao(weight, "Float8Tensor")

    @torch.no_grad()
    def read(self, weight: Tensor) -> QuantizedWeight:
        qdata = weight.qdata  # type: ignore[attr-defined]
        size_n, size_k = qdata.shape
        scale = weight.scale.float().reshape(size_n, -1).contiguous()  # type: ignore[attr-defined]
        return QuantizedWeight(make_grid("fp8"), qdata, scale, None, _group_size(weight, size_k))


@CODE_READERS.register("torchao-int4")
class TorchaoInt4Reader(CodeReader):
    """torchao ``Int4TilePackedTo4dTensor`` (tinygemm): ``w = (code - 8) * scale + zero_f`` with ``zero_f = (8 -
    zero) * scale``; inputs padded to 1024, outputs to 8."""

    def accepts(self, weight: Tensor) -> bool:
        return _is_torchao(weight, "Int4TilePackedTo4dTensor")

    @torch.no_grad()
    def read(self, weight: Tensor) -> QuantizedWeight:
        size_n, size_k = (int(d) for d in weight.shape)
        group_size = _group_size(weight, size_k)
        scale_and_zero = weight.scale_and_zero  # type: ignore[attr-defined]  # [groups', n', 2]
        padded_k = scale_and_zero.shape[0] * group_size
        groups = -(-size_k // group_size)
        rows = []
        for start in range(0, size_k, 2048):  # identity rows through the kernel, a chunk at a time
            stop = min(start + 2048, size_k)
            eye = torch.zeros(stop - start, padded_k, dtype=torch.bfloat16, device=scale_and_zero.device)
            index = torch.arange(stop - start, device=eye.device)
            eye[index, index + start] = 1
            rows.append(torch.ops.aten._weight_int4pack_mm(eye, weight.qdata, group_size, scale_and_zero))  # type: ignore[attr-defined]
        dense = torch.cat(rows)[:, :size_n].T.float()
        scale = scale_and_zero[:groups, :size_n, 0].T.float().contiguous()
        shift = scale_and_zero[:groups, :size_n, 1].T.float()
        safe = torch.where(scale != 0, scale, 1.0)
        zero = torch.where(scale != 0, 8 - (shift / safe).round(), 8.0).clamp(0, 15)
        ratio = (dense - _expand(shift, group_size, size_k)) / _expand(safe, group_size, size_k)
        codes = (ratio.round() + 8).clamp(0, 15).to(torch.uint8)
        return QuantizedWeight(make_grid("int4"), codes, scale, zero.contiguous(), group_size)
