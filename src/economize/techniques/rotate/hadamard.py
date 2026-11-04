"""Walsh-Hadamard matrices and transforms (the rotations of QuaRot, Ashkboos et al., NeurIPS 2024).

The matrix is the Sylvester construction ``H_1 = [1]``, ``H_2n = [[H_n, H_n], [H_n, -H_n]]`` divided by ``sqrt(n)``:
orthogonal and symmetric, so it is its own inverse. It is the matrix compressed-tensors'
``deterministic_hadamard_matrix`` builds (before normalization) and the one vLLM's ``hadacore_transform`` kernel
applies, so a model rotated here runs with the same online transform on vLLM.

A transform of width ``n`` that is not a power of two is block-diagonal: ``n / b`` copies of ``H_b`` with ``b`` the
largest power of two dividing ``n`` (vLLM's ``head_dim`` transforms). :data:`HADAMARD_KERNELS` holds the kernels the
Hugging Face path runs ``x @ H`` with.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import torch
from torch import Tensor

from ...errors import ConfigError
from ...registry import Registry

__all__ = [
    "AUTO_ORDER",
    "HADAMARD_KERNELS",
    "DaoHadamardKernel",
    "HadacoreHadamardKernel",
    "HadamardKernel",
    "TorchHadamardKernel",
    "block_hadamard",
    "hadamard_block",
    "hadamard_transform",
    "is_power_of_two",
    "sylvester",
]


def is_power_of_two(n: int) -> bool:
    """Whether ``n`` is a positive power of two."""
    return n > 0 and n & (n - 1) == 0


def sylvester(n: int, *, normalized: bool = True, dtype: torch.dtype = torch.float32) -> Tensor:
    """The ``n x n`` Sylvester-Hadamard matrix.

    Args:
        n: Its size, a power of two.
        normalized: Divide by ``sqrt(n)`` (orthogonal); otherwise the entries are ``+-1``.
        dtype: Its dtype.

    Returns:
        The matrix, symmetric; normalized it is its own inverse.

    Raises:
        ConfigError: ``n`` is not a power of two.
    """
    if not is_power_of_two(n):
        raise ConfigError(f"a Sylvester-Hadamard matrix needs a power-of-two size, got {n}")
    matrix = torch.ones(1, 1, dtype=torch.float64)
    while matrix.shape[0] < n:
        matrix = torch.cat([torch.cat([matrix, matrix], 1), torch.cat([matrix, -matrix], 1)], 0)
    if normalized:
        matrix = matrix / math.sqrt(n)
    return matrix.to(dtype)


def hadamard_block(n: int, cap: int = 0) -> int:
    """The block size of the block-diagonal Hadamard transform of width ``n``.

    Args:
        n: The transform's width.
        cap: Largest allowed block, a power of two; ``0`` for no cap.

    Returns:
        The largest power of two that divides ``n`` (and is at most ``cap``).

    Raises:
        ConfigError: ``cap`` is neither 0 nor a power of two.
    """
    if cap and not is_power_of_two(cap):
        raise ConfigError(f"the Hadamard block cap must be a power of two, got {cap}")
    block = n & -n
    return min(block, cap) if cap else block


def block_hadamard(n: int, block: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    """The ``n x n`` block-diagonal matrix of ``n / block`` normalized Sylvester blocks (a dense reference)."""
    return torch.block_diag(*[sylvester(block, dtype=dtype)] * (n // block))


class HadamardKernel:
    """A kernel that computes ``x @ H`` along the last dimension of ``x``, with ``H`` block-diagonal of
    normalized Sylvester blocks of size ``block``."""

    def available(self, x: Tensor, block: int) -> bool:
        """Whether this kernel can transform ``x`` in blocks of ``block``."""
        return True

    def __call__(self, x: Tensor, block: int) -> Tensor:
        """Return ``x @ H`` in the dtype of ``x``."""
        raise NotImplementedError


class TorchHadamardKernel(HadamardKernel):
    """The fast Walsh-Hadamard transform in ``log2(block)`` butterfly stages of plain torch ops, in fp32."""

    def __call__(self, x: Tensor, block: int) -> Tensor:
        shape = x.shape
        work = x.float().reshape(-1, block)
        half = 1
        while half < block:
            pairs = work.view(-1, block // (2 * half), 2, half)
            even, odd = pairs[:, :, 0], pairs[:, :, 1]
            work = torch.stack((even + odd, even - odd), dim=2).view(-1, block)
            half *= 2
        return (work * (1.0 / math.sqrt(block))).view(shape).to(x.dtype)


class DaoHadamardKernel(HadamardKernel):
    """Dao AI Lab's CUDA kernel (``pip install fast-hadamard-transform``), imported on first use."""

    module: ClassVar[Any] = None

    def available(self, x: Tensor, block: int) -> bool:
        if not x.is_cuda:
            return False
        if DaoHadamardKernel.module is None:
            try:
                import fast_hadamard_transform
            except ImportError:
                return False
            DaoHadamardKernel.module = fast_hadamard_transform
        return True

    def __call__(self, x: Tensor, block: int) -> Tensor:
        if not self.available(x, block):
            raise ConfigError("the fast_hadamard_transform kernel needs a CUDA tensor and the fast-hadamard-transform "
                              "package")
        shape = x.shape
        out = DaoHadamardKernel.module.hadamard_transform(x.reshape(-1, block), scale=1.0 / math.sqrt(block))
        return out.view(shape)


class HadacoreHadamardKernel(HadamardKernel):
    """vLLM's HadaCore kernel (``vllm._custom_ops.hadacore_transform``, tensor cores, sm80+), imported on first use.

    It transforms the last dimension with the normalized Sylvester matrix, the transform vLLM's compressed-tensors
    ``HadamardTransform`` applies; blocks are separated by a reshape. Blocks below 64 come out wrong and blocks above
    32768 are refused (vLLM 0.30), and ``inplace=False`` returns garbage, so the kernel runs in place on a copy.
    """

    module: ClassVar[Any] = None
    blocks: ClassVar[tuple[int, int]] = (64, 32768)

    def available(self, x: Tensor, block: int) -> bool:
        low, high = self.blocks
        if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16) or not low <= block <= high:
            return False
        if HadacoreHadamardKernel.module is None:
            try:
                from vllm import _custom_ops
            except ImportError:
                return False
            if not hasattr(torch.ops._C, "hadacore_transform"):
                return False
            HadacoreHadamardKernel.module = _custom_ops
        return True

    def __call__(self, x: Tensor, block: int) -> Tensor:
        if not self.available(x, block):
            raise ConfigError(f"the hadacore kernel needs a CUDA fp16 / bf16 tensor, a block in {self.blocks} and "
                              "vLLM's compiled ops")
        out = x.reshape(-1, block).clone()
        HadacoreHadamardKernel.module.hadacore_transform(out, inplace=True)
        return out.view(x.shape)


#: Kernels of ``x @ H`` by name.
HADAMARD_KERNELS: Registry[HadamardKernel] = Registry("Hadamard kernel", "economize.hadamard_kernels")
HADAMARD_KERNELS.register("fast_hadamard_transform", DaoHadamardKernel())
HADAMARD_KERNELS.register("hadacore", HadacoreHadamardKernel())
HADAMARD_KERNELS.register("torch", TorchHadamardKernel())
#: The kernels ``"auto"`` tries first, in this order (then every other registered kernel, by name).
AUTO_ORDER: tuple[str, ...] = ("fast_hadamard_transform", "hadacore", "torch")


def hadamard_transform(x: Tensor, block: int, kernel: str = "auto") -> Tensor:
    """``x @ H`` along the last dimension, ``H`` block-diagonal of normalized Sylvester blocks of size ``block``.

    Args:
        x: The input; its last dimension is a multiple of ``block``.
        block: The block size, a power of two.
        kernel: A name in :data:`HADAMARD_KERNELS`, or ``"auto"`` for the first one that runs on ``x``.

    Returns:
        The transformed tensor, in the dtype of ``x``.

    Raises:
        ConfigError: The last dimension of ``x`` is not a multiple of ``block``.
    """
    if x.shape[-1] % block:
        raise ConfigError(f"a Hadamard block of {block} does not divide the width {x.shape[-1]}")
    if kernel != "auto":
        return HADAMARD_KERNELS.get(kernel)(x, block)
    others = [name for name in HADAMARD_KERNELS.names() if name not in AUTO_ORDER]
    for name in (*[name for name in AUTO_ORDER if name in HADAMARD_KERNELS], *others):
        chosen = HADAMARD_KERNELS.get(name)
        if chosen.available(x, block):
            return chosen(x, block)
    raise ConfigError("no registered Hadamard kernel runs on this tensor")  # pragma: no cover - torch always runs
