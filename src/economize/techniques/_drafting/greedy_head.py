"""Greedy draft heads: the next draft token and its probability straight from the head input, without logits.

A greedy draft step needs two numbers per row from its LM head: the arg-max token and (for confidence logging) its
softmax probability. The plain head writes all ``V`` logits, applies the family's transform (Gemma's softcap) in
separate kernels, casts them to float32 and reduces them again. :func:`greedy_select` fuses the matmul with both
reductions in Triton: every program owns ``BLOCK_N`` rows of the head weight, multiplies them with the head input in
float32, applies the transform and keeps only its block's arg-max and softmax partials (maximum and sum of
exponentials). A second kernel combines the blocks of every row. The logits never reach memory.

The kernel reproduces the eager head's rounding: with a 16-bit head it rounds every logit to the head's dtype and then
the result of every operation of the transform, as PyTorch's 16-bit kernels do (a division by a Python float is a
multiplication by its float32 reciprocal there). The arg-max is taken over these values with the smallest index
winning ties, like :func:`torch.argmax`, so the token equals the eager one whenever the two matmuls round every logit
alike (the float32 summation order is their only difference). A restricted head (a draft vocabulary, ``token_ids``)
maps the winning row to its model token id; its rows are kept in increasing id order, so that ties go to the
smallest id as in the full-vocabulary logits a restricted head is widened to.

* :class:`GreedyHead`: a draft head that returns ``(tokens, top probabilities)`` for draft-path states.
* :class:`FusedGreedyHead`: the fused kernel over a plain linear head (PyTorch off CUDA).
* :class:`LogitsGreedyHead`: arg-max of any logits function (for heads the kernel cannot express).
* :func:`adapter_greedy_head`: the fused head of a model's final norm and LM head, when its transform is registered
  in :data:`LOGIT_TRANSFORMS` (a new family registers its own by extension).

The kernels never synchronize with the host and allocate only small partial buffers, so they run inside CUDA graphs
(:mod:`.chain`).
"""

from __future__ import annotations

import functools
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ...errors import ConfigError
from .head import LinearSlice

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = [
    "IDENTITY",
    "LOGIT_TRANSFORMS",
    "MANY_ROWS",
    "MANY_ROWS_FP32",
    "ONE_ROW",
    "FusedGreedyHead",
    "GreedyHead",
    "HeadKernelConfig",
    "LogitTransform",
    "LogitsGreedyHead",
    "adapter_greedy_head",
    "greedy_select",
    "greedy_select_reference",
    "register_logit_transform",
]

_ROUNDING = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}


def triton_available() -> bool:
    """Whether Triton is importable and a CUDA device is present."""
    return importlib.util.find_spec("triton") is not None and torch.cuda.is_available()


@dataclass(frozen=True)
class LogitTransform:
    """The increasing per-logit transform ``z = tanh((l / divisor) / cap) * cap`` (each step optional), written with
    the same operations as the families' ``transform_logits``.

    Attributes:
        divisor: Positive divisor of the raw logit (Granite's ``logits_scaling``), ``None`` for none.
        cap: Softcap (Gemma's ``final_logit_softcapping``), ``None`` for none.
    """

    divisor: float | None = None
    cap: float | None = None

    def __post_init__(self) -> None:
        if any(v is not None and not v > 0 for v in (self.divisor, self.cap)):
            raise ConfigError(f"a LogitTransform must be increasing, got divisor={self.divisor}, cap={self.cap}")

    def __call__(self, logits: Tensor) -> Tensor:
        z = logits if self.divisor is None else logits / self.divisor
        return z if self.cap is None else torch.tanh(z / self.cap) * self.cap


#: The identity transform (the plain linear head).
IDENTITY = LogitTransform()

#: ``transform_logits`` implementations the fused head can express, keyed by the function object (the
#: ``type(adapter).transform_logits`` of an adapter), each mapped to the :class:`LogitTransform` of an adapter.
LOGIT_TRANSFORMS: dict[Callable[..., torch.Tensor], Callable[[ArchAdapter], LogitTransform]] = {}


def register_logit_transform(
    method: Callable[..., torch.Tensor],
) -> Callable[[Callable[[ArchAdapter], LogitTransform]], Callable[[ArchAdapter], LogitTransform]]:
    """Decorator: register the :class:`LogitTransform` of the ``transform_logits`` implementation ``method``.

    Example (a family whose logits are halved)::

        @register_logit_transform(HalvedAdapter.transform_logits)
        def _halved(adapter):
            return LogitTransform(divisor=2.0)
    """

    def register(fn: Callable[[ArchAdapter], LogitTransform]) -> Callable[[ArchAdapter], LogitTransform]:
        LOGIT_TRANSFORMS[method] = fn
        return fn

    return register


def _softcapped(adapter: ArchAdapter) -> LogitTransform:
    """The transform of :class:`~economize.arch.SoftcappedLogitsMixin`, registered for its ``transform_logits``
    (so ``adapter`` is one)."""
    from ...arch.standard import SoftcappedLogitsMixin

    return LogitTransform(cap=cast(SoftcappedLogitsMixin, adapter).final_logit_softcap)


@functools.cache
def _register_builtin_transforms() -> None:
    from ...arch import ArchAdapter
    from ...arch.families.granite import GraniteAdapter
    from ...arch.standard import SoftcappedLogitsMixin

    LOGIT_TRANSFORMS.setdefault(ArchAdapter.transform_logits, lambda adapter: LogitTransform())
    LOGIT_TRANSFORMS.setdefault(SoftcappedLogitsMixin.transform_logits, _softcapped)
    LOGIT_TRANSFORMS.setdefault(GraniteAdapter.transform_logits,
                                lambda adapter: LogitTransform(divisor=adapter.config.logits_scaling))


class GreedyHead(Protocol):
    """A draft head for greedy drafting."""

    def select(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Arg-max tokens ``[N]`` (model token ids, int64) and their softmax probabilities ``[N]`` (float32) for
        draft-path states ``[..., d]`` (``N`` = the product of the leading dims)."""
        ...


@dataclass(frozen=True)
class HeadKernelConfig:
    """Launch parameters of :func:`greedy_select`.

    Attributes:
        block_n: Head rows (tokens) per program.
        block_k: Input channels per inner step.
        num_warps: Warps per program.
        num_stages: Software pipelining depth of the inner loop.
    """

    block_n: int = 64
    block_k: int = 256
    num_warps: int = 4
    num_stages: int = 3


#: One input row (a draft step at batch size 1): a broadcast multiply-reduce per program. The head is bound by the
#: weight's bandwidth there; on an L40 (262144 x 5376 bfloat16) the configurations of a 3 x 3 x 3 x 3 grid lie within
#: 2% of each other.
ONE_ROW = HeadKernelConfig(block_n=64, block_k=256, num_warps=4, num_stages=3)
#: Several input rows: ``tl.dot`` over blocks of 16 rows (on an L40 within 2% of the best of the same grid at 4 and
#: 16 rows).
MANY_ROWS = HeadKernelConfig(block_n=64, block_k=256, num_warps=4, num_stages=3)
#: Several float32 rows: half the input channels per step, so that the staged tiles fit in shared memory.
MANY_ROWS_FP32 = HeadKernelConfig(block_n=64, block_k=128, num_warps=4, num_stages=2)


@functools.cache
def _kernels() -> tuple[Any, Any]:
    """The two Triton kernels, compiled on first use (Triton is imported lazily)."""
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _round(x, ROUND: tl.constexpr):
        if ROUND == 1:
            x = x.to(tl.bfloat16).to(tl.float32)
        elif ROUND == 2:
            x = x.to(tl.float16).to(tl.float32)
        return x

    @triton.jit
    def head_blocks_kernel(
        x_ptr, w_ptr, b_ptr, zmax_ptr, zarg_ptr, zsum_ptr,
        M, N, D, stride_xm, stride_wn, num_blocks, inv_div, inv_cap, cap,
        HAS_BIAS: tl.constexpr, HAS_DIV: tl.constexpr, HAS_CAP: tl.constexpr, ROUND: tl.constexpr,
        PRECISION: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_ok = offs_n < N
        m_ok = offs_m < M
        if BLOCK_M == 1:
            row = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for k0 in range(0, D, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                k_ok = offs_k < D
                x = tl.load(x_ptr + pid_m * stride_xm + offs_k, mask=k_ok, other=0.0).to(tl.float32)
                w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                            mask=n_ok[:, None] & k_ok[None, :], other=0.0)
                row += tl.sum(w.to(tl.float32) * x[None, :], axis=1)
            acc = row[None, :]
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, D, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                k_ok = offs_k < D
                xs = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                             mask=m_ok[:, None] & k_ok[None, :], other=0.0)
                w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                            mask=n_ok[:, None] & k_ok[None, :], other=0.0)
                acc = tl.dot(xs.to(w.dtype), tl.trans(w), acc, input_precision=PRECISION)
        if HAS_BIAS:
            acc += tl.load(b_ptr + offs_n, mask=n_ok, other=0.0).to(tl.float32)[None, :]
        z = _round(acc, ROUND)
        if HAS_DIV:
            z = _round(z * inv_div, ROUND)
        if HAS_CAP:
            z = _round(_round(libdevice.tanh(_round(z * inv_cap, ROUND)), ROUND) * cap, ROUND)
        z = tl.where(n_ok[None, :], z, float("-inf"))
        z_max = tl.max(z, axis=1)
        z_arg = tl.argmax(z, axis=1, tie_break_left=True)
        z_sum = tl.sum(tl.exp(z - z_max[:, None]), axis=1)
        out = offs_m * num_blocks + pid_n
        tl.store(zmax_ptr + out, z_max, mask=m_ok)
        tl.store(zarg_ptr + out, pid_n * BLOCK_N + z_arg, mask=m_ok)
        tl.store(zsum_ptr + out, z_sum, mask=m_ok)

    @triton.jit
    def head_reduce_kernel(
        zmax_ptr, zarg_ptr, zsum_ptr, ids_ptr, token_ptr, top_ptr, num_blocks,
        HAS_IDS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        base = row * num_blocks
        best = float("-inf")
        best_at = 0
        total = 0.0
        for b0 in range(0, num_blocks, BLOCK):
            offs = b0 + tl.arange(0, BLOCK)
            ok = offs < num_blocks
            maxes = tl.load(zmax_ptr + base + offs, mask=ok, other=float("-inf"))
            sums = tl.load(zsum_ptr + base + offs, mask=ok, other=0.0)
            chunk_max = tl.max(maxes, axis=0)
            chunk_at = b0 + tl.argmax(maxes, axis=0, tie_break_left=True)
            chunk_sum = tl.sum(tl.where(ok, sums * tl.exp(maxes - chunk_max), 0.0), axis=0)
            better = chunk_max > best  # strict: an earlier chunk keeps a tie
            new_max = tl.maximum(best, chunk_max)
            total = total * tl.exp(best - new_max) + chunk_sum * tl.exp(chunk_max - new_max)
            best_at = tl.where(better, chunk_at, best_at)
            best = new_max
        index = tl.load(zarg_ptr + base + best_at)
        if HAS_IDS:  # noqa: SIM108 (a branch on a constexpr, resolved at compile time)
            token = tl.load(ids_ptr + index)
        else:
            token = index.to(tl.int64)
        tl.store(token_ptr + row, token)
        tl.store(top_ptr + row, 1.0 / total)

    return head_blocks_kernel, head_reduce_kernel


def greedy_select_reference(
    x: Tensor, weight: Tensor, bias: Tensor | None = None, *, transform: LogitTransform = IDENTITY,
    token_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """:func:`greedy_select` in PyTorch, as the eager head computes it (the CPU path and the tests' reference)."""
    logits = transform(F.linear(x.to(weight.dtype), weight, bias)).float()
    index = logits.argmax(-1)
    tokens = index if token_ids is None else token_ids.to(index.device)[index]
    return tokens, torch.softmax(logits, -1).amax(-1)


def _reciprocal(value: float | None) -> float:
    """``1 / value`` in float32 (how PyTorch divides a tensor by a Python float on CUDA)."""
    return 1.0 if value is None else float(np.float32(1.0) / np.float32(value))


def greedy_select(
    x: Tensor, weight: Tensor, bias: Tensor | None = None, *, transform: LogitTransform = IDENTITY,
    token_ids: Tensor | None = None, config: HeadKernelConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Arg-max token and its softmax probability of ``transform(x @ weight.T + bias)`` for every row of ``x``.

    Args:
        x: Head inputs ``[B, d]`` (after the final norm); cast to the weight's dtype, as ``F.linear`` would need.
        weight: Head weight ``[K, d]`` (float32, bfloat16 or float16).
        bias: Optional head bias ``[K]``.
        transform: The family's per-logit transform.
        token_ids: Model token id of every head row ``[K]`` (int64, a draft vocabulary), ``None`` for the identity.
        config: Launch parameters (default: :data:`ONE_ROW` for one row, :data:`MANY_ROWS` otherwise).

    Returns:
        ``(tokens [B] int64, top [B] float32)``, as :func:`greedy_select_reference` up to the matmul's summation
        order.
    """
    if x.device.type != "cuda" or not triton_available() or weight.dtype not in _ROUNDING:
        return greedy_select_reference(x, weight, bias, transform=transform, token_ids=token_ids)
    import triton

    x = x.to(weight.dtype)
    if x.stride(-1) != 1:
        x = x.contiguous()
    if weight.stride(-1) != 1:
        weight = weight.contiguous()
    rows, dim = x.shape
    vocab = weight.shape[0]
    block_m = 1 if rows == 1 else 16
    cfg = config or (ONE_ROW if rows == 1 else MANY_ROWS if weight.element_size() <= 2 else MANY_ROWS_FP32)
    num_blocks = triton.cdiv(vocab, cfg.block_n)
    device = x.device
    zmax = torch.empty(rows, num_blocks, dtype=torch.float32, device=device)
    zarg = torch.empty(rows, num_blocks, dtype=torch.int32, device=device)
    zsum = torch.empty_like(zmax)
    blocks, reduce = _kernels()
    blocks[(num_blocks, triton.cdiv(rows, block_m))](
        x, weight, weight if bias is None else bias, zmax, zarg, zsum,
        rows, vocab, dim, x.stride(0), weight.stride(0), num_blocks,
        _reciprocal(transform.divisor), _reciprocal(transform.cap), float(transform.cap or 1.0),
        HAS_BIAS=bias is not None, HAS_DIV=transform.divisor is not None, HAS_CAP=transform.cap is not None,
        ROUND=_ROUNDING[weight.dtype], PRECISION="ieee",
        BLOCK_M=block_m, BLOCK_N=cfg.block_n, BLOCK_K=cfg.block_k,
        num_warps=cfg.num_warps, num_stages=cfg.num_stages,
    )
    tokens = torch.empty(rows, dtype=torch.long, device=device)
    top = torch.empty(rows, dtype=torch.float32, device=device)
    reduce[(rows,)](
        zmax, zarg, zsum, tokens if token_ids is None else token_ids, tokens, top, num_blocks,
        HAS_IDS=token_ids is not None, BLOCK=min(1024, triton.next_power_of_2(num_blocks)), num_warps=4,
    )
    return tokens, top


class FusedGreedyHead:
    """A plain linear head (after ``norm``) evaluated by :func:`greedy_select`.

    Args:
        norm: Maps draft-path states ``[..., d]`` to the head input (the final norm, a readout adapter, a cast).
        weight: Head weight ``[K, d]``.
        bias: Optional head bias ``[K]``.
        transform: The family's per-logit transform.
        token_ids: Model token id of every head row (a draft vocabulary), ``None`` for the identity. Ids out of
            increasing order are sorted once, together with a copy of the rows (ties then go to the smallest id).
        config: Launch parameters (``None``: by the number of rows).
    """

    def __init__(self, norm: Callable[[Tensor], Tensor], weight: Tensor, bias: Tensor | None = None, *,
                 transform: LogitTransform = IDENTITY, token_ids: Tensor | None = None,
                 config: HeadKernelConfig | None = None) -> None:
        self.norm, self.weight = norm, weight.detach()
        self.bias = None if bias is None else bias.detach()
        self.transform, self.config = transform, config
        self.token_ids = None
        if token_ids is not None:
            ids = token_ids.to(weight.device, torch.long)
            order = ids.argsort()
            if not bool((order == torch.arange(len(order), device=order.device)).all()):
                ids, self.weight = ids[order], self.weight[order]
                self.bias = None if self.bias is None else self.bias[order]
            self.token_ids = ids.contiguous()

    def _input(self, hidden: Tensor) -> Tensor:
        x = self.norm(hidden)
        return x.reshape(-1, x.shape[-1])

    def select(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        return greedy_select(self._input(hidden), self.weight, self.bias, transform=self.transform,
                             token_ids=self.token_ids, config=self.config)

    def reference(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """:meth:`select` computed by :func:`greedy_select_reference`."""
        return greedy_select_reference(self._input(hidden), self.weight, self.bias, transform=self.transform,
                                       token_ids=self.token_ids)


class LogitsGreedyHead:
    """Greedy selection from any float logits function ``[..., d] -> [..., V]`` (arg-max and softmax maximum), for
    heads the fused kernel cannot express (quantized LM heads, families that override ``ArchAdapter.logits``).

    Args:
        logits: The logits function.
        token_ids: Model token id of every logit column (a restricted head), ``None`` for the identity. Ties go to
            the first column.
    """

    def __init__(self, logits: Callable[[Tensor], Tensor], token_ids: Tensor | None = None) -> None:
        self.logits, self.token_ids = logits, token_ids

    def select(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.logits(hidden).float()
        logits = logits.reshape(-1, logits.shape[-1])
        index = logits.argmax(-1)
        tokens = index if self.token_ids is None else self.token_ids.to(index.device)[index]
        return tokens, torch.softmax(logits, -1).amax(-1)


def adapter_greedy_head(
    adapter: ArchAdapter, before: Callable[[Tensor], Tensor] | None = None, rows: LinearSlice | None = None,
    final_norm: bool = True,
) -> FusedGreedyHead | None:
    """The fused greedy head of ``adapter.logits`` (final norm, LM head, ``transform_logits``) applied after
    ``before`` (e.g. a readout adapter), restricted to the LM-head rows ``rows`` (a draft vocabulary) when given.
    Without ``final_norm`` the LM head reads ``before``'s output directly (EAGLE-1 drafts so).

    ``None`` when the kernel cannot express it: the family overrides :meth:`~economize.arch.ArchAdapter.logits`, its
    ``transform_logits`` is not in :data:`LOGIT_TRANSFORMS`, or the LM head is not a plain floating-point linear layer
    (a quantized head). The caller then uses a :class:`LogitsGreedyHead`.
    """
    from ...arch import ArchAdapter

    _register_builtin_transforms()
    lm_head = adapter.lm_head
    transform = LOGIT_TRANSFORMS.get(type(adapter).transform_logits)
    if type(adapter).logits is not ArchAdapter.logits or transform is None or not LinearSlice.supports(lm_head):
        return None
    norm, device = adapter.final_norm if final_norm else None, adapter.head_device
    first = before or (lambda hidden: hidden)

    def head_input(hidden: Tensor) -> Tensor:
        hidden = first(hidden).to(device)
        return hidden if norm is None else norm(hidden)

    if rows is None:
        return FusedGreedyHead(head_input, lm_head.weight, lm_head.bias, transform=transform(adapter))
    return FusedGreedyHead(head_input, rows.weight, rows.bias, transform=transform(adapter), token_ids=rows.token_ids)
