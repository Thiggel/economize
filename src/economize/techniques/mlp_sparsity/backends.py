"""Run-time backends of the gated-MLP channel sparsity techniques (``cats``, ``dejavu``).

A backend turns one MLP into two sparse operations, :meth:`MlpOps.rows` (a map whose dropped *outputs* need not
be computed: the up map, and for Deja Vu the gate map) and :meth:`MlpOps.cols` (a map whose dropped *inputs* are
exact zeros: the down map). ``dense`` computes both densely and masks, the numerical reference without a speedup
(it reports a :class:`~economize.Simulation`); ``triton`` runs the row-masked GEMV of :mod:`.kernel` and TEAL's
column-sparse GEMV for calls of up to 16 rows on 16-bit CUDA weights, and the dense path elsewhere; the default
``auto`` (:mod:`.auto_backend`) is ``triton`` that refuses an MLP the kernels cannot serve. New kernels are added as
further registered backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...core.metadata import Saving, Simulation
from ...registry import Registry
from ..sparsity.teal_gemv import MAX_ROWS, GemvConfig, teal_gemv, triton_available
from ..sparsity.triton_backend import kernel_eligible
from .kernel import RowGemvConfig, row_masked_gemv
from .mlp import GatedMlp, masked

__all__ = [
    "MLP_SPARSITY_BACKENDS",
    "DenseMlpBackend",
    "DenseOps",
    "KernelPlan",
    "MlpOps",
    "MlpSparsityBackend",
    "TritonMlpBackend",
    "TritonOps",
    "fallback_simulation",
]


C = TypeVar("C", RowGemvConfig, GemvConfig)


@dataclass(frozen=True)
class KernelPlan(Generic[C]):
    """How one map runs its kernel: the launch parameters by call size and the largest call the kernel takes.

    Attributes:
        one_row: Launch parameters of single-row calls (``None``: the kernel's heuristic).
        many_rows: Launch parameters of calls with 2 to :attr:`max_rows` rows (``None``: the kernel's heuristic).
        max_rows: Calls with more rows take the dense path (at most 16, the kernels' limit).
    """

    one_row: C | None = None
    many_rows: C | None = None
    max_rows: int = MAX_ROWS

    def config(self, rows: int) -> C | None:
        """The launch parameters of a call with ``rows`` rows."""
        return self.one_row if rows == 1 else self.many_rows


class MlpOps(ABC):
    """The sparse operations of one installed MLP.

    Attributes:
        realized: Whether the operations skip the dropped channels in every map they serve (``False``: some map is
            computed densely and masked, so its saving is only simulated).
    """

    realized: bool = True

    @abstractmethod
    def rows(self, x: Tensor, linear: nn.Module, keep: Tensor) -> Tensor:
        """``linear(x)`` with the outputs outside ``keep`` (boolean, ``[..., out]``) set to exactly 0."""

    @abstractmethod
    def cols(self, h: Tensor, linear: nn.Module) -> Tensor:
        """``linear(h)`` for an input whose dropped channels are exact zeros."""

    def remove(self) -> None:
        """Undo what the backend changed on the MLP's maps at install time (nothing by default)."""


class MlpSparsityBackend(ABC):
    """Builds the :class:`MlpOps` of an MLP.

    Attributes:
        simulation: Set when the backend computes the dense maps and only masks (the saving is not realized).
    """

    simulation: ClassVar[Simulation | None] = None

    @abstractmethod
    def ops(self, mlp: GatedMlp) -> MlpOps:
        """The operations of ``mlp`` (may prepare its maps; :meth:`MlpOps.remove` undoes it)."""

    def check(self, mlps: Mapping[int, GatedMlp]) -> None:
        """Raise a :class:`~economize.errors.ConfigError` when this backend cannot serve ``mlps`` (keyed by the
        1-indexed layer).

        The techniques call it before calibrating or fitting, so a backend that would fail at install time fails
        before the expensive part. The default accepts every MLP.
        """


MLP_SPARSITY_BACKENDS: Registry[type[MlpSparsityBackend]] = Registry(
    "MLP sparsity backend", "economize.mlp_sparsity_backends"
)


class DenseOps(MlpOps):
    """Dense maps, masked outputs."""

    realized = False

    def rows(self, x: Tensor, linear: nn.Module, keep: Tensor) -> Tensor:
        return masked(linear(x), keep)

    def cols(self, h: Tensor, linear: nn.Module) -> Tensor:
        return cast(Tensor, linear(h))


@MLP_SPARSITY_BACKENDS.register("dense")
class DenseMlpBackend(MlpSparsityBackend):
    """Dense gate, up and down maps with masked channels (reference numerics, no speedup)."""

    simulation = Simulation(Saving.COMPUTE | Saving.LATENCY, "computes the dense MLP maps and masks the channels")

    def ops(self, mlp: GatedMlp) -> MlpOps:
        return DenseOps()


class TritonOps(DenseOps):
    """The kernels of :class:`TritonMlpBackend` for one MLP.

    Installing stores the down weight transposed (``[in, out]`` row-major behind the same ``[out, in]`` parameter,
    the layout of TEAL's kernel); :meth:`remove` restores the row-major storage.
    """

    def __init__(self, backend: TritonMlpBackend, mlp: GatedMlp) -> None:
        self.backend = backend
        self._down: nn.Linear | None = None
        self._storage_ptr = 0
        self.realized = all(backend.eligible(m) for m in (mlp.gate, mlp.up, mlp.down))
        self._row_plans = {
            id(m): backend.row_plan(cast(nn.Linear, m)) for m in (mlp.gate, mlp.up) if backend.eligible(m)
        }
        self._down_plan: KernelPlan[GemvConfig] = KernelPlan()
        down = mlp.down
        if backend.eligible(down):
            linear = cast(nn.Linear, down)
            linear.weight.data = linear.weight.data.t().contiguous().t()
            self._down, self._storage_ptr = linear, linear.weight.data.data_ptr()
            self._zero = torch.zeros((), dtype=torch.float32, device=linear.weight.device)
            self._down_plan = backend.gemv_plan(linear)

    @staticmethod
    def _small(x: Tensor, weight: Tensor) -> int:
        """The row count of ``x`` when the kernels take it, else 0."""
        rows = x.numel() // max(x.shape[-1], 1)
        grad = torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad)
        return rows if 0 < rows <= MAX_ROWS and not grad and x.dtype == weight.dtype else 0

    def rows(self, x: Tensor, linear: nn.Module, keep: Tensor) -> Tensor:
        weight = getattr(linear, "weight", None)
        if not (isinstance(weight, Tensor) and self.backend.eligible(linear)):
            return super().rows(x, linear, keep)
        rows = self._small(x, weight)
        plan = self._row_plans.get(id(linear))
        if not rows or plan is None or rows > plan.max_rows or weight.stride(1) != 1:
            return super().rows(x, linear, keep)
        out = row_masked_gemv(
            x.reshape(rows, x.shape[-1]), weight, keep.reshape(rows, -1), plan.config(rows)
        ).reshape(*x.shape[:-1], weight.shape[0])
        bias = getattr(linear, "bias", None)
        return out if bias is None else out + masked(bias.expand_as(out), keep)

    def cols(self, h: Tensor, linear: nn.Module) -> Tensor:
        down = self._down
        if down is not linear or down is None or not down.weight.t().is_contiguous():
            return super().cols(h, linear)
        weight = down.weight
        rows = self._small(h, weight)
        plan = self._down_plan
        if not rows or rows > plan.max_rows:
            return F.linear(h, weight, down.bias)
        out = teal_gemv(h.reshape(rows, h.shape[-1]), weight.t(), self._zero, plan.config(rows))
        if down.bias is not None:
            out = out + down.bias
        return out.reshape(*h.shape[:-1], out.shape[-1])

    def remove(self) -> None:
        down = self._down
        if down is not None and isinstance(down.weight, Tensor) and down.weight.data.data_ptr() == self._storage_ptr:
            down.weight.data = down.weight.data.contiguous()
        self._down = None


@MLP_SPARSITY_BACKENDS.register("triton")
class TritonMlpBackend(MlpSparsityBackend):
    """Row-masked GEMV for the up (and gate) maps and TEAL's column-sparse GEMV for the down map (Triton), for calls
    of up to 16 rows on 16-bit CUDA weights; the dense masked path elsewhere (training, prefill, CPU).

    The launch parameters of every map come from :meth:`row_plan` and :meth:`gemv_plan`, which a subclass overrides
    to tune them (the default ``auto`` backend times them on the device, :mod:`.autotune`).

    Attributes:
        row_config: Launch parameters of the row-masked kernel (``None``: :func:`.kernel.default_row_config`).
        gemv_config: Launch parameters of TEAL's kernel (``None``: its default heuristic).
    """

    row_config: ClassVar[RowGemvConfig | None] = None
    gemv_config: ClassVar[GemvConfig | None] = None

    @classmethod
    def usable(cls) -> bool:
        """Whether the kernels can run here (Triton and a CUDA device)."""
        return triton_available()

    @classmethod
    def eligible(cls, module: nn.Module) -> bool:
        """Whether the kernels can serve ``module`` (an ``nn.Linear`` with a plain 16-bit CUDA weight)."""
        return cls.usable() and kernel_eligible(module)

    def row_plan(self, linear: nn.Linear) -> KernelPlan[RowGemvConfig]:
        """How the row-masked kernel runs the up (or gate) map ``linear``: :attr:`row_config` for every call size."""
        return KernelPlan(self.row_config, self.row_config)

    def gemv_plan(self, linear: nn.Linear) -> KernelPlan[GemvConfig]:
        """How TEAL's kernel runs the down map ``linear``: :attr:`gemv_config` for every call size."""
        return KernelPlan(self.gemv_config, self.gemv_config)

    def ops(self, mlp: GatedMlp) -> MlpOps:
        return TritonOps(self, mlp)


def fallback_simulation(ops: Iterable[MlpOps], saves: Saving) -> Simulation | None:
    """The simulation of installed operations of which some compute a map densely (:attr:`MlpOps.realized` is
    ``False``), restricted to ``saves``; ``None`` when every MLP realizes its saving."""
    found = list(ops)
    dense = sum(not o.realized for o in found)
    if not dense:
        return None
    reason = f"computes the dense maps and masks the channels in {dense} of {len(found)} MLPs the kernels cannot serve"
    return Simulation(Saving.COMPUTE | Saving.LATENCY, reason).within(saves)
