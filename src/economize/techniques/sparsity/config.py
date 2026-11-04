"""Configuration of TEAL activation sparsity (``teal``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from ...core.config import TechniqueConfig
from ...validation import (
    check_choice,
    check_globs,
    check_layers,
    check_non_negative,
    check_positive,
    check_registered,
    check_unit_interval,
)
from .sparsify import PhasePolicy

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["TEAL_ALLOCATIONS", "TealConfig"]

TEAL_ALLOCATIONS = ("greedy", "uniform")


@dataclass(frozen=True)
class TealConfig(TechniqueConfig):
    """Magnitude activation sparsity on the inputs of the linear maps of every decoder layer (TEAL).

    Args:
        sparsity: Target layer-level sparsity: the parameter-weighted mean fraction of zeroed inputs, i.e. the
            fraction of weight reads a sparse kernel skips (the paper reports 25%, 40% and 50%).
        allocation: ``"greedy"`` (TEAL's block-wise greedy search, per-map levels) or ``"uniform"`` (every map at
            ``sparsity``; the paper's "uniform" baseline).
        greedy_step: Base step of the greedy search: a map with ``hidden_size**2`` parameters moves by this much per
            accepted move, larger maps proportionally less (the reference's ``base_step_size``, 0.05).
        greedy_fraction: Trailing fraction of sequence positions on which the greedy search measures the layer
            error (the reference's ``last_fraction``, 0.25).
        greedy_batches: Number of calibration batches the greedy search evaluates (0: all). Every accepted move
            costs one layer forward per candidate map, so long calibration sets are expensive here.
        prefill: Which tokens each kind of call sparsifies: a :class:`~.sparsify.PhasePolicy` (or the mapping of its
            fields, converted to the policy), or a mode string for the exact passes and the multi-token calls
            outside decoding (:meth:`PhasePolicy.from_prefill`): ``"dense"`` (default, as TEAL's single-token
            kernels), ``"second_half"`` (TEAL's perplexity protocol) or ``"all"``. With a string, decoding and
            drafting steps are always sparsified.
        backend: Registered kernel backend (``"dense"``: masking hooks, the numerical reference; ``"triton"``: TEAL's
            sparse GEMV for calls of up to 16 rows, :mod:`.triton_backend`; ``"triton_autotune"``: the same kernel with
            launch parameters timed per map shape on the device, :mod:`.triton_autotune`).
        targets: Globs over linear-module names inside a layer.
        exclude: Globs removed from ``targets``.
        layers: 1-indexed layers to sparsify; empty means all.
        max_samples: Magnitudes kept per map for the threshold quantiles.
        offload: Keep calibration activations on the CPU between layers.
    """

    technique: ClassVar[str] = "teal"

    sparsity: float = 0.4
    allocation: str = "greedy"
    greedy_step: float = 0.05
    greedy_fraction: float = 0.25
    greedy_batches: int = 0
    prefill: str | PhasePolicy = "dense"
    backend: str = "dense"
    targets: tuple[str, ...] = ("*",)
    exclude: tuple[str, ...] = ()
    layers: tuple[int, ...] = ()
    max_samples: int = 1 << 20
    offload: bool = False

    def __post_init__(self) -> None:
        from .backends import TEAL_BACKENDS

        check_unit_interval("sparsity", self.sparsity)
        check_choice("allocation", self.allocation, TEAL_ALLOCATIONS)
        check_unit_interval("greedy_step", self.greedy_step, closed_low=False, closed_high=True)
        check_unit_interval("greedy_fraction", self.greedy_fraction, closed_low=False, closed_high=True)
        check_non_negative("greedy_batches", self.greedy_batches)
        policy = PhasePolicy.coerce(self.prefill)
        if not isinstance(self.prefill, str):  # a mapping (JSON) becomes the policy, so the config stays hashable
            object.__setattr__(self, "prefill", policy)
        check_registered("TEAL backend", self.backend, TEAL_BACKENDS)
        check_globs("targets", self.targets)
        check_globs("exclude", self.exclude, allow_empty=True)
        check_layers(self.layers)
        check_positive("max_samples", self.max_samples)

    @property
    def policy(self) -> PhasePolicy:
        """The :class:`~.sparsify.PhasePolicy` that ``prefill`` describes."""
        return PhasePolicy.coerce(self.prefill)

    def validate(self, adapter: ArchAdapter) -> None:
        check_layers(self.layers, adapter)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        if not isinstance(self.prefill, str):
            data["prefill"] = self.policy.to_dict()
        return data
