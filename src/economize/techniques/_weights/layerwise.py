"""The layer-by-layer driver shared by the techniques that rewrite the linear maps of the decoder layers (``prune``,
``lowrank``, ``quant``) and the selection helpers it is built from.

A whole-model rewrite resolves its layers with :func:`~economize.core.calibration.resolve_layers` (re-exported here),
picks each layer's maps with :func:`target_linears` and hands one callback per layer to :func:`rewrite_layerwise`,
which either walks the layers data-free or runs :class:`~economize.core.calibration.LayerwiseCalibration` and passes
the callback each :class:`~economize.core.calibration.CalibrationStep`. :func:`observe_step` collects one statistic per
map from a step's calibration inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, TypeVar

from torch import Tensor, nn

from ...core.calibration import (
    CalibrationStep,
    LayerwiseCalibration,
    linear_modules,
    observing,
    resolve_layers,
    select_modules,
)
from ...errors import ConfigError

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...core.calibration import Observer, Propagation

__all__ = [
    "LayerRewrite",
    "TargetSelection",
    "observe_step",
    "require_calibration",
    "resolve_layers",
    "rewrite_layerwise",
    "target_linears",
]

O = TypeVar("O", bound="Observer")
M = TypeVar("M", bound=nn.Module)

LayerRewrite = Callable[[int, nn.Module, CalibrationStep | None], None]
"""``rewrite(index, layer, step)``: transform one decoder layer (1-indexed ``index``). ``step`` is the layer's
calibration step, or ``None`` on the data-free path."""


class TargetSelection(Protocol):
    """The module selection fields of a technique config: glob patterns over the names inside a decoder layer."""

    @property
    def targets(self) -> Sequence[str]: ...

    @property
    def exclude(self) -> Sequence[str]: ...


def target_linears(adapter: ArchAdapter, layer: nn.Module, selection: TargetSelection) -> dict[str, nn.Linear]:
    """The ``nn.Linear`` maps of one decoder layer that ``selection.targets`` matches and ``selection.exclude`` does
    not, keyed by their dotted name inside the layer, in the layer's module order."""
    linears = linear_modules(adapter, layer)
    selected = select_modules(linears, selection.targets, selection.exclude)
    return {name: linears[name] for name in selected}


def require_calibration(batches: Sequence[Tensor] | None, what: str) -> Sequence[Tensor]:
    """``batches``, or a :class:`~economize.errors.ConfigError` saying that ``what`` needs calibration data."""
    if batches is None:
        raise ConfigError(f"{what} needs calibration data")
    return batches


def observe_step(
    step: CalibrationStep, modules: Mapping[str, M], factory: Callable[[M], O]
) -> dict[str, O]:
    """Run ``step`` once on all its calibration batches with one observer per module (built by ``factory``) on the
    module's inputs, and return the observers."""
    with observing(modules, factory) as observers:
        step.forward()
    return observers


def rewrite_layerwise(
    adapter: ArchAdapter,
    layers: Sequence[int],
    rewrite: LayerRewrite,
    *,
    batches: Sequence[Tensor] | None = None,
    propagate: Propagation = "transformed",
    offload: bool = False,
) -> None:
    """Call ``rewrite`` on each selected decoder layer.

    Without ``batches`` the layers are visited data-free in the order of ``layers`` and ``rewrite`` gets ``step=None``.
    With ``batches`` a :class:`~economize.core.calibration.LayerwiseCalibration` (``propagate``, ``offload``) visits
    them in ascending order and ``rewrite`` gets each layer's step; with ``propagate="transformed"`` the next layer sees
    the outputs of the layer as ``rewrite`` left it.
    """
    if batches is None:
        for index in layers:
            rewrite(index, adapter.layer(index), None)
        return
    for step in LayerwiseCalibration(adapter, batches, propagate=propagate, layers=layers, offload=offload):
        rewrite(step.index, step.layer, step)
