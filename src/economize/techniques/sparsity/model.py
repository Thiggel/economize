"""Calibration and installation of TEAL on a whole model.

:func:`calibrate_teal` follows the reference pipeline (``teal/grab_acts.py`` and ``teal/greedyopt.py``): the dense
model runs on generic calibration text, the input magnitudes of every target map are sampled layer by layer, the
per-map sparsity levels come from the uniform rule or the block-wise greedy search, and each level becomes a
threshold, the matching quantile of that map's dense input magnitudes. :func:`install_teal` puts the thresholds in
front of the maps with a registered backend.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from ...core.calibration import (
    CalibrationStep,
    LayerwiseCalibration,
    MagnitudeSampleObserver,
    observing,
    resolve_layers,
)
from .._weights.layerwise import target_linears
from .backends import TEAL_BACKENDS, DenseBackend, Installed
from .config import TealConfig
from .sparsify import (
    GreedyStep,
    block_sparsity,
    greedy_allocation,
    greedy_error,
    magnitude_quantile,
    track_layer_tokens,
    uniform_allocation,
)

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["SparsityMeter", "TealCalibration", "calibrate_teal", "install_teal", "measure_sparsity", "teal_modules"]


@dataclass(frozen=True)
class TealCalibration:
    """Per-map sparsity levels and thresholds, keyed ``"<layer>.<module>"`` (layers 1-indexed).

    Attributes:
        sparsities: Target input sparsity of every map.
        thresholds: Magnitude threshold of every map (inputs with ``|x| <= threshold`` are zeroed).
        sizes: Parameter count of every map (the weights of :func:`.sparsify.block_sparsity`).
        paths: Accepted moves of the greedy search per layer (empty for the uniform allocation).
    """

    sparsities: dict[str, float]
    thresholds: dict[str, float]
    sizes: dict[str, int]
    paths: dict[int, list[GreedyStep]] = field(default_factory=dict)

    @property
    def model_sparsity(self) -> float:
        """Parameter-weighted mean sparsity over all maps: the fraction of weight reads a sparse kernel skips."""
        return block_sparsity(self.sparsities, self.sizes)


def teal_modules(adapter: ArchAdapter, config: TealConfig) -> dict[str, nn.Module]:
    """The maps TEAL sparsifies, keyed ``"<layer>.<module>"`` in layer order, then in the layer's module order.

    The order depends only on the model and the config, so a technique stores its thresholds as a flat tensor.
    """
    layers = sorted(resolve_layers(config.layers, adapter.num_layers))
    return {
        f"{index}.{name}": module
        for index in layers
        for name, module in target_linears(adapter, adapter.layer(index), config).items()
    }


def _greedy_levels(
    step: CalibrationStep,
    modules: Mapping[str, nn.Module],
    magnitudes: Mapping[str, Tensor],
    sizes: Mapping[str, int],
    config: TealConfig,
    unit: int,
) -> tuple[dict[str, float], list[GreedyStep]]:
    """Run the greedy search of one layer against the dense outputs of :meth:`CalibrationStep.forward`."""
    assert step.first_outputs is not None, "the dense forward of the layer must run first"
    count = len(step.inputs) if config.greedy_batches == 0 else min(config.greedy_batches, len(step.inputs))
    references = step.first_outputs[:count]
    thresholds = {name: torch.zeros((), device=cast(Tensor, module.weight).device) for name, module in modules.items()}
    # The reference sparsifies the second half of each calibration sequence during the search (apply_prefill).
    installed = DenseBackend().install(modules, thresholds, "second_half")
    installed.support += track_layer_tokens([step.layer])  # positions of every sequence, also for flattened inputs

    def error(levels: Mapping[str, float]) -> float:
        for name, level in levels.items():
            thresholds[name].fill_(magnitude_quantile(magnitudes[name], level))
        outputs = step.forward(range(count))
        return sum(float(greedy_error(r, o, config.greedy_fraction)) for r, o in zip(references, outputs)) / count

    try:
        return greedy_allocation(error, sizes, config.sparsity, step=config.greedy_step, unit=unit)
    finally:
        installed.remove()


@torch.no_grad()
def calibrate_teal(adapter: ArchAdapter, config: TealConfig, batches: Sequence[Tensor]) -> TealCalibration:
    """Per-map thresholds for ``config.sparsity`` from the dense model's activations on ``batches``.

    Every layer sees the dense model's inputs (the reference collects all activations from the unmodified model),
    so calibration is independent of the order in which layers are processed.
    """
    sparsities: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    sizes: dict[str, int] = {}
    paths: dict[int, list[GreedyStep]] = {}
    unit = adapter.hidden_size**2
    layers = resolve_layers(config.layers, adapter.num_layers)
    for step in LayerwiseCalibration(adapter, batches, propagate="original", layers=layers, offload=config.offload):
        modules = target_linears(adapter, step.layer, config)
        observer = partial(MagnitudeSampleObserver, max_samples=config.max_samples, seed=step.index)
        with observing(modules, observer) as observers:
            step.forward()
        magnitudes = {name: obs.values().sort().values for name, obs in observers.items()}
        layer_sizes = {name: module.weight.numel() for name, module in modules.items()}
        if config.allocation == "greedy" and modules:
            levels, paths[step.index] = _greedy_levels(step, modules, magnitudes, layer_sizes, config, unit)
        else:
            levels = uniform_allocation(layer_sizes, config.sparsity)
        for name in modules:
            key = f"{step.index}.{name}"
            sparsities[key] = levels[name]
            thresholds[key] = magnitude_quantile(magnitudes[name], levels[name])
            sizes[key] = layer_sizes[name]
    return TealCalibration(sparsities, thresholds, sizes, paths)


def install_teal(
    adapter: ArchAdapter, thresholds: Mapping[str, float | Tensor], config: TealConfig
) -> tuple[Installed, dict[str, Tensor]]:
    """Sparsify the input of every target map with its threshold, using the backend ``config.backend``.

    Maps with threshold 0 (sparsity level 0) are left untouched. Every decoder layer gets a hook that records its
    token layout (:func:`.sparsify.track_layer_tokens`), so maps with flattened inputs follow the phase policy like
    the others. Returns the handles and the installed 0-dim
    threshold tensors (one per map, on the map's device), which may be updated in place.
    """
    modules = {key: module for key, module in teal_modules(adapter, config).items() if float(thresholds[key]) > 0}
    tensors = {
        key: torch.tensor(float(thresholds[key]), dtype=torch.float32, device=cast(Tensor, module.weight).device)
        for key, module in modules.items()
    }
    installed = TEAL_BACKENDS.get(config.backend)().install(modules, tensors, config.policy)
    installed.support += track_layer_tokens(adapter.layers)
    return installed, tensors


class SparsityMeter:
    """Counts zero inputs of the given maps (for :func:`measure_sparsity`)."""

    def __init__(self, module: nn.Module | None = None) -> None:
        self.zeros = 0
        self.total = 0

    def update(self, x: Tensor) -> None:
        self.zeros += int((x == 0).sum())
        self.total += x.numel()

    @property
    def sparsity(self) -> float:
        return self.zeros / max(self.total, 1)


@torch.no_grad()
def measure_sparsity(adapter: ArchAdapter, config: TealConfig, run: Callable[[], object]) -> dict[str, float]:
    """Fraction of zero entries in the inputs every target map actually multiplies while ``run()`` executes.

    The meters sit behind the installed TEAL hooks (forward pre-hooks run in registration order), so they see the
    sparsified inputs. Returns per-map sparsities keyed like :func:`teal_modules` plus ``"model"``, the
    parameter-weighted mean.
    """
    modules = teal_modules(adapter, config)
    with observing(modules, SparsityMeter) as meters:
        run()
    result = {key: meter.sparsity for key, meter in meters.items()}
    sizes = {key: cast(Tensor, module.weight).numel() for key, module in modules.items()}
    result["model"] = block_sparsity(result, sizes)
    return result
