"""Whole-model pruning procedures (pure functions on an adapter; the techniques delegate to them)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from .._weights.layerwise import (
    observe_step,
    require_calibration,
    resolve_layers,
    rewrite_layerwise,
    target_linears,
)
from .._weights.transaction import WeightTransaction
from .config import DepthPruneConfig, PruneConfig
from .depth import DEPTH_CRITERIA, remove_layers
from .methods import PRUNE_METHODS
from .semi_structured import to_semi_structured

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...core.calibration import CalibrationStep, Observer

__all__ = ["depth_prune_model", "prune_model"]


@torch.no_grad()
def prune_model(adapter: ArchAdapter, config: PruneConfig, batches: Sequence[Tensor] | None = None) -> dict[str, Any]:
    """Prune the selected linear maps of every selected layer in place and return a summary.

    Calibrated criteria run layer by layer: the dense inputs of a layer's linear maps are observed in one pass
    (as in the Wanda and SparseGPT references), all maps are pruned, and the pruned layer's outputs feed the next
    layer.

    Returns:
        ``method`` and ``pattern``; ``weights``: the entries of the pruned maps; ``eligible``: those that were
        nonzero before; ``sparsity``: the induced sparsity, the fraction of the eligible entries this call zeroed
        (0 when a model is pruned twice with the same config); ``zeros``: the fraction of all entries that are zero
        afterwards, zeros the weights already had included.

    All or nothing: if pruning, the calibration or the 2:4 conversion fails, every touched layer gets its original
    weights back before the error propagates.
    """
    pruner = PRUNE_METHODS.get(config.method)()
    pattern = config.pattern
    layers = resolve_layers(config.layers, adapter.num_layers)
    newly = eligible = zeros = total = 0
    transaction = WeightTransaction()

    def observer(module: nn.Linear) -> Observer:
        built = pruner.observer(module)
        if built is None:
            raise TypeError(f"pruning method {config.method!r} needs calibration but its observer() returned None")
        return built

    def prune_layer(_index: int, layer: nn.Module, step: CalibrationStep | None) -> None:
        nonlocal newly, eligible, zeros, total
        modules = target_linears(adapter, layer, config)
        stats = observe_step(step, modules, observer) if step is not None else None
        for name, module in modules.items():
            transaction.take(module)
            nonzero = module.weight.data != 0
            pruned = pruner.prune(module.weight.data, stats[name] if stats else None, pattern, config)
            module.weight.data.copy_(pruned)
            newly += int((nonzero & (pruned == 0)).sum())
            eligible += int(nonzero.sum())
            zeros += int((pruned == 0).sum())
            total += pruned.numel()

    with transaction:
        calibration = None
        if pruner.needs_calibration:
            calibration = require_calibration(batches, f"pruning method {config.method!r}")
        rewrite_layerwise(adapter, layers, prune_layer, batches=calibration, offload=config.offload)
        if config.semi_structured:
            for index in layers:
                for module in target_linears(adapter, adapter.layer(index), config).values():
                    to_semi_structured(module, config.semi_structured_kernel)
    return {
        "method": config.method,
        "pattern": str(pattern),
        "sparsity": newly / max(eligible, 1),
        "zeros": zeros / max(total, 1),
        "eligible": eligible,
        "weights": total,
    }


@torch.no_grad()
def depth_prune_model(adapter: ArchAdapter, config: DepthPruneConfig, batches: Sequence[Tensor]) -> list[int]:
    """Select layers with the configured criterion, remove them in place and return their original indices."""
    candidates = config.candidates(adapter.num_layers)
    chosen = DEPTH_CRITERIA.get(config.method)().select(adapter, batches, config.num_blocks, candidates)
    remove_layers(adapter, chosen)
    return chosen
