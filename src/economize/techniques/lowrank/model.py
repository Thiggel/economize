"""Whole-model low-rank compression (a pure function on an adapter; the technique delegates to it)."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from ...core.calibration import GramObserver
from .._weights.layerwise import (
    observe_step,
    require_calibration,
    resolve_layers,
    rewrite_layerwise,
    target_linears,
)
from .._weights.transaction import WeightTransaction
from .config import LowRankConfig
from .factorize import FACTORIZERS, rank_for_ratio
from .module import LowRankLinear

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...core.calibration import CalibrationStep, Propagation

__all__ = ["lowrank_model"]


@torch.no_grad()
def lowrank_model(
    adapter: ArchAdapter, config: LowRankConfig, batches: Sequence[Tensor] | None = None
) -> dict[str, Any]:
    """Replace the selected linear maps by :class:`LowRankLinear` factors in place and return a summary.

    SVD-LLM profiles ``X X^T`` of every map layer by layer on the original model's activations (or, with
    ``sequential``, on the compressed model's). A map is kept dense when the rank for ``compression`` would not
    save parameters.

    All or nothing: if a factorization or the calibration fails, every replaced map is swapped back before the
    error propagates.
    """
    factorizer = FACTORIZERS.get(config.method)()
    layers = resolve_layers(config.layers, adapter.num_layers)
    counts = {"modules": 0, "kept_dense": 0, "params_before": 0, "params_after": 0}
    ranks: dict[str, int] = {}
    # the dense maps wait on the CPU: kept on the GPU next to their factors they overflow it at 31B
    # (see offload_replaced)
    transaction = WeightTransaction(offload_replaced=True)

    def compress(index: int, layer: nn.Module, step: CalibrationStep | None) -> None:
        modules = target_linears(adapter, layer, config)
        grams = observe_step(step, modules, partial(GramObserver, dtype=torch.float64)) if step is not None else None
        for name, module in modules.items():
            # each Gram matrix is freed once its map is done (21504^2 float64 is 3.7 GB at a 31B MLP input)
            observer = grams.pop(name) if grams is not None else None
            out_features, in_features = module.weight.shape
            rank = rank_for_ratio(out_features, in_features, config.compression, config.rank_multiple)
            counts["params_before"] += out_features * in_features
            if rank * (out_features + in_features) >= out_features * in_features:
                counts["kept_dense"] += 1
                counts["params_after"] += out_features * in_features
                continue
            gram = observer.mean_() if observer is not None else None
            del observer
            up, down = factorizer.factorize(module.weight.data, gram, rank, overwrite_gram=True)
            del gram
            transaction.replace_module(layer, name, LowRankLinear.from_factors(up, down, module.bias, module.weight))
            ranks[f"{index}.{name}"] = rank
            counts["modules"] += 1
            counts["params_after"] += rank * (out_features + in_features)

    with transaction:
        calibration = None
        if factorizer.needs_calibration:
            calibration = require_calibration(batches, f"low-rank method {config.method!r}")
        propagate: Propagation = "transformed" if config.sequential else "original"
        rewrite_layerwise(adapter, layers, compress, batches=calibration, propagate=propagate, offload=config.offload)
    kept = counts["params_after"] / max(counts["params_before"], 1)
    return {"method": config.method, "compression": 1 - kept, **counts, "ranks": ranks}
