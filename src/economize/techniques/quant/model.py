"""Whole-model quantization procedures (pure functions; the techniques delegate to them)."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from ...errors import ConfigError
from .._weights.layerwise import require_calibration, resolve_layers, rewrite_layerwise, target_linears
from .._weights.transaction import WeightTransaction
from . import activations
from .backends import QUANT_BACKENDS
from .config import DraftQuantConfig, QuantConfig
from .grid import QuantizedWeight, bind_shards, quantize_rtn
from .methods import QUANT_METHODS

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...core.calibration import CalibrationStep

__all__ = ["quantizable", "quantize_model", "quantize_modules", "require_divisible", "split_quantizable"]

logger = logging.getLogger(__name__)


def quantizable(
    modules: Mapping[str, nn.Module], group_size: int | None, where: str = "", *, warn: bool = True
) -> dict[str, nn.Linear]:
    """The ``nn.Linear`` modules whose input size ``group_size`` divides; the others are skipped (with a warning
    unless ``warn`` is false)."""
    kept, skipped = split_quantizable(modules, group_size)
    if skipped and warn:
        logger.warning(
            "%sskipping %s: not nn.Linear or input size not divisible by group size %s", where, skipped, group_size
        )
    return kept


def split_quantizable(
    modules: Mapping[str, nn.Module], group_size: int | None
) -> tuple[dict[str, nn.Linear], list[str]]:
    """``(quantizable modules, names of the skipped ones)``: see :func:`quantizable`."""
    kept: dict[str, nn.Linear] = {}
    skipped: list[str] = []
    for name, module in modules.items():
        if not isinstance(module, nn.Linear) or (group_size and module.in_features % group_size):
            skipped.append(name)
        else:
            kept[name] = module
    return kept, skipped


def require_divisible(modules: Mapping[str, nn.Linear], group_size: int | None) -> None:
    """Raise unless ``group_size`` divides the input size of every map in ``modules``.

    Whole-model quantization refuses instead of skipping such maps, since a model with some maps left in full
    precision would still be reported (and exported) as the configured format.

    Raises:
        ConfigError: Naming the maps and the group sizes that would divide all of them.
    """
    if not group_size:
        return
    bad = [f"{name} ({m.in_features} inputs)" for name, m in modules.items() if m.in_features % group_size]
    if not bad:
        return
    common = math.gcd(*(m.in_features for m in modules.values()))
    divisor = next((g for g in (128, 64, 32, 16) if g < group_size and common % g == 0), None)
    hint = f"group_size={divisor} (divides every input size) or " if divisor else ""
    raise ConfigError(
        f"group_size={group_size} does not divide the input size of {len(bad)} linear map(s): {', '.join(bad)}. "
        f"Pass {hint}group_size=None (one scale per output channel), or exclude these maps with `exclude`."
    )


def _numel(module: nn.Module) -> int:
    weight = getattr(module, "weight", None)
    return weight.numel() if isinstance(weight, Tensor) else 0


class _Summary:
    def __init__(self) -> None:
        self.modules = 0
        self.weights = 0
        self.bits = 0.0
        self.skipped: dict[str, int] = {}

    def skip(self, modules: Mapping[str, nn.Module]) -> None:
        self.skipped.update({name: _numel(module) for name, module in modules.items()})

    def add(self, q: QuantizedWeight) -> None:
        self.modules += 1
        self.weights += q.codes.numel()
        self.bits += q.bits_per_weight * q.codes.numel()

    def as_dict(self, config: QuantConfig | DraftQuantConfig, method: str, kernel: str) -> dict[str, object]:
        candidates = self.weights + sum(self.skipped.values())
        return {
            "method": method,
            "format": config.format,
            "group_size": config.group_size,
            "symmetric": config.grid.symmetric,  # fp8 is scale-only whatever the flag says
            "backend": config.backend,
            "kernel": kernel,  # the backend that stores the weights (what backend="auto" chose)
            "modules": self.modules,
            "weights": self.weights,
            "bits_per_weight": self.bits / max(self.weights, 1),
            "skipped": list(self.skipped),  # candidates left in their original precision
            # fraction of the candidate weights stored in ``format`` (1.0 unless maps were skipped)
            "coverage": self.weights / candidates if candidates else 1.0,
        }


@torch.no_grad()
def quantize_model(
    adapter: ArchAdapter, config: QuantConfig, batches: Sequence[Tensor] | None = None
) -> dict[str, object]:
    """Quantize the selected linear maps of every selected decoder layer in place and return a summary.

    Calibrated methods run layer by layer (GPTQ on the quantized model's activations, AWQ on the original model's).
    Calibration always sees the dequantized weights; a kernel backend converts a layer once its outputs have been
    propagated, so the codes are identical for every backend.

    All or nothing: the backend's :meth:`~.backends.QuantBackend.check` sees every target before any weight
    changes, and if a method or a conversion fails later, every parameter of the layers touched so far (also norms
    and maps that AWQ rescaled) gets its original value back before the error propagates. The originals of the
    layer being quantized are kept on the CPU until the call returns, so the peak host memory grows by the
    parameters of the selected layers. Activation-quantization hooks (``config.activations``) installed so far are
    removed as well.
    """
    method = QUANT_METHODS.get(config.method)()
    backend = QUANT_BACKENDS.get(config.backend)()
    grid = config.grid
    layers = resolve_layers(config.layers, adapter.num_layers)
    calibration = None
    if method.needs_calibration:
        calibration = require_calibration(batches, f"quantization method {config.method!r}")
    targets: dict[str, nn.Linear] = {}
    for index in layers:
        selected = target_linears(adapter, adapter.layer(index), config)
        targets.update({f"layers.{index - 1}.{name}": module for name, module in selected.items()})
    backend.check(targets, config)  # first: on a machine without a kernel, installing one is the first fix
    require_divisible(targets, config.group_size)
    summary = _Summary()
    pending: list[tuple[nn.Linear, QuantizedWeight]] = []
    transaction = WeightTransaction()

    def convert_pending() -> None:
        for module, q in pending:
            if q.g_idx is not None and not backend.supports_g_idx:
                raise ConfigError(f"backend {backend.resolved(config)!r} cannot store non-contiguous groups (g_idx)")
            backend.apply(module, q, config)
        pending.clear()

    def quantize_layer(_index: int, layer: nn.Module, step: CalibrationStep | None) -> None:
        convert_pending()  # the previous layer's outputs have been propagated by now
        transaction.take(layer)
        modules = target_linears(adapter, layer, config)
        for name, q in method.quantize_layer(modules, step, grid, config, adapter).items():
            pending.append((modules[name], q))
            summary.add(q)

    try:
        with transaction:
            rewrite_layerwise(
                adapter, layers, quantize_layer, batches=calibration, propagate=method.propagate,
                offload=config.offload,
            )
            convert_pending()
    except BaseException:
        for module in targets.values():  # the transaction restores parameters, not activation hooks
            activations.uninstall(module)
        raise
    return summary.as_dict(config, config.method, backend.resolved(config))


@torch.no_grad()
def quantize_modules(
    modules: Mapping[str, nn.Linear], config: DraftQuantConfig, skipped: Mapping[str, nn.Module] | None = None
) -> dict[str, object]:
    """Round-to-nearest quantization of the given linear maps (used for drafter-only quantization); ``skipped``
    holds the candidates that were left out, which the summary lists and counts against its ``coverage``."""
    backend = QUANT_BACKENDS.get(config.backend)()
    backend.check(modules, config)
    grids = bind_shards(modules, config.grid)
    summary = _Summary()
    summary.skip(skipped or {})
    for name, module in modules.items():
        q = quantize_rtn(module.weight.data, grids[name], config.group_size)
        module.weight.data.copy_(q.dequantize(module.weight.dtype))
        backend.apply(module, q, config)
        summary.add(q)
    return summary.as_dict(config, "rtn", backend.resolved(config))

