"""Whole-model LittleBit compression and the factor files (pure functions on an adapter; the technique delegates)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from ...errors import ConfigError
from .._weights.layerwise import require_calibration, resolve_layers, rewrite_layerwise, target_linears
from .._weights.transaction import WeightTransaction
from .config import LittleBitConfig
from .factorize import bits_of, rank_for_bpw
from .methods import LITTLEBIT_METHODS
from .module import LittleBitLinear

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...core.calibration import CalibrationStep, Propagation

__all__ = ["littlebit_model", "load_factors", "save_factors"]

_BUFFERS = ("v_bits", "u_bits", "g", "l", "h")


@torch.no_grad()
def littlebit_model(
    adapter: ArchAdapter, config: LittleBitConfig, batches: Sequence[Tensor] | None = None
) -> dict[str, Any]:
    """Replace the selected linear maps by :class:`LittleBitLinear` in place and return a summary.

    The rank of each map is the largest that fits ``config.bpw`` (signs and scales counted, Appendix D of the
    paper), rounded down to ``config.rank_multiple``; a map is kept dense when not even rank ``rank_multiple``
    fits. With ``config.factors`` the maps are rebuilt from a :func:`save_factors` file instead.

    All or nothing: if a factorization, the training or the calibration fails, every replaced map is swapped back
    before the error propagates.
    """
    if config.factors:
        return _load_model(adapter, config)
    method = LITTLEBIT_METHODS.get(config.method)()
    layers = resolve_layers(config.layers, adapter.num_layers)
    counts = {"modules": 0, "kept_dense": 0, "weights": 0, "bits": 0}
    ranks: dict[str, int] = {}
    losses: dict[int, dict[str, Any]] = {}
    transaction = WeightTransaction()

    def compress(index: int, layer: nn.Module, step: CalibrationStep | None) -> None:
        modules = target_linears(adapter, layer, config)
        chosen: dict[str, int] = {}
        for name, module in modules.items():
            out_features, in_features = module.weight.shape
            counts["weights"] += out_features * in_features
            rank = rank_for_bpw(
                out_features, in_features, config.bpw, paths=config.paths, multiple=config.rank_multiple
            )
            if rank == 0:
                counts["kept_dense"] += 1
                counts["bits"] += 16 * out_features * in_features
            else:
                chosen[name] = rank
        selected = {name: modules[name] for name in chosen}
        factors, stats = method.factorize(layer, selected, chosen, step, config)
        if stats:
            losses[index] = stats
        for name, module in selected.items():
            packed = LittleBitLinear.from_factors(factors.pop(name), module.bias, module.weight, config.backend)
            transaction.replace_module(layer, name, packed)
            ranks[f"{index}.{name}"] = chosen[name]
            counts["modules"] += 1
            counts["bits"] += packed.stored_bits()

    with transaction:
        calibration = None
        if method.needs_calibration:
            calibration = require_calibration(batches, f"LittleBit method {config.method!r}")
        propagate: Propagation = "transformed" if config.sequential else "original"
        rewrite_layerwise(adapter, layers, compress, batches=calibration, propagate=propagate, offload=config.offload)
    return {
        "method": config.method,
        "bpw_target": config.bpw,
        "bpw": counts["bits"] / max(counts["weights"], 1),
        **counts,
        "ranks": ranks,
        "losses": losses,
    }


def _littlebit_modules(adapter: ArchAdapter) -> dict[str, LittleBitLinear]:
    found = {}
    for index in range(1, adapter.num_layers + 1):
        for name, module in adapter.layer(index).named_modules():
            if isinstance(module, LittleBitLinear):
                found[f"{index}.{name}"] = module
    return found


def save_factors(adapter: ArchAdapter, path: str | Path) -> int:
    """Write the packed signs and scales of every :class:`LittleBitLinear` of the decoder layers to a safetensors
    file (keys ``"<layer>.<module>.<buffer>"``, shapes in the metadata) and return the number of maps.

    Biases are not stored: :func:`load_factors` keeps the bias of the map it replaces.
    """
    from safetensors.torch import save_file

    modules = _littlebit_modules(adapter)
    tensors = {f"{key}.{buf}": getattr(m, buf).contiguous().cpu() for key, m in modules.items() for buf in _BUFFERS}
    shapes = {key: [m.in_features, m.out_features, m.rank, m.paths] for key, m in modules.items()}
    save_file(tensors, str(path), metadata={"format": "economize.littlebit/1", "modules": json.dumps(shapes)})
    return len(modules)


def load_factors(path: str | Path) -> dict[str, tuple[list[int], dict[str, Tensor]]]:
    """The maps of a :func:`save_factors` file: ``key -> ([in, out, rank, paths], {buffer: tensor})``."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != "economize.littlebit/1":
            raise ConfigError(f"{path} is not a LittleBit factor file")
        shapes = json.loads(metadata["modules"])
        return {
            key: (shape, {buf: handle.get_tensor(f"{key}.{buf}") for buf in _BUFFERS}) for key, shape in shapes.items()
        }


def _load_model(adapter: ArchAdapter, config: LittleBitConfig) -> dict[str, Any]:
    stored = load_factors(config.factors)
    counts = {"modules": 0, "kept_dense": 0, "weights": 0, "bits": 0}
    ranks: dict[str, int] = {}
    with WeightTransaction() as transaction:
        for key, ((in_features, out_features, rank, paths), buffers) in stored.items():
            index, _, name = key.partition(".")
            layer = adapter.layer(int(index))
            original = layer.get_submodule(name)
            if not isinstance(original, nn.Linear) or tuple(original.weight.shape) != (out_features, in_features):
                raise ConfigError(f"{config.factors}: {key} does not match the model's {type(original).__name__}")
            weight = original.weight
            module = LittleBitLinear(
                in_features, out_features, rank, paths, original.bias is not None, config.backend,
                weight.device, weight.dtype,
            )
            for buf, value in buffers.items():
                getattr(module, buf).copy_(value)
            if original.bias is not None:
                module.bias.copy_(original.bias)
            transaction.replace_module(layer, name, module)
            ranks[key] = rank
            counts["modules"] += 1
            counts["weights"] += out_features * in_features
            counts["bits"] += bits_of(out_features, in_features, rank, paths)
    return {
        "method": "factors",
        "factors": config.factors,
        "bpw": counts["bits"] / max(counts["weights"], 1),
        **counts,
        "ranks": ranks,
    }
