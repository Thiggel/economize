"""Drafter-only quantization: quantize the modules a drafter owns and leave the verified base model untouched.

Speculative decoding verifies every drafted token with the full-precision model, so quantizing only the drafter is
lossless: it changes the acceptance rate and the cost of a draft step, never the output. The hook has three parts.

* :func:`draft_modules` selects the drafter's own linear maps. A drafter may choose them itself
  (:class:`~economize.techniques._drafting.DraftModules`); otherwise every ``nn.Linear`` in its module tree is a
  candidate, minus modules whose weight is a base model parameter (shared layers are never modified in place) and
  minus the ``exclude`` globs (LoRA factors by default, whose rank-r products gain nothing from quantization).
* :func:`quantize_draft` quantizes them (round to nearest; no calibration) and returns what :func:`restore_draft`
  needs to undo it.
* :func:`quantized_copy` builds a quantized copy of a shared module, e.g. the base layers that a frozen or LoRA
  self-drafter runs. Installing such a copy on the draft path needs a layer-override hook in the drafter.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from ...core.calibration import select_modules
from .._drafting.drafters import DraftModules
from .._weights.snapshot import ParameterSnapshot
from .config import DraftQuantConfig
from .model import quantizable, quantize_modules

__all__ = ["DraftSnapshot", "draft_modules", "quantize_draft", "quantized_copy", "restore_draft"]


def draft_modules(
    drafter: nn.Module, config: DraftQuantConfig, base_parameters: Iterable[nn.Parameter] = ()
) -> dict[str, nn.Linear]:
    """The drafter's own linear maps selected by ``config`` (shared base-model weights are excluded).

    A weight counts as shared when it lives in the same storage as a base parameter, so a drafter that wraps a base
    weight in a new ``nn.Parameter`` (or holds a view of one) never gets it quantized in place.
    """
    shared = {_storage_key(p) for p in base_parameters}
    own = drafter.draft_modules() if isinstance(drafter, DraftModules) else drafter.named_modules()
    candidates: dict[str, nn.Module] = dict(own)
    linears = {
        name: module
        for name, module in candidates.items()
        if isinstance(module, nn.Linear) and _storage_key(module.weight) not in shared
    }
    selected = select_modules(linears, config.targets, config.exclude)
    return quantizable(selected, config.group_size, "draft_quant: ")


def _storage_key(tensor: Tensor) -> tuple[str, int]:
    """The device and address of the storage behind ``tensor``; the object identity for a tensor without one (a
    tensor subclass that wraps several tensors, a meta or empty tensor)."""
    try:
        pointer = tensor.untyped_storage().data_ptr()
    except (NotImplementedError, RuntimeError, TypeError):
        pointer = 0
    if pointer == 0:
        return ("object", id(tensor))
    return (str(tensor.device), pointer)


@dataclass
class DraftSnapshot:
    """The quantized drafter modules, their original parameters (a :class:`ParameterSnapshot` on the CPU) and the
    quantization summary."""

    modules: dict[str, nn.Linear]
    params: ParameterSnapshot = field(default_factory=ParameterSnapshot)
    summary: dict[str, Any] = field(default_factory=dict)


@torch.no_grad()
def quantize_draft(modules: dict[str, nn.Linear], config: DraftQuantConfig) -> DraftSnapshot:
    """Quantize ``modules`` in place and return a snapshot that :func:`restore_draft` reverts.

    All or nothing: if quantizing or converting any module fails, every module gets its original parameters back
    before the error propagates.
    """
    snapshot = DraftSnapshot(modules=dict(modules))
    for module in modules.values():
        snapshot.params.take(module)
    try:
        snapshot.summary = quantize_modules(modules, config)
    except BaseException:
        restore_draft(snapshot)
        raise
    for module in modules.values():
        module.weight.requires_grad_(False)  # quantized weights are not trained further
    return snapshot


def restore_draft(snapshot: DraftSnapshot) -> None:
    """Put the original parameters back (undoes :func:`quantize_draft` for every backend): each weight is an
    ``nn.Parameter`` on its original device again, with its original ``requires_grad``."""
    snapshot.params.restore()


@torch.no_grad()
def quantized_copy(module: nn.Module, config: DraftQuantConfig) -> tuple[nn.Module, dict[str, Any]]:
    """A deep copy of ``module`` with its linear maps quantized (for shared layers on a drafter's path)."""
    clone = copy.deepcopy(module).requires_grad_(False)
    linears = {name: m for name, m in clone.named_modules() if isinstance(m, nn.Linear)}
    selected = quantizable(select_modules(linears, config.targets, config.exclude), config.group_size, "draft_quant: ")
    return clone, quantize_modules(selected, config)
