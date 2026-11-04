"""Run-time backends of TEAL: how calibrated thresholds act on the linear maps of the base model.

``dense`` masks the input of every map with a forward pre-hook and leaves the matmul dense. It is the numerical
reference and gives no speedup by itself: the savings of TEAL come from a sparse GEMV kernel that skips the weight
columns of zeroed inputs during single-token decoding (TEAL's Triton kernel, about 1.5-1.8x at 40-50% sparsity on
memory-bound batch-1 decoding). Such a kernel is added as a new registered backend that replaces the maps' forward
in :meth:`TealBackend.install`; nothing else changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from ...core.metadata import Saving, Simulation
from ...registry import Registry
from .sparsify import PhasePolicy, sparsify_tokens

__all__ = ["TEAL_BACKENDS", "DenseBackend", "Installed", "TealBackend"]


class Installed:
    """Handles of an installed backend; :meth:`remove` restores the original maps.

    Attributes:
        handles: One handle per sparsified map.
        support: Further hooks that serve all maps (the layer-layout tracking of
            :func:`.sparsify.track_layer_tokens`).
    """

    def __init__(self, handles: list[RemovableHandle], support: list[RemovableHandle] | None = None) -> None:
        self.handles = handles
        self.support = support or []

    def remove(self) -> None:
        for handle in (*self.handles, *self.support):
            handle.remove()
        self.handles, self.support = [], []


class TealBackend(ABC):
    """Makes linear maps act on sparsified inputs."""

    @abstractmethod
    def install(
        self, modules: Mapping[str, nn.Module], thresholds: Mapping[str, Tensor], policy: PhasePolicy
    ) -> Installed:
        """Sparsify the input of every map in ``modules`` with its threshold (a 0-dim tensor on the map's device,
        read at call time, so updating it in place changes the installed behaviour) where ``policy`` says so
        (:func:`.sparsify.sparsify_tokens`)."""


TEAL_BACKENDS: Registry[type[TealBackend]] = Registry("TEAL backend", "economize.teal_backends")


@TEAL_BACKENDS.register("dense")
class DenseBackend(TealBackend):
    """Masking pre-hooks in front of the unchanged dense matmul (reference numerics, no speedup)."""

    simulation = Simulation(Saving.COMPUTE | Saving.LATENCY, "masks the inputs of the dense matmul")

    def install(
        self, modules: Mapping[str, nn.Module], thresholds: Mapping[str, Tensor], policy: PhasePolicy | str
    ) -> Installed:
        policy = PhasePolicy.coerce(policy)
        handles = []
        for name, module in modules.items():

            def hook(
                _module: nn.Module, args: tuple[Any, ...], threshold: Tensor = thresholds[name]
            ) -> tuple[Any, ...]:
                return (sparsify_tokens(args[0], threshold, policy), *args[1:])

            handles.append(module.register_forward_pre_hook(hook))
        return Installed(handles)
