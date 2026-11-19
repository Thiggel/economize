"""Run-time backends of TEAL: how calibrated thresholds act on the linear maps of the base model.

``dense`` masks the input of every map with a forward pre-hook and leaves the matmul dense. It is the numerical
reference and gives no speedup by itself (it reports a :class:`~economize.Simulation`). The savings of TEAL come from
a sparse GEMV kernel that skips the weight rows of zeroed inputs during decoding: ``triton``
(:mod:`.triton_backend`), ``triton_autotune`` (:mod:`.triton_autotune`) and the default ``auto``
(:mod:`.auto_backend`, the Triton kernel on every map or a :class:`~economize.errors.ConfigError` saying why it
cannot run). A further kernel is added as a new registered backend that replaces the maps' forward in
:meth:`TealBackend.install`; nothing else changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar, Protocol

from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from ...core.metadata import Saving, Simulation, WeightAccess
from ...registry import Registry
from .sparsify import PhasePolicy, sparsify_tokens

__all__ = ["TEAL_BACKENDS", "DenseBackend", "Installed", "Removable", "TealBackend"]


class Removable(Protocol):
    """A hook or patch that undoes itself (a :class:`~torch.utils.hooks.RemovableHandle`, a kernel patch)."""

    def remove(self) -> None: ...


class Installed:
    """Handles of an installed backend; :meth:`remove` restores the original maps.

    Attributes:
        handles: One handle per sparsified map.
        support: Further hooks that serve all maps (the layer-layout tracking of
            :func:`.sparsify.track_layer_tokens`).
        masked: Names of the maps that got the masking pre-hook in front of their dense matmul, so their saving is
            only simulated (every map under ``dense``; the maps the kernel cannot serve under ``triton``).
    """

    def __init__(
        self,
        handles: list[Removable],
        support: list[RemovableHandle] | None = None,
        masked: tuple[str, ...] = (),
    ) -> None:
        self.handles = handles
        self.support: list[RemovableHandle] = support or []
        self.masked = masked

    def remove(self) -> None:
        for handle in (*self.handles, *self.support):
            handle.remove()
        self.handles, self.support, self.masked = [], [], ()


class TealBackend(ABC):
    """Makes linear maps act on sparsified inputs.

    Attributes:
        weight_access: Set when the backend's kernel reads the maps' weights itself, with the weight storages it
            reads (:class:`~economize.WeightAccess`); a technique that leaves the weights in another storage is
            then refused. ``None``: the maps run their own forward.
    """

    weight_access: ClassVar[WeightAccess | None] = None

    @abstractmethod
    def install(
        self, modules: Mapping[str, nn.Module], thresholds: Mapping[str, Tensor], policy: PhasePolicy
    ) -> Installed:
        """Sparsify the input of every map in ``modules`` with its threshold (a 0-dim tensor on the map's device,
        read at call time, so updating it in place changes the installed behaviour) where ``policy`` says so
        (:func:`.sparsify.sparsify_tokens`)."""

    def check(self, modules: Mapping[str, nn.Module]) -> None:
        """Raise a :class:`~economize.errors.ConfigError` when this backend cannot serve ``modules``.

        :class:`~.technique.TealTechnique` calls it before calibrating, so a backend that would fail at install
        time fails before the expensive part. The default accepts every map.
        """


TEAL_BACKENDS: Registry[type[TealBackend]] = Registry("TEAL backend", "economize.teal_backends")


@TEAL_BACKENDS.register("dense")
class DenseBackend(TealBackend):
    """Masking pre-hooks in front of the unchanged dense matmul (reference numerics, no speedup)."""

    simulation = Simulation(Saving.COMPUTE | Saving.LATENCY, "masks the inputs of the dense matmul")

    def install(
        self, modules: Mapping[str, nn.Module], thresholds: Mapping[str, Tensor], policy: PhasePolicy | str
    ) -> Installed:
        policy = PhasePolicy.coerce(policy)
        handles: list[Removable] = []
        for name, module in modules.items():

            def hook(
                _module: nn.Module, args: tuple[Tensor, ...], threshold: Tensor = thresholds[name]
            ) -> tuple[Tensor, ...]:
                return (sparsify_tokens(args[0], threshold, policy), *args[1:])

            handles.append(module.register_forward_pre_hook(hook))
        return Installed(handles, masked=tuple(modules))
