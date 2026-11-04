"""Drafter exporters: how a drafter technique is written out for one serving backend.

An exporter turns the trained drafter of one technique into a folder that one inference engine loads: a
``config.json`` and a ``model.safetensors`` with the tensors the drafter owns. Exporters are registered per
``(backend, technique)`` pair in :data:`EXPORTERS` under the key ``"<backend>/<technique>"``, so a new backend or a
new drafter technique is one registered class and no existing module is edited::

    from economize.backends import DrafterExporter, register_exporter

    @register_exporter("my_engine", "eagle3")
    class Eagle3MyEngineExporter(DrafterExporter):
        def config(self, model, **options):
            technique = self.technique_of(model)
            adapter = technique.adapter
            return {..., "shared": [adapter.module_path(adapter.lm_head)]}  # module names from the adapter

        def tensors(self, model):
            return {name: t.float() for name, t in self.technique_of(model).state_dict().items()}

    economize.backends.export_drafter(model, "out", backend="my_engine")

Module names of the target (layers, final norm, embedding, LM head) come from the model's
:class:`~economize.arch.ArchAdapter` (:meth:`~economize.arch.ArchAdapter.module_path`), never from literal strings, so
an exporter works for every family with an adapter. Other packages register exporters through the
``economize.exporters`` entry-point group (name ``"<backend>/<technique>"``, value the exporter class). The built-in
exporters are registered by the backend subpackages, which :mod:`economize.backends` imports.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import torch
from torch import nn

from ..errors import ConfigError, EconomizeError
from ..registry import Registry

__all__ = [
    "EXPORTERS",
    "DrafterExporter",
    "UnsupportedBackendError",
    "export_drafter",
    "exporter_for",
    "exporter_key",
    "register_exporter",
]


class UnsupportedBackendError(EconomizeError, NotImplementedError):
    """A backend is registered for a technique but cannot serve it yet."""


class DrafterExporter(ABC):
    """Writes the drafter of one technique for one serving backend.

    Subclasses implement :meth:`config` and :meth:`tensors`; :meth:`export` writes both. The class attributes are
    set by :func:`register_exporter`.

    Attributes:
        backend: The serving backend (``"vllm"``).
        technique: The technique name whose drafter is exported (``"early_exit"``).
        config_file: File name of the configuration in the exported folder.
        weights_file: File name of the tensors in the exported folder.
    """

    backend: ClassVar[str] = ""
    technique: ClassVar[str] = ""
    config_file: ClassVar[str] = "config.json"
    weights_file: ClassVar[str] = "model.safetensors"

    def technique_of(self, model: Any) -> nn.Module:
        """The exported technique of an EfficientModel (or ``model`` itself if it is that technique).

        Raises:
            ConfigError: The model has no such technique.
        """
        if getattr(model, "name", None) == self.technique and not hasattr(model, "techniques"):
            return model
        techniques = getattr(model, "techniques", None) or {}
        if self.technique not in techniques:
            raise ConfigError(f"the model has no {self.technique} technique to export for {self.backend}")
        return techniques[self.technique]

    @abstractmethod
    def config(self, model: Any, **options: Any) -> dict[str, Any]:
        """The content of :attr:`config_file`. ``options`` are the backend's export options."""

    @abstractmethod
    def tensors(self, model: Any) -> dict[str, torch.Tensor]:
        """The drafter-owned tensors written to :attr:`weights_file`, by their name in that file."""

    def live_tensors(self, model: Any) -> dict[str, torch.Tensor]:
        """The tensors of :meth:`tensors` under the same names, as they live in the model (detached, no copy and
        no dtype or device change), for a weight sync into a running engine. The default returns :meth:`tensors`;
        an exporter whose :meth:`tensors` converts overrides it."""
        return self.tensors(model)

    def metadata(self) -> dict[str, str]:
        """The ``safetensors`` metadata of :attr:`weights_file`."""
        return {"format": "pt"}

    def export(self, model: Any, path: str | Path, **options: Any) -> Path:
        """Write :attr:`config_file` and :attr:`weights_file` to the folder ``path`` (created if missing).

        Returns:
            ``path`` as a :class:`~pathlib.Path`.
        """
        from safetensors.torch import save_file

        config = self.config(model, **options)
        tensors = self.tensors(model)
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / self.config_file).write_text(json.dumps(config, indent=2, sort_keys=True, default=str) + "\n")
        save_file(tensors, str(directory / self.weights_file), metadata=self.metadata())
        return directory


#: Exporter classes by ``"<backend>/<technique>"`` (see :func:`exporter_key`). Built as ``cls()``.
EXPORTERS: Registry[type[DrafterExporter]] = Registry("drafter exporter", "economize.exporters")

_E = TypeVar("_E", bound=type[DrafterExporter])


def exporter_key(backend: str, technique: str) -> str:
    """The :data:`EXPORTERS` key of a backend and a technique: ``"<backend>/<technique>"``."""
    return f"{backend}/{technique}"


def register_exporter(backend: str, technique: str, *, override: bool = False) -> Callable[[_E], _E]:
    """Class decorator registering a :class:`DrafterExporter` for ``technique`` on ``backend``."""

    def add(cls: _E) -> _E:
        cls.backend, cls.technique = backend, technique
        EXPORTERS.register(exporter_key(backend, technique), cls, override=override)
        return cls

    return add


def _technique_names(model: Any) -> list[str]:
    techniques = getattr(model, "techniques", None)
    if isinstance(techniques, (Mapping, nn.ModuleDict)):
        return list(techniques.keys())
    name = getattr(model, "name", None)
    return [name] if isinstance(name, str) else []


def exporter_for(model: Any, backend: str, technique: str | None = None) -> DrafterExporter:
    """The exporter of the model's drafter technique for ``backend``.

    Args:
        model: An EfficientModel (or a technique).
        backend: The serving backend.
        technique: The technique to export; by default the one attached technique with an exporter for ``backend``.

    Raises:
        UnknownNameError: ``technique`` is given and has no exporter for ``backend``.
        ConfigError: No attached technique, or more than one, has an exporter for ``backend``.
    """
    if technique is not None:
        return EXPORTERS.get(exporter_key(backend, technique))()
    names = _technique_names(model)
    found = [name for name in names if exporter_key(backend, name) in EXPORTERS]
    if len(found) != 1:
        problem = "none of" if not found else "several of"
        raise ConfigError(
            f"{problem} the model's techniques {names} can be exported for {backend!r} (exporters: "
            f"{EXPORTERS.names()}); pass technique=... or register one with economize.backends.register_exporter"
        )
    return EXPORTERS.get(exporter_key(backend, found[0]))()


def export_drafter(model: Any, path: str | Path, *, backend: str, technique: str | None = None,
                   **options: Any) -> Path:
    """Export the model's drafter for ``backend`` to the folder ``path`` (see :func:`exporter_for`).

    ``options`` go to the exporter (for vLLM: ``format``, ``num_speculative_tokens``, ``verifier``).
    """
    return exporter_for(model, backend, technique).export(model, path, **options)
