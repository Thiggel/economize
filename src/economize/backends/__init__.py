"""Serving backends: economize drafters inside inference engines.

Each backend lives in its own subpackage and imports its engine lazily, so importing ``economize.backends.<name>``
works without the engine installed. The backends know no technique. Each drafter technique defines its exported
format (typically the target's ``config.json`` with a block that describes the drafter, and the drafter-owned
tensors in ``model.safetensors``), and an exporter writes it for one engine. The layers a self-drafter shares with
the target, the final norm, the embedding and the LM head are never exported: an engine runs the target's own
modules, so a policy update of the target reaches the drafter at once.

A drafter is written for a backend by the :class:`DrafterExporter` registered in :data:`EXPORTERS` for that
``(backend, technique)`` pair (:mod:`economize.backends.exporters`); :func:`export_drafter` picks it. Importing this
package imports every backend subpackage (found with :func:`economize.registry.import_submodules`, so no list is
edited), which registers the built-in exporters:

* :mod:`economize.backends.vllm`: the ``vllm.general_plugins`` entry point, the ``"vllm/*"`` exporters and the
  configuration builders for ``vllm serve``, TRL's server mode and verl.
* :mod:`economize.backends.sglang`: the ``sglang.srt.plugins`` entry point (economize's self-drafters as an EAGLE-3
  draft model), the ``"sglang/*"`` exporters and the serving plans for ``sglang.Engine`` and ``launch_server``.
* The runtimes that run a whole model on their own kernels (:mod:`economize.backends.gguf` for llama.cpp,
  :mod:`economize.backends.onnx` for ONNX Runtime, :mod:`economize.backends.mlx` for MLX,
  :mod:`economize.backends.trtllm` for TensorRT-LLM): each is an
  :class:`~economize.backends.runtime_export.ExportRuntime` with its own technique adapters, reported by
  :func:`economize.backend_support` (docs/exports.md).

Built-in exporters register with the :func:`register_exporter` decorator when their subpackage is imported, and
nowhere else (pyproject.toml declares no built-in ``economize.exporters`` entry point, so no exporter is registered
twice). A new backend is a new subpackage here, or a separate package that registers its exporters through the
``economize.exporters`` entry-point group.
"""

from __future__ import annotations

from ..registry import import_submodules
from .exporters import (
    EXPORTERS,
    DrafterExporter,
    UnsupportedBackendError,
    export_drafter,
    exporter_for,
    exporter_key,
    register_exporter,
)

__all__ = [
    "EXPORTERS",
    "DrafterExporter",
    "UnsupportedBackendError",
    "export_drafter",
    "exporter_for",
    "exporter_key",
    "register_exporter",
]

import_submodules(__name__, __path__)
