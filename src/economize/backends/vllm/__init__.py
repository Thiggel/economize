"""vLLM backend: serve an economize drafter as vLLM's draft model.

    from economize.backends import vllm as economize_vllm

    economize_vllm.export_drafter(model, "qwen3-8b-drafter")             # config.json + drafter tensors only
    llm = vllm.LLM("Qwen/Qwen3-8B", speculative_config=economize_vllm.speculative_config("qwen3-8b-drafter", 4))

    # the same setting for ``vllm serve``, TRL's ``trl vllm-serve`` and verl
    economize_vllm.serve_args("qwen3-8b-drafter", 4)
    economize_vllm.serve_args("qwen3-8b-drafter", 4, flag="--speculative_config")
    economize_vllm.verl_engine_kwargs("qwen3-8b-drafter", 4)

vLLM finds the drafter through the ``vllm.general_plugins`` entry point ``economize = economize.backends.vllm:register``
(see :mod:`economize.backends.vllm.plugin`); no patch of vLLM, TRL or verl is needed. The modules of this package
know no technique: the plugin installs what the drafters registered in
:data:`~economize.backends.vllm.drafters.VLLM_DRAFTERS` declare, and the export functions pick the exporter
registered for ``"vllm"`` and the model's drafter technique. Each servable technique is a subpackage
``economize.backends.vllm.<technique>`` that registers both. Importing this package imports every such subpackage
(found with :func:`economize.registry.import_submodules`, so no list is edited), for example
:mod:`economize.backends.vllm.early_exit` and :mod:`economize.backends.vllm.eagle3`.

An exported self-drafter shares the target's layers, final norm, embedding and LM head, so a weight sync of the
policy reaches it with no copy. :func:`drafter_weights` names a re-distilled drafter's tensors for an RL weight sync
(:mod:`economize.integrations.trl_vllm`). Names that earlier releases exported from this package stay importable.

Importing this package does not import vLLM.
"""

from __future__ import annotations

from typing import Any

from ...registry import import_submodules
from .drafters import VLLM_DRAFTERS, VllmDrafter, register_vllm_drafter
from .export import drafter_config, drafter_weights, export_drafter
from .spec import serve_args, speculative_config, verl_engine_kwargs

__all__ = [
    "DRAFTER_ARCHITECTURE",
    "VLLM_DRAFTERS",
    "JumpVllmExporter",
    "VllmDrafter",
    "drafter_config",
    "drafter_weights",
    "export_drafter",
    "register",
    "register_vllm_drafter",
    "serve_args",
    "speculative_config",
    "verl_engine_kwargs",
]

import_submodules(__name__, __path__)  # each servable technique registers its exporter and its VllmDrafter


def __getattr__(name: str) -> Any:
    if name in ("DRAFTER_ARCHITECTURE", "JumpVllmExporter"):  # the built-in jump drafter's names, kept here
        from . import jump

        return getattr(jump, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register() -> None:
    """The ``vllm.general_plugins`` entry point (see :func:`economize.backends.vllm.plugin.register`)."""
    from .plugin import register as _register

    _register()
