"""vLLM serving of Medusa heads trained with economize.

vLLM ships its own Medusa draft model (``vllm.model_executor.models.medusa.Medusa``) and proposer, so this subpackage
only registers the ``"vllm/medusa"`` exporter (:class:`MedusaVllmExporter`). No
:class:`~economize.backends.vllm.VllmDrafter` is registered (vLLM's model is used as is); pass ``method="medusa"`` to
:func:`economize.backends.vllm.speculative_config` for an exported folder, or serve the whole model with
:func:`economize.backends.vllm.serving.to_vllm`, which sets it.
"""

from __future__ import annotations

from .export import MedusaVllmExporter

__all__ = ["MedusaVllmExporter"]
