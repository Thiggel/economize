"""vLLM serving of an EAGLE-3 drafter trained with economize.

vLLM ships its own EAGLE-3 draft model (``Eagle3LlamaForCausalLM``) and proposer, so this subpackage only registers
the ``"vllm/eagle3"`` exporter (:class:`Eagle3VllmExporter`), which writes the drafter in the official layout. No
:class:`~economize.backends.vllm.VllmDrafter` is registered: :func:`economize.backends.vllm.speculative_config`
falls back to vLLM's ``"eagle3"`` method for the exported folder.
"""

from __future__ import annotations

from .export import Eagle3VllmExporter

__all__ = ["Eagle3VllmExporter"]
