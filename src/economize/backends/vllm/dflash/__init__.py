"""vLLM serving of a DFlash drafter trained or loaded with economize.

vLLM ships its own DFlash draft model (``DFlashQwen3ForCausalLM``) and proposer (``method="dflash"``), so this
subpackage only registers the ``"vllm/dflash"`` exporter (:class:`DFlashVllmExporter`), which writes the drafter in
the z-lab or the ``speculators`` layout.
"""

from __future__ import annotations

from .export import DFlashVllmExporter

__all__ = ["DFlashVllmExporter"]
