"""vLLM serving of early-exit drafters, through the jump drafter's plugin.

An early exit after layer ``r`` is exported as a jump drafter with ``m = u = 0`` and the parameter-free
``"embedding"`` predictor (:mod:`economize.techniques.early_exit.export`). The export carries the ``economize_jump``
block and the ``EconomizeJumpEagle3`` architecture, so the jump drafter's vLLM entry
(:data:`economize.backends.vllm.jump.JUMP_DRAFTER`) runs it; this subpackage only registers the
``"vllm/early_exit"`` exporter (:class:`EarlyExitVllmExporter`).
"""

from __future__ import annotations

from .export import EarlyExitVllmExporter

__all__ = ["EarlyExitVllmExporter"]
