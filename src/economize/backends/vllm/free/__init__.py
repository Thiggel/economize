"""vLLM serving of FREE drafters, through the jump drafter's plugin.

FREE's shallow model is an early exit after layer ``L_S`` that shares the target's layers, so it is exported as the
early-exit drafter is (a jump drafter with ``m = u = 0``, :mod:`economize.backends.vllm.early_exit`) and runs on the
jump drafter's vLLM entry (:data:`economize.backends.vllm.jump.JUMP_DRAFTER`). This subpackage only registers the
``"vllm/free"`` exporter (:class:`FreeVllmExporter`). vLLM verifies every draft and drafts a fixed number of tokens,
so FREE is served as a lossless self-speculative drafter; its kept shallow exits and the softmax-gap stop run on the
HF backend only.
"""

from __future__ import annotations

from .export import FreeVllmExporter

__all__ = ["FreeVllmExporter"]
