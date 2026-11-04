"""vLLM serving of DMTD drafters, through the jump drafter's plugin.

DMTD's draft step is the jump with ``m = u = L - L_d``, ``r = L`` and the parameter-free ``"embedding"`` predictor
(:mod:`economize.techniques.dynamic_depth.dmtd_export`). The export carries the ``economize_jump`` block and the
``EconomizeJumpEagle3`` architecture, so the jump drafter's vLLM entry
(:data:`economize.backends.vllm.jump.JUMP_DRAFTER`) runs it; this subpackage only registers the ``"vllm/dmtd"``
exporter (:class:`DmtdVllmExporter`). vLLM verifies every draft, so DMTD is served as a lossless self-speculative
drafter; its lossy cycles (every draft kept) have no vLLM equivalent.
"""

from __future__ import annotations

from .export import DmtdVllmExporter

__all__ = ["DmtdVllmExporter"]
