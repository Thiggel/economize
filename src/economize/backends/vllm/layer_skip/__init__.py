"""vLLM serving of layer-skip self-drafters, through the jump drafter's plugin.

A draft path that runs the target's layers minus a skip set is exported as a jump drafter with ``m = u = 0``, the
parameter-free ``"embedding"`` predictor and the skip set as holes in the path
(:mod:`economize.techniques.layer_skip.export`). The export carries the ``economize_jump`` block and the
``EconomizeJumpEagle3`` architecture, so the jump drafter's vLLM entry
(:data:`economize.backends.vllm.jump.JUMP_DRAFTER`)
runs it; this subpackage only registers the ``"vllm/layer_skip"`` exporter (:class:`LayerSkipVllmExporter`).
"""

from __future__ import annotations

from .export import LayerSkipVllmExporter

__all__ = ["LayerSkipVllmExporter"]
