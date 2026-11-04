"""Decoding across several devices.

* :class:`AsyncSpeculativeDecoder`: speculative speculative decoding (SSD). A draft model on a second GPU drafts the
  next round for the likeliest verification outcomes while the target verifies the current one; the output is
  exactly the target's (greedy, or speculative sampling).
* :class:`SpeculationWorker`: the drafter side of SSD, extended by subclassing (:class:`ModelWorker` runs any causal
  LM with an economize architecture adapter).
* :class:`LinkedSpeculativeDecoder` and :class:`LinkedWorker`: SSD with a drafter that runs part of the target's own
  computation and drafts from the target's exact states (for example :class:`DraftLayerWorker` for
  EAGLE-1/2 and EAGLE-3). :func:`speculative_decoder` builds the decoder of any
  drafter replica from the workers registered for its technique in :data:`DRAFTER_WORKERS`.
* :mod:`economize.distributed.tp`: tensor parallelism through transformers' ``tp_plan`` under ``torchrun`` (plan
  checks and loading). A split model runs through its own forward and through economize's layer runner (economize's
  own decoding works on it with every adaptation); only SSD rejects split models.
"""

from __future__ import annotations

from .draft_layer_worker import (
    DRAFT_LAYER_WORKERS,
    DraftLayerSpeculativeDecoder,
    DraftLayerWorker,
    Eagle3SpeculativeDecoder,
    Eagle3Worker,
    GraphedDraftLayerSpeculativeDecoder,
    GraphedDraftLayerWorker,
    GraphedEagle3SpeculativeDecoder,
    GraphedEagle3Worker,
)
from .drafters import DRAFTER_WORKERS, DrafterWorkers, drafter_workers, speculative_decoder
from .graphed import GraphedJumpWorker, GraphedLinkedSpeculativeDecoder, GraphedSelfSpeculativeDecoder
from .linked import CapturingTargetWorker, LinkedDrafter, LinkedSpeculativeDecoder, LinkedWorker
from .self_drafting import JumpWorker, SelfSpeculativeDecoder
from .ssd import AsyncSpeculativeDecoder, SSDConfig, SSDResult
from .workers import ModelWorker, Speculation, SpeculationWorker, Spine, TargetWorker

__all__ = [
    "DRAFTER_WORKERS",
    "DRAFT_LAYER_WORKERS",
    "AsyncSpeculativeDecoder",
    "CapturingTargetWorker",
    "DraftLayerSpeculativeDecoder",
    "DraftLayerWorker",
    "DrafterWorkers",
    "Eagle3SpeculativeDecoder",
    "Eagle3Worker",
    "GraphedDraftLayerSpeculativeDecoder",
    "GraphedDraftLayerWorker",
    "GraphedEagle3SpeculativeDecoder",
    "GraphedEagle3Worker",
    "GraphedJumpWorker",
    "GraphedLinkedSpeculativeDecoder",
    "GraphedSelfSpeculativeDecoder",
    "JumpWorker",
    "LinkedDrafter",
    "LinkedSpeculativeDecoder",
    "LinkedWorker",
    "ModelWorker",
    "SSDConfig",
    "SSDResult",
    "SelfSpeculativeDecoder",
    "Speculation",
    "SpeculationWorker",
    "Spine",
    "TargetWorker",
    "drafter_workers",
    "speculative_decoder",
]
