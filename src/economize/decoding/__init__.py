"""Decoding: the draft-and-refresh loop and its pluggable parts.

* :class:`CycleDecoder` runs batched, ragged generation with an optional :class:`Drafter`.
* :data:`VERIFIERS` (greedy, rejection, typical, accept_all) decide which drafts survive an exact pass.
* :data:`POLICIES` (fixed, confidence, dynamic_tree, sequoia, specexec) decide when a row stops drafting within a
  cycle, or (the tree policies) turn a chain drafter that implements :class:`TreeStepDrafter` into a tree drafter:
  EAGLE-2 dynamic trees, static Sequoia trees, or SpecExec's best-first search.
* Token selectors (:class:`GreedySelector`, :class:`TopPSampler`) turn logits into tokens; the optional capability
  protocols (:class:`BatchSelector`, :class:`DistributionSelector`, :class:`BatchDistributions`,
  :class:`SeededSelector`) say what else a custom selector can do.
* :class:`StopChecker` turns a :data:`StopCondition` into the check run after every committed token.
* :class:`CostLog` records every model call for cost accounting.
* :class:`DecodeGraphs` holds the CUDA graphs and static cache of a decoder with ``cuda_graphs=True``.
"""

from __future__ import annotations

from .capabilities import BatchDistributions, BatchSelector, DistributionSelector, SeededSelector
from .config import DecodeConfig
from .cost import CostEvent, CostKind, CostLog, LayerFootprint
from .dynamic_tree import DynamicTreeDrafter, DynamicTreePolicy, TreeOptions, TreeStepDrafter
from .graphs import DecodeGraphs
from .loop import CycleDecoder, DecodeResult, StopCondition, VerifyObserver
from .policies import POLICIES, ConfidenceStop, FixedLength
from .protocols import (
    Draft,
    Drafter,
    DraftPolicy,
    TokenSelector,
    Verifier,
    VerifyEvent,
    WrappableDrafter,
    draft_footprint,
)
from .request import DraftRequest
from .selectors import GreedySelector, TopPSampler
from .sequoia import (
    AcceptanceProfile,
    SequoiaDrafter,
    SequoiaPolicy,
    StaticTree,
    measure_acceptance,
    sequoia_tree,
)
from .specexec import SearchOptions, SpecExecDrafter, SpecExecPolicy
from .stop import StopChecker
from .tree_search import TreePolicy, TreeSearchDrafter
from .verifiers import VERIFIERS, AcceptAllVerifier, GreedyVerifier, RejectionVerifier, TypicalVerifier

__all__ = [
    "POLICIES",
    "VERIFIERS",
    "AcceptAllVerifier",
    "AcceptanceProfile",
    "BatchDistributions",
    "BatchSelector",
    "ConfidenceStop",
    "CostEvent",
    "CostKind",
    "CostLog",
    "CycleDecoder",
    "DecodeConfig",
    "DecodeGraphs",
    "DecodeResult",
    "DistributionSelector",
    "Draft",
    "DraftPolicy",
    "DraftRequest",
    "Drafter",
    "DynamicTreeDrafter",
    "DynamicTreePolicy",
    "FixedLength",
    "GreedySelector",
    "GreedyVerifier",
    "LayerFootprint",
    "RejectionVerifier",
    "SearchOptions",
    "SeededSelector",
    "SequoiaDrafter",
    "SequoiaPolicy",
    "SpecExecDrafter",
    "SpecExecPolicy",
    "StaticTree",
    "StopChecker",
    "StopCondition",
    "TokenSelector",
    "TopPSampler",
    "TreeOptions",
    "TreePolicy",
    "TreeSearchDrafter",
    "TreeStepDrafter",
    "TypicalVerifier",
    "Verifier",
    "VerifyEvent",
    "VerifyObserver",
    "WrappableDrafter",
    "draft_footprint",
    "measure_acceptance",
    "sequoia_tree",
]
