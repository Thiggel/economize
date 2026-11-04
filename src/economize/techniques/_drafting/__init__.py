"""Helpers shared by the drafter techniques (``jump``, ``early_exit``, ``eagle3``, ``medusa``, ``vendor_mtp``,
``draft_model``, ``frspec``, ``lookup``, ``hybrid``). The leading underscore keeps the package out of technique
autodiscovery.

* :mod:`.formats`: weight-file I/O (:func:`first_file`, :func:`load_state_file`, :func:`detect_config_dir`,
  :func:`require_weights`, :func:`state_shapes`).
* :mod:`.drafters`: drafter mechanics (:class:`GraphedStep`, :class:`BoundedDrafter`, :class:`CommittedColumns`,
  :func:`previous_states`, :class:`ObserveCosts`, :func:`take_rows`) and :class:`DraftModules`, the protocol of a
  drafter technique that names its own modules.
* :mod:`.training`: :class:`StepDrafterDefaults`, the common answers of ``economize.train``'s step protocol, and the
  learning rates of draft-path LoRA factors and layer copies (:data:`LORA_LR`, :data:`BLOCK_LR`).
* :mod:`.head`: draft heads restricted to a draft vocabulary (:class:`HeadSliceable`, :class:`HeadInput`,
  :class:`LinearSlice`, :class:`SlicedAdapterHead`, :func:`adapter_head_slice`).
* :mod:`.greedy_head`: the greedy choice of a draft head without writing its logits (:class:`GreedyHead`,
  :class:`FusedGreedyHead`, the fused Triton LM-head kernel :func:`greedy_select`, :class:`LogitsGreedyHead`,
  :func:`adapter_greedy_head`).
* :mod:`.chain`: :class:`ChainGraphs`, a greedy proposal of ``k`` draft steps replayed from one CUDA graph.
* :mod:`.tree_step`: :func:`tree_step_context`, the context of a draft step over a tree's frontier (dynamic draft
  trees, :mod:`economize.decoding.dynamic_tree`).
* :mod:`.wrapping`: drafter techniques around other drafter techniques (:class:`WrappingConfig`,
  :class:`WrappingTechnique`, :class:`DelegatingDrafter`, :class:`NotWrappable`, :class:`PartialDrafter`).
"""

from __future__ import annotations

from .chain import ChainGraphs
from .drafters import (
    BoundedDrafter,
    CommittedColumns,
    DraftModules,
    GraphedStep,
    ObserveCosts,
    StepFn,
    previous_states,
    take_rows,
)
from .formats import (
    JSONObject,
    detect_config_dir,
    first_file,
    load_state_file,
    require_weights,
    state_shapes,
)
from .greedy_head import FusedGreedyHead, GreedyHead, LogitsGreedyHead, adapter_greedy_head, greedy_select
from .head import AdapterHeadSlicing, HeadInput, HeadSliceable, LinearSlice, SlicedAdapterHead, adapter_head_slice
from .training import BLOCK_LR, LORA_LR, StepDrafterDefaults
from .tree_step import tree_step_context
from .wrapping import (
    DelegatingDrafter,
    NotWrappable,
    PartialDrafter,
    WrappingConfig,
    WrappingTechnique,
    drafting_class,
)

__all__ = [
    "BLOCK_LR",
    "LORA_LR",
    "AdapterHeadSlicing",
    "BoundedDrafter",
    "ChainGraphs",
    "CommittedColumns",
    "DelegatingDrafter",
    "DraftModules",
    "FusedGreedyHead",
    "GraphedStep",
    "GreedyHead",
    "HeadInput",
    "HeadSliceable",
    "JSONObject",
    "LinearSlice",
    "LogitsGreedyHead",
    "NotWrappable",
    "ObserveCosts",
    "PartialDrafter",
    "SlicedAdapterHead",
    "StepDrafterDefaults",
    "StepFn",
    "WrappingConfig",
    "WrappingTechnique",
    "adapter_greedy_head",
    "adapter_head_slice",
    "detect_config_dir",
    "drafting_class",
    "first_file",
    "greedy_select",
    "load_state_file",
    "previous_states",
    "require_weights",
    "state_shapes",
    "take_rows",
    "tree_step_context",
]
