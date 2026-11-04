"""The decoder rejects combinations of verifier and drafter it cannot run when it is built, before any decoding.

A verifier that commits drafts as they are proposed sees no exact pass over them, so it cannot verify draft trees,
and a drafter that drafts trees needs a verifier that verifies them. :class:`~economize.decoding.CycleDecoder` checks
both up front; :meth:`~economize.decoding.Drafter.drafts_trees` tells it which drafters draft trees.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch

from economize import ConfigError, DecodeConfig
from economize.arch import adapter_for
from economize.decoding import VERIFIERS, AcceptAllVerifier, CostLog, CycleDecoder, Verifier
from economize.runtime import LayerRunner
from helpers import ScriptedDrafter, ScriptedTreeDrafter, make_tiny_model

SCRIPT = [torch.arange(40) % 128]


class _Tree(ScriptedTreeDrafter):
    def drafts_trees(self, verifier: Verifier) -> bool:
        return True


def _runner() -> LayerRunner:
    return LayerRunner(adapter_for(make_tiny_model()))


def test_a_tree_drafter_needs_a_tree_verifier() -> None:
    runner = _runner()
    with pytest.raises(ConfigError, match="drafts trees"):
        CycleDecoder(runner, _Tree(SCRIPT), DecodeConfig(cycle_length=3, lossless=False))
    CycleDecoder(runner, _Tree(SCRIPT), DecodeConfig(cycle_length=3))  # the greedy verifier verifies trees
    CycleDecoder(runner, _Tree(SCRIPT), DecodeConfig(cycle_length=1, lossless=False))  # never drafts
    CycleDecoder(runner, ScriptedDrafter(SCRIPT), DecodeConfig(cycle_length=3, lossless=False))


def test_a_committing_verifier_cannot_claim_trees() -> None:
    class Both(AcceptAllVerifier):
        supports_trees: ClassVar[bool] = True

    VERIFIERS.register("test_commits_trees")(Both)
    try:
        with pytest.raises(ConfigError, match="commits drafts"):
            CycleDecoder(_runner(), ScriptedDrafter(SCRIPT),
                         DecodeConfig(cycle_length=3, verifier="test_commits_trees"))
    finally:
        VERIFIERS.unregister("test_commits_trees")


# ---- cost logs -----------------------------------------------------------------------------------------------------
def test_truncate_after_step_drops_later_draft_steps_and_keeps_the_rest() -> None:
    log = CostLog()
    log.record("draft", 1, 2, 2, 4, step=1)  # the previous cycle, before the mark
    mark = len(log)
    log.record("exact", 4, 0, 1, 4)
    for step in range(3):
        log.record("catch_up", 2, 4 + step, 1, 1)  # a drafter's own call before the step it serves
        log.record("draft", 1, 4 + step, 2, 4, step=step)
    kinds = [(e.kind, e.step) for e in log.events]
    log.truncate_after_step(5, mark)  # a step that was not recorded: every event stays
    assert [(e.kind, e.step) for e in log.events] == kinds
    log.truncate_after_step(1, mark)  # the later step and its catch-up call were wasted
    assert [(e.kind, e.step) for e in log.events] == kinds[:6]
    log.truncate_after_step(1)  # from the start, the previous cycle's step 1 matches first
    assert [(e.kind, e.step) for e in log.events] == kinds[:1]
