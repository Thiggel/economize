"""The SSD drafter-worker registry (economize.distributed.drafters): built-in drafters resolve to their workers,
subclasses of a registered technique inherit them, and the decoder built from a replica reproduces the target."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import economize
from economize.distributed import (
    DRAFTER_WORKERS,
    DrafterWorkers,
    DraftLayerSpeculativeDecoder,
    GraphedDraftLayerSpeculativeDecoder,
    GraphedSelfSpeculativeDecoder,
    JumpWorker,
    SelfSpeculativeDecoder,
    SSDConfig,
    drafter_workers,
    speculative_decoder,
)
from economize.errors import ConfigError
from economize.techniques.eagle import EagleConfig
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.jump import JumpConfig, JumpTechnique
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, perturb, reference_greedy

NEW = 16


@pytest.fixture(scope="module")
def target() -> nn.Module:
    return make_tiny_model(seed=0)


@pytest.fixture(scope="module")
def prompt() -> torch.Tensor:
    return torch.randint(1, VOCAB, (PROMPT_LEN,), generator=torch.Generator().manual_seed(1))


def replica(config: economize.TechniqueConfig) -> nn.Module:
    model = economize.apply(make_tiny_model(seed=0), config)
    perturb(model.drafting_technique, 0.05, seed=5)
    return model


JUMP = JumpConfig(target_layer=2, source_layer=4)


def test_builtin_drafters_are_registered_under_their_technique_names() -> None:
    assert {JumpConfig.technique, EagleConfig.technique, Eagle3Config.technique} <= set(DRAFTER_WORKERS)


@pytest.mark.parametrize(("config", "eager", "graphed"), [
    (JUMP, SelfSpeculativeDecoder, GraphedSelfSpeculativeDecoder),
    (EagleConfig(), DraftLayerSpeculativeDecoder, GraphedDraftLayerSpeculativeDecoder),
    (Eagle3Config(), DraftLayerSpeculativeDecoder, GraphedDraftLayerSpeculativeDecoder),
])
@pytest.mark.parametrize("use_graphs", [False, True])
def test_the_built_decoder_reproduces_the_target(target: nn.Module, prompt: torch.Tensor, config, eager, graphed,
                                                 use_graphs: bool) -> None:
    decoder = speculative_decoder(target, replica(config), SSDConfig(draft_length=3), fanout=2, graphed=use_graphs)
    assert type(decoder) is (graphed if use_graphs else eager)
    assert decoder.drafter.target is decoder.target  # linked
    assert torch.equal(decoder.generate(prompt, NEW).tokens, reference_greedy(target, prompt, NEW))


class MyJump(JumpTechnique):
    name = "my_jump"


def holder(technique: object) -> nn.Module:
    """Stands in for a replica: the lookup reads only ``drafting_technique``."""
    return SimpleNamespace(drafting_technique=technique)  # type: ignore[return-value]  # duck-typed replica


@pytest.fixture
def my_jump() -> Iterator[object]:
    yield MyJump.__new__(MyJump)  # the lookup reads only the type
    if "my_jump" in DRAFTER_WORKERS:
        DRAFTER_WORKERS.unregister("my_jump")


def test_a_subclass_inherits_the_workers_of_its_base(my_jump: object) -> None:
    assert drafter_workers(holder(my_jump)) == ("jump", DRAFTER_WORKERS.get("jump"))


def test_a_subclass_registers_its_own_workers(my_jump: object) -> None:
    mine = DrafterWorkers(worker=JumpWorker, decoder=SelfSpeculativeDecoder)
    DRAFTER_WORKERS.register("my_jump", mine)
    assert drafter_workers(holder(my_jump)) == ("my_jump", mine)
    with pytest.raises(ConfigError, match="no CUDA-graphed SSD workers"):
        speculative_decoder(nn.Linear(1, 1), holder(my_jump), graphed=True)


def test_replicas_without_registered_workers_are_rejected() -> None:
    with pytest.raises(ConfigError, match="drafting technique attached"):
        drafter_workers(holder(None))
    with pytest.raises(ConfigError, match="no SSD drafter workers"):
        drafter_workers(holder(object()))


def test_graphed_workers_come_in_pairs() -> None:
    with pytest.raises(ConfigError, match="both graphed_worker and graphed_decoder"):
        DrafterWorkers(worker=JumpWorker, decoder=SelfSpeculativeDecoder, graphed_worker=JumpWorker)
