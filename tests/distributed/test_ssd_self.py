"""Speculative speculative decoding with a jump drafter on its own device (economize.distributed.self_drafting), on
the CPU in float32: the output is the target's for every drafter, mode and link, and an exact drafter is always
accepted."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import economize
from economize.decoding import DecodeConfig
from economize.distributed import AsyncSpeculativeDecoder, SSDConfig
from economize.distributed.self_drafting import JumpWorker, SelfSpeculativeDecoder
from economize.errors import ConfigError
from helpers import PROMPT_LEN, VOCAB, make_identity_jump, make_tiny_model, perturb, reference_greedy

NEW = 24
#: Noise on a jump drafter that would compute the target exactly: some drafts are accepted, some rejected.
NOISE = 0.02


@pytest.fixture(scope="module")
def target() -> nn.Module:
    return make_tiny_model(seed=0)


@pytest.fixture(scope="module")
def prompt() -> torch.Tensor:
    return torch.randint(0, VOCAB, (PROMPT_LEN,), generator=torch.Generator().manual_seed(1))


@pytest.fixture(scope="module")
def reference(target: nn.Module, prompt: torch.Tensor) -> torch.Tensor:
    return reference_greedy(target, prompt, NEW)


def jump_replica(adaptation: str, sources: tuple[int, ...] = ()) -> nn.Module:
    """A replica of the target with a randomly perturbed jump drafter (m = 2, u = 4): some drafts are accepted."""
    model = economize.apply(make_tiny_model(seed=0), economize.JumpConfig(
        target_layer=2, source_layer=4, extra_source_layers=sources, adaptation=adaptation))
    perturb(model.techniques["jump"], 0.05, seed=5)
    return model


def decoder(target: nn.Module, replica: nn.Module, linked: bool, **config) -> AsyncSpeculativeDecoder:
    worker = JumpWorker(replica, fanout=config.pop("fanout", 2))
    cls = SelfSpeculativeDecoder if linked else AsyncSpeculativeDecoder
    return cls(target, worker, SSDConfig(**config))


@pytest.mark.parametrize("linked", [True, False])
@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize(("k", "fanout"), [(1, 1), (3, 2)])
@pytest.mark.parametrize(("adaptation", "sources"), [("frozen", ()), ("full", (0, 3))])
def test_greedy_output_is_the_targets(target: nn.Module, prompt: torch.Tensor, reference: torch.Tensor,
                                      linked: bool, mode: str, k: int, fanout: int, adaptation: str,
                                      sources: tuple[int, ...]) -> None:
    result = decoder(target, jump_replica(adaptation, sources), linked, draft_length=k, fanout=fanout,
                     mode=mode).generate(prompt, NEW)
    assert torch.equal(result.tokens, reference)
    assert all(0 <= accepted <= k for accepted in result.accepted)


def noisy_exact_replica() -> nn.Module:
    replica = make_identity_jump(make_tiny_model(seed=0), "full")
    perturb(replica.techniques["jump"], NOISE, seed=5)
    return replica


def test_unlinked_async_and_sync_commit_the_same_rounds(target: nn.Module, prompt: torch.Tensor) -> None:
    """Unlinked, a speculated branch holds the drafts the worker would draft after that outcome, so the mode changes
    nothing but the latency."""
    runs = [decoder(target, noisy_exact_replica(), False, draft_length=3, mode=mode).generate(prompt, NEW)
            for mode in ("async", "sync")]
    assert runs[0].accepted == runs[1].accepted
    assert 0 < sum(runs[0].accepted) < 3 * runs[0].rounds  # the noisy drafter is accepted sometimes


def test_linked_sync_drafts_from_exact_states(target: nn.Module, prompt: torch.Tensor) -> None:
    """Linked and synchronous, every chain starts from the target's states, as in single-device jump decoding. A
    speculated branch (async) is drafted before the target has verified its parent, from the drafter's own states,
    so the two modes may commit different rounds (both reproduce the target)."""
    for replica in (jump_replica("full", (0, 3)), noisy_exact_replica()):
        linked = decoder(target, replica, True, draft_length=3, mode="sync").generate(prompt, NEW)
        single = replica.decode(prompt, NEW, decoding=DecodeConfig(cycle_length=4))
        assert linked.accepted == [accepted for _, accepted in single.acceptance][: len(linked.accepted)]


@pytest.mark.parametrize("linked", [True, False])
@pytest.mark.parametrize("adaptation", ["frozen", "full"])
def test_an_exact_drafter_is_always_accepted(target: nn.Module, prompt: torch.Tensor, reference: torch.Tensor,
                                             linked: bool, adaptation: str) -> None:
    """With m = u = 0 and r = L a draft step computes what the target computes: from the copied exact K/V (linked)
    or from the replica's own prompt pass (unlinked), every draft is accepted and every speculation hits."""
    replica = make_identity_jump(make_tiny_model(seed=0), adaptation)
    result = decoder(target, replica, linked, draft_length=3, fanout=1).generate(prompt, NEW)
    assert torch.equal(result.tokens, reference)
    assert all(accepted == 3 for accepted in result.accepted[:-1])
    assert result.hits >= result.rounds - 1


def test_linking_moves_only_the_window(target: nn.Module, prompt: torch.Tensor) -> None:
    """Linked, the worker runs no exact pass and holds K/V of layers m+1..r only."""
    worker = JumpWorker(jump_replica("full"))
    SelfSpeculativeDecoder(target, worker, SSDConfig(draft_length=3)).generate(prompt, NEW)
    assert {event.kind for event in worker.cost.events} == {"draft"}
    assert set(worker.cache.keys) == set(worker.window_keys) == {2, 3, 4, 5}  # layers 3..6


def test_rejects_models_without_a_jump_drafter(target: nn.Module) -> None:
    with pytest.raises(ConfigError, match="jump technique"):
        JumpWorker(target)
    with pytest.raises(ConfigError, match="JumpWorker"):
        SelfSpeculativeDecoder(target, target)  # type: ignore[arg-type]
