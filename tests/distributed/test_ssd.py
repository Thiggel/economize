"""Speculative speculative decoding on the CPU: the output is the target's, whatever the drafter and the mode.

The 2-device version of these checks is in ``test_ssd_gpu.py``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from economize.distributed import AsyncSpeculativeDecoder, ModelWorker, SpeculationWorker, SSDConfig
from economize.errors import ConfigError
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, perturb, reference_greedy

NEW = 24


@pytest.fixture(scope="module")
def target() -> nn.Module:
    return make_tiny_model(seed=0)


@pytest.fixture(scope="module")
def close_drafter() -> nn.Module:
    """The target with 2 % noise on its weights: accepts some drafts and rejects others."""
    model = make_tiny_model(seed=0)
    perturb(model, 0.004, seed=3)
    return model


@pytest.fixture(scope="module")
def prompt() -> torch.Tensor:
    return torch.randint(0, VOCAB, (PROMPT_LEN,), generator=torch.Generator().manual_seed(1))


@pytest.fixture(scope="module")
def reference(target: nn.Module, prompt: torch.Tensor) -> torch.Tensor:
    return reference_greedy(target, prompt, NEW)


@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize(("k", "fanout"), [(1, 1), (3, 2)])
def test_greedy_output_is_the_targets(target: nn.Module, close_drafter: nn.Module, prompt: torch.Tensor,
                                      reference: torch.Tensor, mode: str, k: int, fanout: int) -> None:
    decoder = AsyncSpeculativeDecoder(target, close_drafter, SSDConfig(draft_length=k, fanout=fanout, mode=mode))
    result = decoder.generate(prompt, NEW)
    assert torch.equal(result.tokens, reference)
    assert all(0 <= accepted <= k for accepted in result.accepted)
    if mode == "sync":
        assert result.hits == 0


def test_async_and_sync_commit_the_same_rounds(target: nn.Module, close_drafter: nn.Module,
                                               prompt: torch.Tensor) -> None:
    """Greedy drafts do not depend on the mode, so a hit only saves the drafter's latency."""
    runs = [AsyncSpeculativeDecoder(target, close_drafter, SSDConfig(draft_length=3, mode=mode)).generate(prompt, NEW)
            for mode in ("async", "sync")]
    assert runs[0].accepted == runs[1].accepted
    assert 0 < runs[0].hits < runs[0].rounds  # the noisy drafter's guesses hit some outcomes and miss others


def test_identical_drafter_hits_every_round(target: nn.Module, prompt: torch.Tensor,
                                            reference: torch.Tensor) -> None:
    k = 3
    decoder = AsyncSpeculativeDecoder(target, make_tiny_model(seed=0), SSDConfig(draft_length=k, fanout=2))
    result = decoder.generate(prompt, NEW)
    assert torch.equal(result.tokens, reference)
    assert result.accepted == [k] * result.rounds and result.acceptance_length == k + 1
    assert result.hit_rate == 1.0
    assert result.rounds == -(-(NEW - 1) // (k + 1))


def test_identical_drafter_in_sync_mode_keeps_every_draft(target: nn.Module, prompt: torch.Tensor) -> None:
    """After a round that kept all drafts, the drafter catches up on ``d_k`` before drafting on."""
    decoder = AsyncSpeculativeDecoder(target, make_tiny_model(seed=0), SSDConfig(draft_length=2, mode="sync"))
    result = decoder.generate(prompt, NEW)
    assert result.accepted == [2] * result.rounds


def test_unrelated_drafter_still_exact(target: nn.Module, prompt: torch.Tensor, reference: torch.Tensor) -> None:
    drafter = make_tiny_model(seed=5, num_layers=2)
    result = AsyncSpeculativeDecoder(target, drafter, SSDConfig(draft_length=2, fanout=2)).generate(prompt, NEW)
    assert torch.equal(result.tokens, reference)


def test_near_greedy_sampling_is_greedy(target: nn.Module, close_drafter: nn.Module, prompt: torch.Tensor,
                                        reference: torch.Tensor) -> None:
    """With a nucleus of one token, speculative sampling reproduces greedy decoding: the sampled drafts, their
    distributions and the adopted branches all reach the verifier correctly."""
    config = SSDConfig(draft_length=3, fanout=2, temperature=1.0, top_p=1e-6)
    result = AsyncSpeculativeDecoder(target, close_drafter, config).generate(prompt, NEW)
    assert torch.equal(result.tokens, reference)
    assert result.hits > 0


def test_sampling_is_reproducible(target: nn.Module, close_drafter: nn.Module, prompt: torch.Tensor) -> None:
    config = SSDConfig(draft_length=3, fanout=2, temperature=1.0, top_p=0.9, seed=7)
    first, second = (AsyncSpeculativeDecoder(target, close_drafter, config).generate(prompt, NEW) for _ in range(2))
    assert torch.equal(first.tokens, second.tokens) and first.tokens.numel() == NEW
    assert int(first.tokens.max()) < VOCAB


def test_stop_token_ends_generation(target: nn.Module, close_drafter: nn.Module, prompt: torch.Tensor,
                                    reference: torch.Tensor) -> None:
    stop = int(reference[6])
    first = reference.tolist().index(stop)
    result = AsyncSpeculativeDecoder(target, close_drafter, SSDConfig(draft_length=3)).generate(
        prompt, NEW, stop_tokens=[stop])
    assert result.tokens.tolist() == reference[: first + 1].tolist()


def test_cost_accounting(target: nn.Module, close_drafter: nn.Module, prompt: torch.Tensor) -> None:
    k = 3
    result = AsyncSpeculativeDecoder(target, close_drafter, SSDConfig(draft_length=k, fanout=2)).generate(prompt, NEW)
    exact = [event for event in result.cost.events if event.kind == "exact"]
    assert len(exact) == result.rounds + 1  # the prompt pass, then one verification per round
    assert exact[0].tokens == PROMPT_LEN and all(event.tokens == k + 1 for event in exact[1:])
    summary = result.summary()
    assert summary["generated"] == NEW and summary["rounds"] == result.rounds
    assert summary["exact_calls"] == result.rounds + 1 and summary["draft_calls"] == len(result.draft_cost.events)
    # Every round speculates k + 1 tree passes; a miss adds k chain passes.
    misses = result.rounds - result.hits
    calls = len(result.draft_cost.events) - (1 + k + result.rounds * (k + 1))
    assert (misses - 1) * k <= calls <= misses * k  # no chain is drafted after the last round


def test_max_new_tokens_one(target: nn.Module, close_drafter: nn.Module, prompt: torch.Tensor,
                            reference: torch.Tensor) -> None:
    result = AsyncSpeculativeDecoder(target, close_drafter).generate(prompt[None], 1)
    assert result.tokens.tolist() == reference[:1].tolist() and result.rounds == 0


def test_custom_worker_is_used(target: nn.Module, close_drafter: nn.Module) -> None:
    worker = ModelWorker(close_drafter, fanout=3)
    decoder = AsyncSpeculativeDecoder(target, worker, SSDConfig(fanout=1))
    assert decoder.drafter is worker and isinstance(worker, SpeculationWorker)


@pytest.mark.parametrize("bad", [dict(draft_length=0), dict(fanout=0), dict(mode="tree"), dict(temperature=-1.0),
                                 dict(top_p=0.0), dict(top_p=1.5)])
def test_config_validation(bad: dict) -> None:
    with pytest.raises(ConfigError):
        SSDConfig(**bad)


def test_empty_prompt_rejected(target: nn.Module, close_drafter: nn.Module) -> None:
    with pytest.raises(ConfigError):
        AsyncSpeculativeDecoder(target, close_drafter).generate(torch.empty(0, dtype=torch.long), 4)


def test_worker_rejects_bad_fanout(close_drafter: nn.Module) -> None:
    with pytest.raises(ConfigError):
        ModelWorker(close_drafter, fanout=0)
