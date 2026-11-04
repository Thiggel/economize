"""Speculative speculative decoding with CUDA graphs (economize.distributed.graphed), in float32: the graphed workers
commit the rounds of the eager ones and reproduce the target. On the CPU the steps run eagerly on the static caches;
the ``gpu`` tests replay real graphs (``pytest -m gpu tests/distributed/test_ssd_graphed.py``)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import economize
from economize.distributed import AsyncSpeculativeDecoder, SSDConfig
from economize.distributed.graphed import GraphedJumpWorker, GraphedSelfSpeculativeDecoder, GraphedTargetWorker
from economize.distributed.self_drafting import JumpWorker, SelfSpeculativeDecoder
from economize.errors import ConfigError
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, perturb, reference_greedy

NEW = 24


def jump_replica(adaptation: str = "full", sources: tuple[int, ...] = (0, 3), device: str = "cpu") -> nn.Module:
    model = economize.apply(make_tiny_model(seed=0), economize.JumpConfig(
        target_layer=2, source_layer=4, extra_source_layers=sources, adaptation=adaptation))
    perturb(model.techniques["jump"], 0.05, seed=5)
    return model.to(device)


def prompt_ids(seed: int = 1) -> torch.Tensor:
    return torch.randint(0, VOCAB, (PROMPT_LEN,), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize(("k", "fanout"), [(1, 1), (3, 2)])
@pytest.mark.parametrize(("adaptation", "sources"), [("frozen", ()), ("full", (0, 3))])
def test_graphed_rounds_are_the_eager_ones(mode: str, k: int, fanout: int, adaptation: str,
                                           sources: tuple[int, ...]) -> None:
    target, prompt = make_tiny_model(seed=0), prompt_ids()
    config = SSDConfig(draft_length=k, mode=mode)
    eager = SelfSpeculativeDecoder(target, JumpWorker(jump_replica(adaptation, sources), fanout), config)
    graphed = GraphedSelfSpeculativeDecoder(target, GraphedJumpWorker(jump_replica(adaptation, sources), fanout),
                                            config)
    expected = eager.generate(prompt, NEW)
    result = graphed.generate(prompt, NEW)
    assert torch.equal(result.tokens, reference_greedy(target, prompt, NEW))
    assert result.accepted == expected.accepted and result.hits == expected.hits


def test_caches_are_reused_across_generations() -> None:
    target = make_tiny_model(seed=0)
    worker = GraphedJumpWorker(jump_replica())
    decoder = GraphedSelfSpeculativeDecoder(target, worker, SSDConfig(draft_length=3))
    decoder.generate(prompt_ids(1), NEW)
    caches = worker.cache, decoder.target.cache
    result = decoder.generate(prompt_ids(2), NEW // 2)
    assert (worker.cache, decoder.target.cache) == caches
    assert torch.equal(result.tokens, reference_greedy(target, prompt_ids(2), NEW // 2))


def test_unlinked_graphed_worker_with_a_graphed_plain_target() -> None:
    target, prompt = make_tiny_model(seed=0), prompt_ids()
    decoder = AsyncSpeculativeDecoder(target, GraphedJumpWorker(jump_replica()), SSDConfig(draft_length=3))
    decoder.target = GraphedTargetWorker(target)
    assert torch.equal(decoder.generate(prompt, NEW).tokens, reference_greedy(target, prompt, NEW))


def test_rejects_an_eager_worker() -> None:
    with pytest.raises(ConfigError, match="GraphedJumpWorker"):
        GraphedSelfSpeculativeDecoder(make_tiny_model(seed=0), JumpWorker(jump_replica()))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("mode", ["async", "sync"])
def test_replayed_graphs_reproduce_the_target(mode: str) -> None:
    drafter_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    target = make_tiny_model(seed=0).to("cuda:0")
    eager = SelfSpeculativeDecoder(target, JumpWorker(jump_replica(device=drafter_device), 2),
                                   SSDConfig(draft_length=3, mode=mode))
    worker = GraphedJumpWorker(jump_replica(device=drafter_device), 2)
    graphed = GraphedSelfSpeculativeDecoder(target, worker, SSDConfig(draft_length=3, mode=mode))
    for seed in (1, 2, 3, 4):  # graphs are recorded on the third call of a shape, then replayed
        prompt = prompt_ids(seed)
        expected = eager.generate(prompt, 40)
        result = graphed.generate(prompt, 40)
        assert torch.equal(result.tokens, reference_greedy(target, prompt.to("cuda:0"), 40).cpu())
        assert result.accepted == expected.accepted
    # a synchronous round drafts one chain; asynchronous speculation adds the tree passes
    assert worker.graphs.num_graphs >= (3 if mode == "async" else 1) and graphed.target.graphs.num_graphs >= 1
