"""Speculative speculative decoding with a draft-layer drafter (EAGLE-1/2 and EAGLE-3,
economize.distributed.draft_layer_worker), in float32: the output is the target's in both modes, synchronous rounds
are those of single-device decoding with the same drafter, and the graphed worker commits the rounds of the eager
one. The ``gpu`` test replays real graphs (``pytest -m gpu tests/distributed/test_ssd_draft_layer.py``)."""

from __future__ import annotations

from functools import cache

import pytest
import torch
from torch import nn

import economize
from economize import DecodeConfig
from economize.distributed import (
    DRAFT_LAYER_WORKERS,
    DRAFTER_WORKERS,
    DraftLayerSpeculativeDecoder,
    DraftLayerWorker,
    Eagle3Worker,
    GraphedDraftLayerSpeculativeDecoder,
    GraphedDraftLayerWorker,
    SSDConfig,
    speculative_decoder,
)
from economize.errors import ConfigError
from economize.techniques._draft_layer import DraftLayerConfig
from economize.techniques.eagle import EagleConfig
from economize.techniques.eagle3 import Eagle3Config
from economize.train import fit
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, perturb, reference_greedy

NEW = 24
VARIANTS: dict[str, DraftLayerConfig] = {
    "eagle": EagleConfig(),
    "eagle-zero_source": EagleConfig(zero_source_first=True),
    "eagle3-own": Eagle3Config(),
    "eagle3-shared": Eagle3Config(head="shared"),
    "eagle3-zero_source": Eagle3Config(zero_source_first=True),
}


@cache
def trained_state(variant: str) -> dict[str, torch.Tensor]:
    """An EAGLE-1/2 head fitted to the tiny model's greedy continuations: a perturbed new head (zero output
    projections plus noise) drafts nothing the target accepts, so the round comparisons would be vacuous."""
    base = make_tiny_model(seed=0)
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        prompts = [torch.randint(1, VOCAB, (PROMPT_LEN,), generator=generator) for _ in range(4)]
        rows = torch.stack([torch.cat([p, reference_greedy(base, p, 28)]) for p in prompts])
    model = economize.apply(base, VARIANTS[variant])
    with torch.enable_grad():
        fit(model, rows, steps=60, batch_size=4, lr=1e-2, warmup_steps=0, seed=0)
    return {k: v.detach().clone() for k, v in model.techniques["eagle"].state_dict().items()}


def replica(variant: str = "eagle3-own", device: str = "cpu", cycle_length: int | None = None) -> nn.Module:
    decoding = DecodeConfig(cycle_length=cycle_length) if cycle_length else None
    config = VARIANTS[variant]
    model = economize.apply(make_tiny_model(seed=0), config, decoding=decoding)
    technique = model.techniques[config.technique]
    if config.technique == "eagle":
        technique.load_state_dict(trained_state(variant))
    else:
        perturb(technique, 0.05, seed=1)
    return model.to(device)


def prompt_ids(seed: int = 1) -> torch.Tensor:
    return torch.randint(1, VOCAB, (PROMPT_LEN,), generator=torch.Generator().manual_seed(seed))


@pytest.fixture(scope="module")
def target() -> nn.Module:
    return make_tiny_model(seed=0)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize(("k", "fanout"), [(1, 1), (3, 2)])
def test_reproduces_the_target(target: nn.Module, variant: str, mode: str, k: int, fanout: int) -> None:
    prompt = prompt_ids()
    decoder = DraftLayerSpeculativeDecoder(target, DraftLayerWorker(replica(variant), fanout),
                                           SSDConfig(draft_length=k, mode=mode))
    result = decoder.generate(prompt, NEW)
    assert torch.equal(result.tokens, reference_greedy(target, prompt, NEW))
    assert all(0 <= accepted <= k for accepted in result.accepted)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_sync_rounds_are_single_device_decoding(target: nn.Module, variant: str) -> None:
    """Synchronously every chain starts from the target's fused states, as in single-device decoding."""
    prompt, k = prompt_ids(), 3
    decoder = DraftLayerSpeculativeDecoder(target, DraftLayerWorker(replica(variant)),
                                           SSDConfig(draft_length=k, mode="sync"))
    linked = decoder.generate(prompt, NEW)
    single = replica(variant, cycle_length=k + 1).decode(prompt, NEW)
    assert linked.accepted == [accepted for _, accepted in single.acceptance][: len(linked.accepted)]
    assert sum(linked.accepted) > 0


def test_async_hits_and_costs(target: nn.Module) -> None:
    prompt = prompt_ids(2)
    decoder = DraftLayerSpeculativeDecoder(target, DraftLayerWorker(replica(), 2), SSDConfig(draft_length=3))
    result = decoder.generate(prompt, NEW)
    assert torch.equal(result.tokens, reference_greedy(target, prompt, NEW))
    assert result.hits > 0
    kinds = {event.kind for event in result.draft_cost.events}
    assert {"draft", "draft_observe"} <= kinds


def test_rejects_other_models_and_an_unlinked_worker() -> None:
    with pytest.raises(ConfigError, match="draft-layer technique"):
        DraftLayerWorker(make_tiny_model(seed=0))
    with pytest.raises(ConfigError, match="DraftLayerSpeculativeDecoder"):
        DraftLayerWorker(replica()).begin(prompt_ids(), 64)
    with pytest.raises(ConfigError, match="GraphedDraftLayerWorker"):
        GraphedDraftLayerSpeculativeDecoder(
            make_tiny_model(seed=0), DraftLayerWorker(replica()))  # type: ignore[arg-type]  # the refused input


@pytest.mark.parametrize("technique", ["eagle", "eagle3"])
def test_both_techniques_are_registered(technique: str) -> None:
    """``speculative_decoder`` finds the draft-layer workers by the technique's name; the EAGLE-3 names alias them."""
    assert DRAFTER_WORKERS.get(technique) is DRAFT_LAYER_WORKERS and Eagle3Worker is DraftLayerWorker
    variant = "eagle" if technique == "eagle" else "eagle3-own"
    decoder = speculative_decoder(make_tiny_model(seed=0), replica(variant), SSDConfig(draft_length=3))
    assert type(decoder) is DraftLayerSpeculativeDecoder and type(decoder.drafter) is DraftLayerWorker


@pytest.mark.parametrize("variant", ["eagle", "eagle3-own"])
@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize(("k", "fanout"), [(1, 1), (3, 2)])
def test_graphed_rounds_are_the_eager_ones(target: nn.Module, variant: str, mode: str, k: int, fanout: int) -> None:
    config = SSDConfig(draft_length=k, mode=mode)
    eager = DraftLayerSpeculativeDecoder(target, DraftLayerWorker(replica(variant), fanout), config)
    worker = GraphedDraftLayerWorker(replica(variant), fanout)
    graphed = GraphedDraftLayerSpeculativeDecoder(target, worker, config)
    for seed in (1, 2):  # the second generation reuses the caches
        prompt = prompt_ids(seed)
        expected = eager.generate(prompt, NEW)
        result = graphed.generate(prompt, NEW)
        assert torch.equal(result.tokens, reference_greedy(target, prompt, NEW))
        assert result.accepted == expected.accepted and result.hits == expected.hits


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("variant", ["eagle", "eagle3-own"])
@pytest.mark.parametrize("mode", ["async", "sync"])
def test_replayed_graphs_reproduce_the_target(variant: str, mode: str) -> None:
    drafter_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    target = make_tiny_model(seed=0).to("cuda:0")
    eager = DraftLayerSpeculativeDecoder(target, DraftLayerWorker(replica(variant, device=drafter_device), 2),
                                         SSDConfig(draft_length=3, mode=mode))
    worker = GraphedDraftLayerWorker(replica(variant, device=drafter_device), 2)
    graphed = GraphedDraftLayerSpeculativeDecoder(target, worker, SSDConfig(draft_length=3, mode=mode))
    for seed in (1, 2, 3, 4):  # graphs are recorded on the third call of a shape, then replayed
        prompt = prompt_ids(seed)
        expected = eager.generate(prompt, 40)
        result = graphed.generate(prompt, 40)
        assert torch.equal(result.tokens, reference_greedy(target, prompt.to("cuda:0"), 40).cpu())
        assert result.accepted == expected.accepted
    # a synchronous round drafts one chain; asynchronous speculation adds the tree passes
    assert worker.num_graphs >= (2 if mode == "async" else 1) and graphed.target.graphs.num_graphs >= 1
