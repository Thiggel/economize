"""Dynamic-depth decode steps through the step engine (:class:`~economize.decoding.DepthSteps`).

With ``cuda_graphs=True`` and ordinary decoding, the one-token decode steps of a depth policy run through its step
interface: the host plans every layer, skipped layers launch nothing, and on a CUDA device the segments between
probes are replayed from graphs. The steps must give the tokens and layer counts of the eager path (``decide`` on
every layer). On the CPU the engine runs the same host loop eagerly over the decoder's static cache.
"""

from __future__ import annotations

import pytest
import torch

import economize
from economize import DecodeConfig
from economize.decoding import CycleDecoder
from economize.techniques.dynamic_depth import (
    CalmConfig,
    FlexiDepthConfig,
    GateSkipConfig,
    LoRADropConfig,
    MoDConfig,
    SkipDecodeConfig,
)
from helpers import VOCAB, make_tiny_model, perturb

CONFIGS = [
    (GateSkipConfig(gate_bias=0.0, gate_init_std=0.5), False),
    (MoDConfig(router_bias=0.0, router_init_std=0.3), False),
    (FlexiDepthConfig(router_bias=0.0), True),
    (SkipDecodeConfig(max_layers=6, min_layers=3, warmup=12, decay=10), False),
    (SkipDecodeConfig(min_layers=2, decay=20, skip="top"), False),
    (CalmConfig(confidence="state", threshold=0.5), False),
    (CalmConfig(confidence="softmax", threshold=0.3), False),
    (CalmConfig(confidence="classifier", threshold=0.4, decay_temperature=4.0), True),
    (LoRADropConfig(period=2), True),
    (GateSkipConfig(gate_bias=0.0, gate_init_std=0.5, decision_group=4), False),
    (GateSkipConfig(gate_bias=0.0, gate_init_std=0.5, decision_group=0), False),
    (MoDConfig(router_bias=0.0, router_init_std=0.3, decision_group=2), False),
    (FlexiDepthConfig(router_bias=0.0, decision_group=0), True),
    (CalmConfig(confidence="softmax", threshold=0.3, min_layer=2, exit_stride=2), False),
]
IDS = ["gateskip", "mod", "flexidepth", "skipdecode-bottom", "skipdecode-top", "calm-state", "calm-softmax",
       "calm-classifier", "loradrop", "gateskip-group4", "gateskip-group-all", "mod-group2", "flexidepth-group-all",
       "calm-stride2"]
MAX_NEW = 24
EAGER = DecodeConfig(cycle_length=1)
STEPS = DecodeConfig(cycle_length=1, cuda_graphs=True)


def _model(config: object, noise: bool, device: str) -> economize.EfficientModel:
    model = economize.apply(make_tiny_model().to(device), config)
    if noise:
        perturb(model.techniques[config.technique], 0.3)
    return model


def _prompts(device: str) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    return [torch.randint(1, VOCAB, (n,), generator=generator).to(device) for n in (12, 7, 20)]


def _run(model: economize.EfficientModel, prompt: torch.Tensor, decoding: DecodeConfig) -> tuple[list[int], dict]:
    result = model.decode(prompt, MAX_NEW, decoding=decoding)
    technique = next(iter(model.techniques.values()))
    return result.tokens.tolist(), technique.depth_stats()


@pytest.fixture
def cpu_graphs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    monkeypatch.setattr(CycleDecoder, "_graphable", lambda self: True)


def _compare(model: economize.EfficientModel, prompts: list[torch.Tensor]) -> CycleDecoder:
    """Decode every prompt eagerly, then twice through the step engine, and compare tokens and layer counts."""
    expected = [_run(model, p, EAGER) for p in prompts]
    assert sum(stats["skipped_layer_tokens"] for _, stats in expected) > 0  # the setting skips layers
    assert model.decoder(EAGER).depth_steps is None
    for _ in range(2):  # the second round replays the recorded segments and reuses the policy
        for prompt, want in zip(prompts, expected, strict=True):
            tokens, stats = _run(model, prompt, STEPS)
            assert tokens == want[0]
            assert stats == want[1]
    decoder = model.decoder(STEPS)
    assert decoder.depth_steps is not None
    return decoder


@pytest.mark.parametrize("config, noise", CONFIGS, ids=IDS)
@torch.no_grad()
def test_steps_decode_like_the_eager_policy(cpu_graphs: None, config: object, noise: bool) -> None:
    model = _model(config, noise, "cpu")
    decoder = _compare(model, _prompts("cpu"))
    assert not decoder.depth_steps.graphed


@pytest.mark.parametrize("group, probes", [(1, 6), (4, 2), (0, 1)])
@torch.no_grad()
def test_decision_groups_read_one_probe_per_group(cpu_graphs: None, monkeypatch: pytest.MonkeyPatch, group: int,
                                                  probes: int) -> None:
    model = _model(GateSkipConfig(gate_bias=0.0, gate_init_std=0.5, decision_group=group), False, "cpu")
    policy = model.techniques["gateskip"].layer_policy()
    assert [len(members) for members in policy.groups().values()] == {1: [1] * 6, 4: [4, 2], 0: [6]}[group]
    calls: list[int] = []
    after_probe = type(policy).after_probe
    monkeypatch.setattr(type(policy), "after_probe",
                        lambda self, index, values: calls.append(index) or after_probe(self, index, values))
    model.decode(_prompts("cpu")[0], 4, decoding=STEPS)
    assert len(calls) == 3 * probes  # the first token comes from the prefill, the other three from steps


@torch.no_grad()
def test_batches_fall_back_to_the_eager_policy(cpu_graphs: None) -> None:
    model = _model(GateSkipConfig(gate_bias=0.0, gate_init_std=0.5), False, "cpu")
    prompts = _prompts("cpu")
    expected = [r.tokens.tolist() for r in model.decode_batch(prompts, [MAX_NEW] * 3, decoding=EAGER)]
    got = [r.tokens.tolist() for r in model.decode_batch(prompts, [MAX_NEW] * 3, decoding=STEPS)]
    assert got == expected


def test_steps_need_ordinary_decoding_and_a_step_policy() -> None:
    model = _model(GateSkipConfig(), False, "cpu")
    assert model.decoder(STEPS).depth_steps is not None  # eager steps on the CPU
    assert model.decoder(DecodeConfig(cycle_length=1)).depth_steps is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("config, noise", CONFIGS, ids=IDS)
@torch.no_grad()
def test_gpu_graphed_steps_decode_like_the_eager_policy(config: object, noise: bool) -> None:
    model = _model(config, noise, "cuda")
    decoder = _compare(model, _prompts("cuda"))
    steps = decoder.depth_steps
    assert steps.graphed and steps.num_graphs > 0
    recorded = steps.num_graphs
    _run(model, _prompts("cuda")[0], STEPS)
    assert steps.num_graphs == recorded  # a repeated prompt records nothing new
