"""A model split over two devices layer by layer (what ``device_map`` pipelining leaves behind): bypassing and
EAGLE-3 stay lossless and train.

The lower half of the layers and the embedding sit on the CPU, the upper half, the rotary module and the head on the
GPU, so that every per-layer part of a technique (window copies, LoRA factors, readout, the predictor's inputs, the
rotary tables) has to follow its layer. Bypassing reads layer ``u = L - 1`` and drafts through ``2..L-1``, which crosses
the split.
"""

from __future__ import annotations

import pytest
import torch

import economize
from economize import JumpConfig
from economize.decoding import DecodeConfig
from economize.techniques.eagle3 import Eagle3Config
from economize.train import Batch, default_objective, fit
from economize.train.data import Example, collate
from helpers import VOCAB, make_family_model, perturb, reference_greedy

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]

LOW, HIGH = torch.device("cpu"), torch.device("cuda:0")
NEW = 12
# Dense, MoE, attention sinks (gpt-oss), hybrid Gated DeltaNet (Qwen3.5) and hybrid Mamba-2 (Nemotron-H) layers.
FAMILIES = ("llama", "qwen3_moe", "gpt_oss", "gemma4_text_moe", "qwen3_5_moe_text", "nemotron_h")
OWN_ROPE = {"nemotron_h": {"rope_theta": 10000.0}}  # no rotary table of its own for the EAGLE-3 layer


class _Source:
    def __init__(self) -> None:
        self.generator = torch.Generator().manual_seed(5)

    def next_batch(self) -> Batch:
        rows = [Example(torch.randint(3, VOCAB, (24,), generator=self.generator), torch.ones(24, dtype=torch.long))
                for _ in range(2)]
        return collate(rows, 0)

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


def _split(model: torch.nn.Module, low: torch.device = LOW, high: torch.device = HIGH) -> torch.nn.Module:
    layers = next(m for n, m in model.named_modules() if n.endswith(".layers") and isinstance(m, torch.nn.ModuleList))
    model.to(high)
    for layer in list(layers)[: len(layers) // 2]:
        layer.to(low)
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            module.to(low)
    return model


def _drafter(name: str, family: str, num_layers: int):
    if name == "eagle3":
        return Eagle3Config(head="shared", **OWN_ROPE.get(family, {}))
    return JumpConfig(target_layer=1, source_layer=num_layers - 1, exit_layer=num_layers - 1, adaptation=name)


# "gpu_low" puts the inputs on the GPU and the upper layers on the CPU, so that position indices made on the input
# device must follow the states of the upper layers (two GPUs behave the same way, CPU indices would not).
@pytest.mark.parametrize("orientation", ["cpu_low", "gpu_low"])
@pytest.mark.parametrize("drafter", ["full", "lora", "eagle3"])
@pytest.mark.parametrize("family", FAMILIES)
def test_split_model_is_lossless_and_trains(family: str, drafter: str, orientation: str) -> None:
    reference = make_family_model(family)
    generator = torch.Generator().manual_seed(2)
    prompts = [torch.randint(3, VOCAB, (n,), generator=generator) for n in (9, 5)]
    greedy = [reference_greedy(reference, p, NEW) for p in prompts]

    devices = (LOW, HIGH) if orientation == "cpu_low" else (HIGH, LOW)
    model = _split(make_family_model(family), *devices)
    num_layers = sum(1 for n, m in model.named_modules() if n.endswith(".layers") for _ in m)
    efficient = economize.apply(model, _drafter(drafter, family, num_layers))
    name = next(iter(efficient.techniques))
    perturb(efficient.techniques[name], 0.02, seed=1)
    decoding = {"decoding": DecodeConfig(cycle_length=3)} if drafter != "eagle3" else {}
    results = efficient.decode_batch(prompts, NEW, **decoding)
    for result, want in zip(results, greedy):
        assert torch.equal(torch.as_tensor(result.tokens)[-NEW:].cpu(), want.cpu())

    history = fit(efficient, _Source(), default_objective(efficient), steps=2, lr=1e-3, log_every=1).history
    losses = [entry["loss"] for entry in history if "loss" in entry]
    assert len(losses) == 2 and all(torch.isfinite(torch.tensor(losses)))
