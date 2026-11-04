"""EfficientModel.decoder: one drafter per generation unless the decoder records CUDA graphs."""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

import pytest
import torch

from economize import Capability, DecodeConfig, Technique, TechniqueConfig
from economize.arch import adapter_for
from economize.core.metadata import Family, Saving, TechniqueInfo
from economize.core.model import EfficientModel
from helpers import ScriptedDrafter


class DriftingDrafter(ScriptedDrafter):
    """A scripted drafter that counts its generations and never resets the count, so a reused instance proposes
    other tokens in its second generation (offset by the count)."""

    def begin(self, runner: Any, cache: Any, batch_size: int) -> None:
        super().begin(runner, cache, batch_size)
        self.offset += 1


@dataclasses.dataclass(frozen=True)
class DriftingConfig(TechniqueConfig):
    technique: ClassVar[str] = "drifting"


class Drifting(Technique):
    name = "drifting"
    config_class = DriftingConfig
    capabilities = Capability.DRAFTER
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True)

    def drafter(self, runner: Any) -> DriftingDrafter:
        return DriftingDrafter(torch.arange(64) % 7 + 3)


def _model(factory: Any, cuda_graphs: bool) -> EfficientModel:
    base = factory()
    adapter = adapter_for(base)
    decoding = DecodeConfig(cycle_length=4, lossless=False, cuda_graphs=cuda_graphs)
    return EfficientModel(base, adapter, [Drifting(DriftingConfig(), adapter)], decoding)


@pytest.mark.parametrize("cuda_graphs", [False, True])
def test_consecutive_generations_match_fresh_models(tiny_factory, prompt, cuda_graphs: bool) -> None:
    model = _model(tiny_factory, cuda_graphs)
    first, second = (model.decode(prompt, 16).tokens for _ in range(2))
    fresh = [_model(tiny_factory, cuda_graphs).decode(prompt, 16).tokens for _ in range(2)]
    assert torch.equal(torch.as_tensor(first), torch.as_tensor(fresh[0]))
    assert torch.equal(torch.as_tensor(second), torch.as_tensor(fresh[1]))
    assert torch.equal(torch.as_tensor(first), torch.as_tensor(second))


def test_eager_decoders_are_not_cached(tiny_factory) -> None:
    model = _model(tiny_factory, cuda_graphs=True)  # on the CPU the decoder cannot record graphs
    assert model.decoder().graphs is None
    assert model.decoder() is not model.decoder()
    assert model.decoder().drafter is not model.decoder().drafter
