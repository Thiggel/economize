"""Recipes of three and more techniques against the decoding of their own compressed model.

:mod:`test_compatibility_pairs` covers every compatible pair. Here whole recipes stack weight compression (pruning,
weight quantization), a drafter and, where the drafter has modules of its own, a quantization of the drafter. The
weight techniques are lossy, the drafting is lossless, so greedy decoding of the full recipe must equal greedy
decoding, by full recomputation without a cache, of the base model as the weight techniques alone leave it. The
model drafter is a full copy of the base model, so the drafts agree with the compressed model often and the accept
path runs.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

import economize
from economize import DecodeConfig, TechniqueConfig
from economize.techniques.draft_model import DraftModelConfig
from economize.techniques.early_exit import EarlyExitConfig
from economize.techniques.prune import PruneConfig
from economize.techniques.quant import DraftQuantConfig, QuantConfig
from helpers import VOCAB, make_tiny_model, reference_greedy

NEW = 24


def _weights() -> list[TechniqueConfig]:
    return [PruneConfig(method="wanda", sparsity="2:4", semi_structured=False),
            QuantConfig(backend="fake")]


def _apply(*configs: TechniqueConfig) -> economize.EfficientModel:
    calibration = torch.randint(1, VOCAB, (4, 24), generator=torch.Generator().manual_seed(5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", economize.SimulatedPathWarning)
        return economize.apply(make_tiny_model(), *configs, decoding=DecodeConfig(cycle_length=4),
                               calibration=calibration)


@pytest.fixture(scope="module")
def draft_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    path: Path = tmp_path_factory.mktemp("full_copy")
    make_tiny_model().save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def compressed() -> economize.EfficientModel:
    """The weight techniques alone: the model every recipe below must decode exactly."""
    return _apply(*_weights())


def _prompts() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in (12, 7, 20)]


@pytest.mark.parametrize("drafting", ["early_exit+draft_quant", "early_exit", "draft_model"])
def test_a_recipe_decodes_its_compressed_model_exactly(compressed: economize.EfficientModel, draft_dir: str,
                                                       drafting: str) -> None:
    drafters: dict[str, list[TechniqueConfig]] = {
        "early_exit+draft_quant": [EarlyExitConfig(exit_layer=3), DraftQuantConfig(group_size=None, backend="fake")],
        "early_exit": [EarlyExitConfig(exit_layer=3)],
        "draft_model": [DraftModelConfig(model=draft_dir)],
    }
    model = _apply(*_weights(), *drafters[drafting])
    assert list(model.techniques)[:2] == ["prune", "quant"] and len(model.techniques) >= 3
    prompts = _prompts()
    results = model.decode_batch(prompts, NEW)
    for prompt, result in zip(prompts, results):
        assert torch.equal(result.tokens, reference_greedy(compressed.base_model, prompt, NEW))
    cycles = [a for r in results for _, a in r.acceptance]
    assert sum(cycles) > 0  # some drafts were accepted
