"""Two-model speculative decoding with the draft LM on a device of its own: lossless on any placement."""

from __future__ import annotations

import copy

import pytest
import torch

import economize
from economize import ConfigError, DecodeConfig
from economize.core.technique import TECHNIQUES
from economize.decoding import LayerFootprint
from economize.techniques.draft_model import DraftModelConfig
from economize.techniques.placed_draft import (
    PlacedDraftModelConfig,
    PlacedDraftModelDrafter,
    PlacedDraftModelTechnique,
)
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, reference_greedy

DRAFT_LAYERS = 5


def _truncated(base: torch.nn.Module, layers: int) -> torch.nn.Module:
    from transformers import Qwen2ForCausalLM

    config = copy.deepcopy(base.config)
    config.num_hidden_layers = layers
    config.layer_types = list(config.layer_types[:layers])
    draft = Qwen2ForCausalLM(config).float().eval()
    draft.load_state_dict(base.state_dict(), strict=False)
    return draft


@pytest.fixture(scope="module")
def draft_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("placed_draft5")
    _truncated(make_tiny_model(), DRAFT_LAYERS).save_pretrained(path)
    return str(path)


def _prompts(seed: int = 1, lengths: tuple[int, ...] = (PROMPT_LEN, 7, 20)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in lengths]


def test_registration_and_config() -> None:
    assert TECHNIQUES.get("placed_draft_model") is PlacedDraftModelTechnique
    config = PlacedDraftModelConfig(model="org/draft", device="cuda:1")
    assert PlacedDraftModelConfig.from_dict(config.to_dict()) == config
    assert isinstance(config, DraftModelConfig)
    with pytest.raises(ConfigError):
        PlacedDraftModelConfig(model="org/draft", device="gpu-two")
    with pytest.raises(ConfigError):
        PlacedDraftModelConfig(device="cpu")


def test_draft_lm_lives_on_its_device(draft_dir: str) -> None:
    model = economize.apply(make_tiny_model(), PlacedDraftModelConfig(model=draft_dir, device="cpu"))
    technique = model.techniques["placed_draft_model"]
    assert technique.state_dict() == {} and not list(technique.parameters())
    assert next(technique.draft_model.parameters()).device == torch.device("cpu")
    drafter = technique.drafter(None)
    assert isinstance(drafter, PlacedDraftModelDrafter)
    assert drafter.home == torch.device("cpu") and drafter.draft_device == torch.device("cpu")
    assert drafter.footprint == LayerFootprint(1, DRAFT_LAYERS, "draft_model")


@pytest.mark.parametrize("decoding", [DecodeConfig(cycle_length=4), DecodeConfig(cycle_length=6, min_confidence=0.6)])
def test_greedy_equals_plain_greedy(draft_dir: str, decoding: DecodeConfig) -> None:
    base = make_tiny_model()
    model = economize.apply(base, PlacedDraftModelConfig(model=draft_dir, device="cpu"))
    prompts = _prompts()
    for prompt, result in zip(prompts, model.decode_batch(prompts, 24, decoding=decoding)):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 24))


def _devices() -> list[tuple[str, str]]:
    """(model, draft) device pairs this machine has: model on the CPU or GPU 0, draft on GPU 0 or GPU 1."""
    pairs = []
    if torch.cuda.is_available():
        pairs += [("cpu", "cuda:0"), ("cuda:0", "cuda:0")]
    if torch.cuda.device_count() >= 2:
        pairs += [("cuda:0", "cuda:1"), ("cuda:1", "cuda:0")]
    return pairs


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("home,device", _devices())
@torch.no_grad()
def test_placed_draft_is_lossless_eager_and_graphed(draft_dir: str, home: str, device: str) -> None:
    """Eager and with ``cuda_graphs`` (the draft steps captured on the draft device), also after rows leave the
    batch and after a longer generation replaces the draft LM's cache."""
    base = make_tiny_model().to(home)
    model = economize.apply(base, PlacedDraftModelConfig(model=draft_dir, device=device))
    prompts = [p.to(home) for p in _prompts()]
    graphs = [False] + ([True] if home.startswith("cuda") else [])
    for cuda_graphs in graphs:
        decoding = DecodeConfig(cycle_length=4, cuda_graphs=cuda_graphs)
        for limits in ([24, 24, 24], [24, 24, 24], [24, 3, 11], [150, 24, 24]):
            results = model.decode_batch(prompts, limits, decoding=decoding)
            for prompt, result, n in zip(prompts, results, limits):
                assert torch.equal(result.tokens.cpu(), reference_greedy(base, prompt, n).cpu())
        drafter = model.decoder(decoding).active_drafter
        assert isinstance(drafter, PlacedDraftModelDrafter)
        assert drafter.draft_device == torch.device(device)
        assert drafter._step_fn.enabled == cuda_graphs
