"""The jump drafter on ragged batches and odd hidden sizes, and its save/load round trip."""

from __future__ import annotations

import pytest
import torch

import economize
from economize import DecodeConfig, JumpConfig
from economize.techniques.jump.predictor import PREDICTORS
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, perturb, reference_greedy

MAX_NEW = 14


VARIANTS = {
    "frozen": dict(target_layer=2, source_layer=4),
    "readout": dict(target_layer=1, source_layer=3, exit_layer=4),
    "lora": dict(target_layer=2, source_layer=4, adaptation="lora", lora_rank=4),
    "full_readout": dict(target_layer=1, source_layer=3, exit_layer=5, adaptation="full"),
    "embedding_source": dict(target_layer=0, source_layer=2, extra_source_layers=(0,)),
    "updated_sources": dict(target_layer=2, source_layer=5, extra_source_layers=(1, 2, 4), mixer="layer_attention",
                            mixer_heads=4),
    "flow": dict(target_layer=2, source_layer=4, predictor="flow"),
}


def _model(**kwargs) -> economize.EfficientModel:
    model = economize.apply(make_tiny_model(), JumpConfig(**kwargs))
    perturb(model.techniques["jump"], 0.05)
    return model


def _prompts() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(4)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in (PROMPT_LEN, 5, 21)]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("lossless", [True, False])
def test_ragged_batch_decodes_like_single_rows(variant: str, lossless: bool) -> None:
    """Each row of a ragged batch decodes as it does alone (drafts included, with ``lossless=False``); the lossless
    decode is greedy."""
    model = _model(**VARIANTS[variant])
    prompts = _prompts()
    config = DecodeConfig(cycle_length=4, lossless=lossless)
    together = [r.tokens.tolist() for r in model.decode_batch(prompts, MAX_NEW, decoding=config)]
    alone = [model.decode(p, MAX_NEW, decoding=config).tokens.tolist() for p in prompts]
    assert together == alone
    if lossless:
        assert together == [reference_greedy(model.base_model, p, MAX_NEW).tolist() for p in prompts]


@pytest.mark.parametrize("name", ["mlp", "linear", "flow", "flow_linear"])
def test_predictors_accept_an_odd_hidden_size(name: str) -> None:
    """Regression: the flow predictors sized their velocity network's input as ``2 d + time`` while ``z`` has
    ``2 (d // 2)`` channels, so an odd hidden size failed with a shape mismatch."""
    d = 63
    predictor = PREDICTORS.get(name)(d, JumpConfig(target_layer=1, source_layer=1, predictor=name))
    source, embedding = torch.randn(2, 3, d), torch.randn(2, 3, d)
    assert predictor(source, embedding).shape == (2, 3, d)
    if predictor.has_auxiliary_loss:
        assert predictor.auxiliary_loss(source, embedding, torch.randn(2, 3, d)).shape == (2, 3)


@pytest.mark.parametrize("variant", ["lora", "full_readout", "flow", "updated_sources"])
def test_save_load_round_trip_decodes_the_same(tmp_path, variant: str) -> None:
    model = _model(**VARIANTS[variant])
    technique = model.techniques["jump"]
    if variant == "flow":
        technique.predictor.residual_scale.uniform_(0.5, 2.0)  # a trained scale must survive the round trip
    config = DecodeConfig(cycle_length=4, lossless=False)
    prompt = _prompts()[0]
    drafts = model.decode(prompt, MAX_NEW, decoding=config).tokens.tolist()
    economize.save(model, tmp_path / "jump")
    loaded = economize.load(make_tiny_model(), tmp_path / "jump")
    state, restored = technique.state_dict(), loaded.techniques["jump"].state_dict()
    assert state.keys() == restored.keys()
    for key in state:
        torch.testing.assert_close(restored[key], state[key], atol=0, rtol=0, msg=key)
    assert loaded.decode(prompt, MAX_NEW, decoding=config).tokens.tolist() == drafts
