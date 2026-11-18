"""Speculative sampling with a registered drafter samples exactly from the model (fp32, tiny model, CPU).

:mod:`test_sampling_exact` checks the verifiers with a synthetic drafter. Here the drafts come from the
``draft_model`` technique, built by :func:`economize.apply` around a perturbed copy of the model, so that most
drafts are accepted and the accept path of the real drafter (its distributions, its cache repair after a rejection,
a draft cut short by the confidence policy) is what the statistics see. Tree policies choose drafts by rank and are
refused with the rejection verifier, so only chains are sampled. The checks and their thresholds are those of
:mod:`test_sampling_exact`.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from test_sampling_exact import ALPHA, ROWS, G, joint_keys, ks_p, pit, tv_bound

import economize
from economize import DecodeConfig, DecodeResult, TopPSampler
from economize.techniques.draft_model import DraftModelConfig
from helpers import VOCAB, make_tiny_model, perturb

KEYS = [(1.0, 1.0), (0.8, 0.9)]
DECODINGS = {
    "chain": DecodeConfig(cycle_length=3, verifier="rejection"),
    "confidence stop": DecodeConfig(cycle_length=4, verifier="rejection", min_confidence=0.6),
}


def _peaked() -> torch.nn.Module:
    model = make_tiny_model(seed=2)
    with torch.no_grad():
        model.model.norm.weight.mul_(3.0)  # peaked distributions, so a few thousand samples resolve them
    return model


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory) -> tuple[economize.EfficientModel, torch.nn.Module, torch.Tensor]:
    draft = copy.deepcopy(_peaked())
    perturb(draft, std=0.01, seed=1)
    path: Path = tmp_path_factory.mktemp("perturbed_draft")
    draft.save_pretrained(path)
    model = economize.apply(_peaked(), DraftModelConfig(model=str(path)))
    prompt = torch.randint(1, VOCAB, (9,), generator=torch.Generator().manual_seed(6))
    return model, _peaked(), prompt


def _samples(results: list[DecodeResult]) -> torch.Tensor:
    assert all(r.tokens.numel() == G for r in results)
    return torch.stack([r.tokens for r in results])


@pytest.fixture(scope="module")
def reference(setup: tuple[economize.EfficientModel, torch.nn.Module, torch.Tensor]) -> dict[tuple[float, float],
                                                                                             torch.Tensor]:
    """Plain autoregressive samples (cycle length 1, the drafter unused) for every sampler of the tests."""
    model, _, prompt = setup
    plain = DecodeConfig(cycle_length=1)
    return {key: _samples(model.decode_batch([prompt] * ROWS, G, decoding=plain, selector=TopPSampler(*key, seed=123)))
            for key in KEYS}


@pytest.mark.parametrize("decoding", list(DECODINGS))
@pytest.mark.parametrize("key", KEYS, ids=["T1", "T0.8-p0.9"])
def test_the_draft_model_samples_exactly(setup: tuple[economize.EfficientModel, torch.nn.Module, torch.Tensor],
                                         reference: dict[tuple[float, float], torch.Tensor], decoding: str,
                                         key: tuple[float, float]) -> None:
    model, plain, prompt = setup
    results = model.decode_batch([prompt] * ROWS, G, decoding=DECODINGS[decoding], selector=TopPSampler(*key, seed=7))
    drafted = sum(d for r in results for d, _ in r.acceptance)
    accepted = sum(a for r in results for _, a in r.acceptance)
    assert accepted / len([c for r in results for c in r.acceptance]) > 0.5, (accepted, drafted)
    samples = _samples(results)
    p_value = ks_p(pit(plain, TopPSampler(*key), prompt, samples))
    tvs = [tv_bound(joint_keys(samples, k), joint_keys(reference[key], k)) for k in (2, 3)]
    assert p_value > ALPHA, (p_value, tvs)
    for tv, bound in tvs:
        assert tv <= bound, (p_value, tvs)
