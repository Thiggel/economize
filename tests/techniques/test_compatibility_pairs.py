"""Every pair of techniques that :func:`economize.compatible` accepts builds on the tiny model and generates.

A pair of lossless techniques (:attr:`~economize.TechniqueInfo.lossless`) must reproduce plain greedy decoding in
fp32; any other pair must produce ``max_new`` valid tokens. The specimens come from :mod:`zoo`, so a newly
registered technique joins this test once it has a specimen there.
"""

from __future__ import annotations

import itertools
import re
import warnings
from pathlib import Path

import pytest
import torch

import economize
from economize.attention import install
from economize.core.metadata import unmet_requirement
from helpers import VOCAB, make_tiny_model, reference_greedy
from zoo import ATTENTION, ELSEWHERE, names, specimen

MAX_NEW = 10


def _runnable(*techniques: str) -> bool:
    """Whether a recipe of exactly these techniques is complete (a technique that needs a drafter has one)."""
    infos = [economize.TECHNIQUES.get(name).info for name in techniques]
    return all(
        unmet_requirement(name, info, infos[:i] + infos[i + 1 :]) is None
        for i, (name, info) in enumerate(zip(techniques, infos))
    )


def _pairs() -> list[tuple[str, ...]]:
    zoo = [name for name in names() if name not in ELSEWHERE]
    combos = [(name,) for name in zoo] + [
        pair for pair in itertools.combinations(zoo, 2) if economize.compatible(*pair)
    ]
    return [combo for combo in combos if _runnable(*combo)]


def test_every_registered_technique_is_covered() -> None:
    assert set(economize.TECHNIQUES.names()) == set(names()) | set(ELSEWHERE)


@pytest.fixture(scope="module")
def assets(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("zoo")


@pytest.mark.parametrize("combo", _pairs(), ids="+".join)
def test_compatible_techniques_generate_together(combo: tuple[str, ...], assets: Path) -> None:
    attention = {ATTENTION.get(name, "sdpa") for name in combo}
    attn = "economize" if "economize" in attention else "sdpa"
    if attn == "economize":
        install()
    base = make_tiny_model(attn)
    prompt = torch.randint(1, VOCAB, (12,), generator=torch.Generator().manual_seed(3))
    expected = reference_greedy(base, prompt, MAX_NEW)
    calibration = torch.randint(1, VOCAB, (4, 24), generator=torch.Generator().manual_seed(5))
    configs = [specimen(name, assets) for name in combo]
    reason = economize.incompatibility(*configs) if len(configs) == 2 else None
    if reason is not None:  # a rule of the configurations, which the names alone do not decide
        with pytest.raises(economize.IncompatibleTechniquesError, match=re.escape(reason)):
            economize.apply(base, *configs, calibration=calibration)
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", economize.SimulatedPathWarning)
        model = economize.apply(base, *configs, calibration=calibration)
    tokens = model.decode(prompt, MAX_NEW).tokens

    assert tokens.shape == (MAX_NEW,)
    assert bool(((tokens >= 0) & (tokens < VOCAB)).all())
    if all(economize.TECHNIQUES.get(name).info.lossless for name in combo):
        assert torch.equal(tokens, expected), f"{'+'.join(combo)} is declared lossless but changes greedy output"
