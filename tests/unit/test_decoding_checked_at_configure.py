"""Decoding options the decoder would reject raise :class:`~economize.ConfigError` when they are set, before any
decoding: a tree budget above what ``cycle_length`` allows, and a tree policy over a drafter without tree steps."""

from __future__ import annotations

from pathlib import Path

import pytest

import economize
from economize import ConfigError, DecodeConfig
from helpers import make_tiny_model
from zoo import specimen


def _model(name: str, assets: Path, decoding: DecodeConfig | None = None) -> economize.EfficientModel:
    return economize.apply(make_tiny_model("sdpa"), specimen(name, assets), decoding=decoding)


def test_tree_budget_above_a_short_cycle_raises_at_apply(tmp_path: Path) -> None:
    # depth defaults to cycle_length - 1 = 1, so topk 10 offers 10 candidates for a budget of 60
    with pytest.raises(ConfigError, match="total_tokens=60 exceeds the 10 candidates"):
        _model("jump", tmp_path, DecodeConfig(cycle_length=2, draft_policy="dynamic_tree",
                                              draft_policy_options={"total_tokens": 60}))


def test_default_tree_budget_shrinks_to_a_short_cycle(tmp_path: Path) -> None:
    model = _model("jump", tmp_path, DecodeConfig(cycle_length=2, draft_policy="dynamic_tree"))
    assert model.decoding.draft_policy == "dynamic_tree"


def test_setting_an_invalid_tree_budget_raises_and_keeps_the_old_options(tmp_path: Path) -> None:
    model = _model("jump", tmp_path)
    before = model.decoding
    with pytest.raises(ConfigError, match="exceeds the 10 candidates"):
        model.decoding = DecodeConfig(cycle_length=2, draft_policy="dynamic_tree",
                                      draft_policy_options={"total_tokens": 60})
    assert model.decoding is before


@pytest.mark.parametrize("name", ["medusa", "kangaroo"])
def test_tree_policy_over_a_drafter_without_tree_steps_raises_at_apply(name: str, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot draft a tree for draft_policy 'dynamic_tree'"):
        _model(name, tmp_path, DecodeConfig(cycle_length=4, draft_policy="dynamic_tree",
                                            draft_policy_options={"total_tokens": 8}))


def test_a_valid_tree_configuration_is_accepted(tmp_path: Path) -> None:
    decoding = DecodeConfig(cycle_length=4, draft_policy="dynamic_tree", draft_policy_options={"total_tokens": 8})
    assert _model("jump", tmp_path, decoding).decoding == decoding
