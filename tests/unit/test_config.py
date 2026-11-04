"""Validation and serialization of the configuration objects."""

from __future__ import annotations

import json

import pytest

import economize
from economize import ConfigError, DecodeConfig, JumpConfig, UnknownNameError


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(target_layer=2, source_layer=4, adaptation="partial"), "adaptation"),
        (dict(target_layer=5, source_layer=4), "target_layer <= source_layer"),
        (dict(target_layer=-1, source_layer=4), "target_layer <= source_layer"),
        (dict(target_layer=2, source_layer=4, exit_layer=3), "source_layer <= exit_layer"),
        (dict(target_layer=4, source_layer=4, exit_layer=4), "at least one layer"),
        (dict(target_layer=2, source_layer=4, extra_source_layers=(4,)), "extra_source_layers"),
        (dict(target_layer=2, source_layer=4, predictor_width=0.0), "predictor_width"),
        (dict(target_layer=2, source_layer=4, predictor_depth=0), "predictor_depth"),
        (dict(target_layer=2, source_layer=4, lora_rank=0), "lora_rank"),
    ],
)
def test_jump_config_rejects_invalid_values(kwargs: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        JumpConfig(**kwargs)


def test_config_errors_are_value_errors() -> None:
    with pytest.raises(ValueError):
        JumpConfig(target_layer=5, source_layer=4)


def test_jump_config_derived_values() -> None:
    config = JumpConfig(target_layer=2, source_layer=4, extra_source_layers=[3, 0, 3], lora_rank=4)
    assert config.extra_source_layers == (0, 3)
    assert config.source_layers == (0, 3, 4)
    assert config.resolved_mixer == "concat"
    assert config.resolved_lora_alpha == 8.0
    assert config.exit_layer_of(6) == 6
    single = JumpConfig(target_layer=2, source_layer=4, exit_layer=5)
    assert single.resolved_mixer is None and single.exit_layer_of(6) == 5
    assert JumpConfig(target_layer=2, source_layer=4, mixer="fused_linear").resolved_mixer == "fused_linear"


def test_jump_config_round_trips_through_json() -> None:
    config = JumpConfig(
        target_layer=2, source_layer=4, exit_layer=5, extra_source_layers=(1,), mixer="layer_attention", mixer_heads=4
    )
    data = json.loads(json.dumps(config.to_dict()))
    assert data["extra_source_layers"] == [1]
    assert JumpConfig.from_dict(data) == config


def test_jump_config_rejects_unknown_fields() -> None:
    with pytest.raises(ConfigError, match="unknown JumpConfig fields \\['m'\\]"):
        JumpConfig.from_dict({"target_layer": 2, "source_layer": 4, "m": 2})


def test_jump_config_from_dict_reports_missing_fields() -> None:
    with pytest.raises(ConfigError, match="JumpConfig needs the fields \\['target_layer'\\]"):
        JumpConfig.from_dict({"source_layer": 4})


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(target_layer=2, source_layer=4, exit_layer=7), "exceeds the model's 6 layers"),
        (dict(target_layer=2, source_layer=4, extra_source_layers=(1,), mixer="layer_attention", mixer_heads=5),
         "not divisible"),
    ],
)
def test_jump_config_is_checked_against_the_model(tiny_factory, kwargs: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        economize.apply(tiny_factory(), JumpConfig(**kwargs))


def test_unknown_predictor_or_mixer_names_list_the_choices(tiny_factory) -> None:
    with pytest.raises(UnknownNameError, match="'linear', 'mlp'"):
        economize.apply(tiny_factory(), JumpConfig(target_layer=2, source_layer=4, predictor="transformer"))
    with pytest.raises(UnknownNameError, match="source mixer 'sum'"):
        economize.apply(tiny_factory(), JumpConfig(target_layer=2, source_layer=4, mixer="sum"))


def test_decode_config_validation_and_resolution() -> None:
    with pytest.raises(ConfigError, match="cycle_length"):
        DecodeConfig(cycle_length=0)
    with pytest.raises(ConfigError, match="min_confidence"):
        DecodeConfig(min_confidence=1.5)
    assert DecodeConfig().verifier_name == "greedy"
    assert DecodeConfig(lossless=False).verifier_name == "accept_all"
    assert DecodeConfig(lossless=False, verifier="greedy").verifier_name == "greedy"
    assert DecodeConfig().draft_policy_name == "fixed"
    assert DecodeConfig(min_confidence=0.3).draft_policy_name == "confidence"


def test_decode_config_round_trip() -> None:
    config = DecodeConfig(cycle_length=6, lossless=False, min_confidence=0.25)
    assert DecodeConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    with pytest.raises(ConfigError, match="unknown DecodeConfig fields"):
        DecodeConfig.from_dict({"N": 4})


def test_decode_config_options_round_trip_and_hash() -> None:
    config = DecodeConfig(cycle_length=4, verifier="typical", verifier_options={"alpha": 0.5}, cuda_graphs=True)
    again = DecodeConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert again == config and hash(again) == hash(config)
    assert len({config, again, DecodeConfig(cycle_length=4, verifier="typical")}) == 2
    options = {"alpha": 0.5}
    copied = DecodeConfig(verifier="typical", verifier_options=options)
    options["alpha"] = 0.1  # the config keeps its own copy
    assert copied.verifier_options == {"alpha": 0.5}
    with pytest.raises(ConfigError, match="mapping"):
        DecodeConfig(verifier_options=[("alpha", 0.5)])  # type: ignore[arg-type]


def test_confidence_policy_needs_a_threshold(tiny_factory) -> None:
    model = economize.apply(tiny_factory(), JumpConfig(target_layer=2, source_layer=4))
    with pytest.raises(ConfigError, match="needs min_confidence"):
        model.decoder(DecodeConfig(draft_policy="confidence"))
