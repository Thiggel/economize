"""Translation of the research checkpoints of ``tskip``: every key the translation reads is validated up front, and
every failure is a :class:`~economize.CheckpointFormatError` (naming the checkpoint when one is read from disk)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import economize
from economize import CheckpointFormatError
from economize.compat.tskip import jump_config_from_tskip

CFG = dict(m=2, u=4, r=5, mode="frozen", pred="mlp", width_mult=2.0, depth=1, feed="layers", src="", mix="cat")


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(pred="block"), "predictor 'block'"),
        (dict(pred=None), "predictor None"),
        (dict(feed="fused"), "feed='fused'"),
        (dict(mix="gate"), "mix='gate'"),
        (dict(mode="half"), "mode='half'"),
        (dict(mode=None), "mode=None"),
        (dict(m="two"), "'m'"),
        (dict(r=None), "'r'"),
        (dict(u=4.5), "'u'"),
        (dict(src="1,x,4"), "src='1,x,4'"),
        (dict(pred_depth=0), "pred_depth=0"),
        (dict(pred_depth="deep"), "pred_depth='deep'"),
    ],
)
def test_invalid_tskip_keys_are_format_errors(overrides: dict, message: str) -> None:
    with pytest.raises(CheckpointFormatError, match=message):
        jump_config_from_tskip({**CFG, **overrides})


@pytest.mark.parametrize("missing", ["m", "u", "r"])
def test_missing_layer_keys_are_format_errors(missing: str) -> None:
    """Regression: a missing ``m``/``u``/``r`` raised a bare ``KeyError``."""
    cfg = {key: value for key, value in CFG.items() if key != missing}
    with pytest.raises(CheckpointFormatError, match=f"'{missing}'"):
        jump_config_from_tskip(cfg)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(m=5, u=4), "target_layer <= source_layer"),
        (dict(u=6, r=5), "source_layer <= exit_layer"),
        (dict(width_mult=-1.0), "predictor_width must be positive"),
        (dict(lora_rank=0), "lora_rank"),
    ],
)
def test_invalid_jump_configs_are_format_errors(overrides: dict, message: str) -> None:
    """Regression: an inconsistent tskip config raised the ``ConfigError`` of :class:`JumpConfig` unchanged."""
    with pytest.raises(CheckpointFormatError, match=message):
        jump_config_from_tskip({**CFG, **overrides})


def test_errors_name_the_checkpoint(tmp_path: Path) -> None:
    torch.save({"cfg": {**CFG, "m": 7}, "state": {}}, tmp_path / "jump.pt")
    with pytest.raises(CheckpointFormatError, match=r"jump\.pt.*target_layer <= source_layer"):
        economize.read_checkpoint(tmp_path)
    torch.save({"cfg": CFG, "state": {"head.weight": torch.zeros(1)}}, tmp_path / "jump.pt")
    with pytest.raises(CheckpointFormatError, match=r"jump\.pt.*unexpected parameter 'head\.weight'"):
        economize.read_checkpoint(tmp_path)


def test_pred_depth_is_the_number_of_predictor_layers() -> None:
    assert jump_config_from_tskip(CFG).predictor_layers == 1
    assert jump_config_from_tskip({**CFG, "pred_depth": 3}).predictor_layers == 3
    assert jump_config_from_tskip({**CFG, "depth": 2}).predictor_depth == 2  # tskip's depth stays the MLP's stages


def test_the_layer_attention_mixer_needs_heads_that_divide_the_hidden_size() -> None:
    from economize.errors import ConfigError
    from economize.techniques.jump import JumpConfig
    from economize.techniques.jump.mixer import LayerAttentionMixer

    config = JumpConfig(target_layer=1, source_layer=3, extra_source_layers=(2,), mixer="layer_attention",
                        mixer_heads=3)
    with pytest.raises(ConfigError, match="mixer_heads"):
        LayerAttentionMixer(64, config)


def test_unknown_tskip_mixer_is_a_format_error() -> None:
    """Regression: an unknown tskip ``mix`` raised a bare ``KeyError`` instead of a ``CheckpointFormatError``."""
    cfg = dict(m=2, u=4, r=6, src="1,4", mix="gate")
    with pytest.raises(CheckpointFormatError, match="mix='gate'"):
        jump_config_from_tskip(cfg)
    assert jump_config_from_tskip({**cfg, "mix": "lattn"}).mixer == "layer_attention"
