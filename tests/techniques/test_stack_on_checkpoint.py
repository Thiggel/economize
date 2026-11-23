"""Techniques stacked on a saved drafter checkpoint.

A trained drafter is often combined with techniques that were never saved with it: weight quantization or pruning,
which rewrite the layers its trained copies were made from, calibrated techniques that fit their state at attach time,
and depth pruning, which renumbers the layers the drafter names. Its trained state must survive each of these, and
drafting must stay lossless on the modified model.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

import economize
from economize import DecodeConfig, JumpConfig
from economize.core.serialization import Checkpoint, read_checkpoint
from economize.errors import ConfigError
from economize.techniques.lowrank import LowRankConfig
from economize.techniques.mlp_sparsity import CatsConfig
from economize.techniques.prune import DepthPruneConfig, PruneConfig
from economize.techniques.quant import QuantConfig
from economize.techniques.width_prune import WidthPruneConfig
from helpers import VOCAB, make_tiny_model, perturb, reference_greedy

PROMPT = torch.randint(0, VOCAB, (1, 8), generator=torch.Generator().manual_seed(3))


def _saved_jump(tmp_path: Path, **fields: object):
    config = JumpConfig(**{"target_layer": 2, "source_layer": 4, "adaptation": "full", **fields})
    trained = economize.apply(make_tiny_model(), config)
    perturb(trained.techniques["jump"])
    economize.save(trained, tmp_path / "jump")
    return read_checkpoint(tmp_path / "jump")


def _assert_lossless(model) -> None:
    result = model.decode(PROMPT, 12, decoding=DecodeConfig(cycle_length=3))
    assert result.tokens.tolist() == reference_greedy(economize.unwrap.base_model_of(model), PROMPT, 12).tolist()


def _assert_same_state(model, saved, name: str = "jump") -> None:
    state = model.techniques[name].state_dict()
    assert state.keys() == saved.states[name].keys()
    for key, value in saved.states[name].items():
        torch.testing.assert_close(state[key].float(), value.float(), msg=key)


@pytest.mark.parametrize("weights", [QuantConfig(format="int8", symmetric=True, group_size=None, backend="fake"),
                                     PruneConfig(method="wanda", sparsity="2:4", semi_structured=False),
                                     LowRankConfig(method="svd", compression=0.6)],
                         ids=["int8", "2:4", "lowrank"])
def test_a_trained_full_copy_drafter_stacks_with_a_weight_transform(weights, tmp_path: Path, calib) -> None:
    # The copies of the trained layers stay dense with their saved values; only the base layers are rewritten (a copy
    # of a factorized layer would not even hold the saved dense weights).
    saved = _saved_jump(tmp_path)
    model = economize.attach(make_tiny_model(), Checkpoint([*saved.configs, weights], dict(saved.states)),
                             calibration=[calib], check_teacher=False)
    assert set(model.techniques) == {"jump", weights.technique}
    _assert_same_state(model, saved)
    _assert_lossless(model)


@pytest.mark.parametrize("stacked", [CatsConfig(sparsity=0.5, backend="dense"),
                                     WidthPruneConfig(method="flap", sparsity=0.4)], ids=["cats", "flap"])
def test_a_calibrated_technique_without_saved_state_is_fitted_at_attach(stacked, tmp_path: Path, calib) -> None:
    saved = _saved_jump(tmp_path)
    model = economize.attach(make_tiny_model(), Checkpoint([*saved.configs, stacked], dict(saved.states)),
                             calibration=[calib], check_teacher=False)
    assert stacked.technique in model.techniques
    result = model.decode(PROMPT, 12, decoding=DecodeConfig(cycle_length=3))
    assert result.tokens.shape[-1] >= 12


def test_depth_pruning_outside_the_span_renumbers_the_drafter(tmp_path: Path, calib) -> None:
    # Six layers; the drafter jumps 2 -> 4 through layers 3..6. Removing layer 1 shifts every named layer down by one.
    saved = _saved_jump(tmp_path)
    prune = DepthPruneConfig(method="shortgpt", num_blocks=1, layers=(1,))
    model = economize.attach(make_tiny_model(), Checkpoint([*saved.configs, prune], dict(saved.states)),
                             calibration=[calib], check_teacher=False)
    config = model.techniques["jump"].config
    assert (config.target_layer, config.source_layer) == (1, 3)
    state = model.techniques["jump"].state_dict()
    renamed = {re.sub(r"^blocks\.(\d+)\.", lambda m: f"blocks.{int(m[1]) - 1}.", k): v
               for k, v in saved.states["jump"].items()}
    assert state.keys() == renamed.keys()
    for key, value in renamed.items():
        torch.testing.assert_close(state[key], value, msg=key)
    _assert_lossless(model)


def test_depth_pruning_a_layer_the_drafter_names_is_rejected(tmp_path: Path, calib) -> None:
    saved = _saved_jump(tmp_path)
    prune = DepthPruneConfig(method="shortgpt", num_blocks=1, layers=(3,))
    with pytest.raises(ConfigError):
        economize.attach(make_tiny_model(), Checkpoint([*saved.configs, prune], dict(saved.states)),
                         calibration=[calib], check_teacher=False)
