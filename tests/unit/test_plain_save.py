"""``transformers``' plain ``save_pretrained`` of a base model that a weight technique rewrote.

Online Hadamard rotations live in forward hooks and low-rank factors in modules ``transformers`` cannot rebuild, so
a plain save of such a base reloads into a different function without any error. The base model's
``save_pretrained`` refuses those cases and still saves (exactly) everything folded into the weights.
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM

import economize
from economize.core.config import TechniqueConfig
from economize.errors import ConfigError
from economize.techniques.lowrank import LowRankConfig
from economize.techniques.quant import QuantConfig
from economize.techniques.rotate import RotateConfig
from helpers import VOCAB, make_tiny_model

RTN = QuantConfig(format="int4", method="rtn", group_size=32, backend="fake")


def _apply(*configs: TechniqueConfig) -> economize.EfficientModel:
    calibration = torch.randint(0, VOCAB, (8, 32), generator=torch.Generator().manual_seed(0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return economize.apply(make_tiny_model(num_layers=2), *configs, calibration=calibration)


@pytest.mark.parametrize(
    ("configs", "names"),
    [((RotateConfig(), RTN), "rotate: the online rotations"), ((LowRankConfig(),), "lowrank: factorized maps")],
    ids=["rotate_online", "lowrank"],
)
def test_plain_save_refuses_a_base_it_would_save_wrongly(
    configs: tuple[TechniqueConfig, ...], names: str, tmp_path: Path
) -> None:
    model = _apply(*configs)
    with pytest.raises(ConfigError, match=names) as caught:
        model.base_model.save_pretrained(tmp_path / "plain")
    assert "economize.save" in str(caught.value)
    assert not (tmp_path / "plain").exists()
    twin = copy.deepcopy(model.base_model)  # a copy is guarded too, and saves itself
    with pytest.raises(ConfigError, match=names):
        twin.save_pretrained(tmp_path / "twin")
    state = {key: value.clone() for key, value in twin.state_dict().items()}
    twin.save_pretrained(tmp_path / "chosen", state_dict=state)  # what the serving exporters do
    assert (tmp_path / "chosen" / "config.json").exists()


def test_plain_save_keeps_working_when_every_transform_is_in_the_weights(tmp_path: Path) -> None:
    model = _apply(RotateConfig(online=()), RTN)
    model.base_model.save_pretrained(tmp_path / "plain")
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "plain")
    x = torch.randint(0, VOCAB, (2, 16), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        torch.testing.assert_close(reloaded(x).logits, model(x).logits, atol=1e-5, rtol=0)
