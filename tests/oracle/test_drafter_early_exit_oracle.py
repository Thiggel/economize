"""Early-exit drafting against the research decoder: a stateless ``tskip`` jump from the embedding (``m = 0``,
``u = r < L``, ``use_state = False``, the ``ee_*_ns`` runs) whose predictor returns the token embedding is the
early-exit technique with a readout head. Both decoders must emit the same tokens, acceptance and call log."""

from __future__ import annotations

import copy
import dataclasses

import pytest
import torch

import economize
from economize import DecodeConfig
from economize.compat.tskip import cost_events, early_exit_config_from_tskip, early_exit_state_from_tskip
from economize.core.serialization import Checkpoint
from economize.errors import CheckpointFormatError
from helpers import MAX_POSITIONS, make_tiny_model, perturb

tskip_core = pytest.importorskip("tskip.core")
tskip_decode = pytest.importorskip("tskip.decode")

pytestmark = [pytest.mark.oracle, pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")]

MAX_NEW = 24
LAYERS = 6


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("tiny_ee")
    make_tiny_model().save_pretrained(path)
    return str(path)


def _pair(base_dir: str, r: int, mode: str, std: float):
    lm = tskip_core.LM(base_dir, device="cpu", dtype=torch.float32)
    torch.manual_seed(0)
    cfg = tskip_core.JumpConfig(m=0, u=r, r=r, mode=mode, lora_rank=4, use_state=False)
    jump = tskip_core.Jump(lm, cfg).float()
    perturb(jump, std)
    with torch.no_grad():  # a predictor that returns the token embedding: the early-exit drafter has no such term
        for name in ("skip", "out"):
            getattr(jump.pred, name).weight.zero_()
        jump.pred.bias.zero_()
    config = early_exit_config_from_tskip(dataclasses.asdict(cfg), lm.L)
    checkpoint = Checkpoint([config], {"early_exit": early_exit_state_from_tskip(jump.state_dict())})
    model = economize.attach(copy.deepcopy(lm.hf), checkpoint)
    return tskip_decode.Decoder(lm, jump, maxT=MAX_POSITIONS), model


@torch.no_grad()
@pytest.mark.parametrize("mode", ["frozen", "lora", "full"])
@pytest.mark.parametrize("r", [2, 5])
def test_early_exit_matches_tskip(base_dir: str, prompt, mode: str, r: int) -> None:
    decoder, model = _pair(base_dir, r, mode, std=0.03)
    assert model.drafting_technique.config.adaptation == mode
    accepted = 0
    for cycle_length in (1, 3, 6):
        tokens, info = decoder.generate(prompt, MAX_NEW, N=cycle_length, mode="spec", seed=3)
        result = model.decode(prompt, MAX_NEW, decoding=DecodeConfig(cycle_length=cycle_length))
        assert result.tokens.tolist() == tokens
        assert result.acceptance == info["acc"]
        assert cost_events(result.cost) == info["log"]
        accepted += sum(a for _, a in result.acceptance)
    if r == LAYERS - 1:
        assert accepted > 0


def test_conversion_refuses_jumps_that_are_not_early_exits(base_dir: str) -> None:
    lm = tskip_core.LM(base_dir, device="cpu", dtype=torch.float32)
    for cfg in (dict(m=1, u=3, r=3, use_state=False), dict(m=0, u=3, r=3), dict(m=0, u=2, r=3, use_state=False),
                dict(m=0, u=LAYERS, r=LAYERS, use_state=False)):
        with pytest.raises(CheckpointFormatError):
            early_exit_config_from_tskip(dataclasses.asdict(tskip_core.JumpConfig(**cfg)), lm.L)
    jump = tskip_core.Jump(lm, tskip_core.JumpConfig(m=0, u=3, r=3, use_state=False))  # trained-like predictor
    with pytest.raises(CheckpointFormatError, match="not zero"):
        early_exit_state_from_tskip(jump.state_dict())
