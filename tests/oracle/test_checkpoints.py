"""economize against ``tskip.decode.Decoder`` with trained research checkpoints on Qwen2.5-0.5B (float32, CPU).

Run with ``pytest -m slow``. The tests skip when tskip, the base model (in the Hugging Face cache) or a checkpoint is
missing. The checkpoints are only read.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

import economize
from economize import DecodeConfig
from economize.compat.tskip import cost_events

tskip_core = pytest.importorskip("tskip.core")
tskip_decode = pytest.importorskip("tskip.decode")

pytestmark = [pytest.mark.oracle, pytest.mark.slow, pytest.mark.filterwarnings("ignore:User provided device_type")]

BASE = "Qwen/Qwen2.5-0.5B"
RUNS = Path(__file__).resolve().parents[3] / "runs"
CHECKPOINTS = [
    "q05L_fr_m12_u18",  # frozen, m=12, u=18, r=L
    "q05_lo_m6_u18_r20",  # LoRA on layers 7..20, readout after layer 20
    "je_fu_m15_u17_r17_lin",  # linear predictor, trained copies of layers 16..17, readout after layer 17
]
TEXT = (
    "The water cycle describes how water moves between the oceans, the atmosphere and the land. Heat from the sun "
    "evaporates water from the surface of the sea, and the vapour rises until it cools"
)
MAX_NEW = 40


@pytest.fixture(scope="module")
def lm():
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache(BASE, "config.json"), str):
        pytest.skip(f"{BASE} is not in the Hugging Face cache")
    return tskip_core.LM(BASE, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def text_prompt() -> torch.Tensor:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BASE)(TEXT, return_tensors="pt").input_ids[0]


def _checkpoint(name: str) -> Path:
    path = RUNS / name / "jump.pt"
    if not path.exists():
        pytest.skip(f"checkpoint {path} is missing")
    return path.parent


def _tskip_decoder(lm, run: Path):
    state = torch.load(run / "jump.pt", map_location="cpu", weights_only=False)
    jump = tskip_core.Jump(lm, tskip_core.JumpConfig(**state["cfg"])).float()
    jump.load_state_dict(state["state"])
    return tskip_decode.Decoder(lm, jump.eval(), maxT=256)


@pytest.mark.parametrize("name", CHECKPOINTS)
def test_trained_drafters_match_tskip(lm, text_prompt: torch.Tensor, name: str) -> None:
    run = _checkpoint(name)
    decoder = _tskip_decoder(lm, run)
    model = economize.load(copy.deepcopy(lm.hf), run)  # detected as a tskip checkpoint
    for lossless in (True, False):
        for cycle_length in (3, 6):
            tokens, info = decoder.generate(text_prompt, MAX_NEW, N=cycle_length, mode="spec" if lossless else "lossy")
            result = model.decode(text_prompt, MAX_NEW, decoding=DecodeConfig(cycle_length, lossless=lossless))
            assert result.tokens.tolist() == tokens
            assert result.acceptance == info["acc"]
            assert cost_events(result.cost) == info["log"]
            if lossless:  # the comparison covers accepted and rejected drafts
                drafted, accepted = map(sum, zip(*result.acceptance))
                assert 0 < accepted < drafted


def test_trained_drafter_generates_like_transformers(lm, text_prompt: torch.Tensor) -> None:
    run = _checkpoint(CHECKPOINTS[0])
    model = economize.from_pretrained(run, dtype=torch.float32)  # the base model named in the checkpoint
    assert model.techniques["jump"].layer_span == (13, 24)
    options = dict(max_new_tokens=MAX_NEW, do_sample=False, repetition_penalty=1.3)
    expected = lm.hf.generate(text_prompt[None], **options)
    tokens = model.generate(text_prompt[None], decoding=DecodeConfig(cycle_length=4), **options)
    assert tokens.tolist() == expected.tolist()
