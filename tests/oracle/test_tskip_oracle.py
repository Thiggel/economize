"""economize against the research decoder ``tskip.decode.Decoder`` on a tiny model.

Both decoders get the same base weights and the same jump parameters (tskip's, converted by the ``tskip`` checkpoint
format), and must produce the same tokens, the same log of model calls and the same acceptance history.
"""

from __future__ import annotations

import copy
import dataclasses

import pytest
import torch

import economize
from economize import DecodeConfig, TopPSampler
from economize.compat.tskip import cost_events, jump_config_from_tskip, jump_state_from_tskip
from economize.core.serialization import Checkpoint
from helpers import MAX_POSITIONS, make_tiny_model, perturb

tskip_core = pytest.importorskip("tskip.core")
tskip_decode = pytest.importorskip("tskip.decode")

pytestmark = [pytest.mark.oracle, pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")]

MAX_NEW = 24

VARIANTS = {
    "frozen": dict(m=2, u=4, r=6),
    "readout": dict(m=2, u=4, r=5),
    "lora": dict(m=2, u=4, r=6, mode="lora", lora_rank=4),
    "lora_readout": dict(m=1, u=3, r=5, mode="lora", lora_rank=4),
    "full": dict(m=2, u=4, r=5, mode="full"),
    "m_equals_u": dict(m=3, u=3, r=6),
    "from_embedding": dict(m=0, u=2, r=6),
    "concat": dict(m=2, u=4, r=6, src="0,3,4"),
    "layer_attention": dict(m=2, u=5, r=6, src="1,3,5", mix="lattn"),
    "fused_linear": dict(m=2, u=4, r=6, src="1,4", mix="fc"),
    "linear": dict(m=2, u=4, r=6, pred="lin"),
    "deep": dict(m=2, u=4, r=6, depth=2, width_mult=1.0),
    "no_state": dict(m=2, u=4, r=6, use_state=False),
    "no_token": dict(m=2, u=4, r=6, use_action=False, emb_residual=False),
}


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("tiny")
    make_tiny_model().save_pretrained(path)
    return str(path)


def _pair(base_dir: str, std: float = 0.05, near_identity: bool = False, **cfg):
    """A tskip decoder with random jump parameters, and economize with the same base weights and converted parameters.

    ``near_identity``: start from a predictor that returns the token embedding (with ``m = u = 0``, ``r = L`` every
    draft would be exact), so that a small ``std`` gives a mix of accepted and rejected drafts.
    """
    lm = tskip_core.LM(base_dir, device="cpu", dtype=torch.float32)
    torch.manual_seed(0)
    jump = tskip_core.Jump(lm, tskip_core.JumpConfig(**cfg)).float()
    if near_identity:
        with torch.no_grad():
            for name, parameter in jump.pred.named_parameters():
                if name != "out_gain":
                    parameter.zero_()
    perturb(jump, std)
    base = copy.deepcopy(lm.hf)  # economize's LoRA hooks must not reach tskip's model
    config = jump_config_from_tskip(dataclasses.asdict(jump.cfg))
    model = economize.attach(base, Checkpoint([config], {"jump": jump_state_from_tskip(jump.state_dict(), config)}))
    return tskip_decode.Decoder(lm, jump, maxT=MAX_POSITIONS), model


def _compare(
    decoder, model, prompt, cycle_length: int, lossless: bool, tau=None, sample=None
) -> economize.DecodeResult:
    tokens, info = decoder.generate(
        prompt, MAX_NEW, N=cycle_length, mode="spec" if lossless else "lossy", tau=tau, sample=sample, seed=3
    )
    selector = TopPSampler(*sample, seed=3) if sample else None
    config = DecodeConfig(cycle_length=cycle_length, lossless=lossless, min_confidence=tau)
    result = model.decode(prompt, MAX_NEW, decoding=config, selector=selector)
    assert result.tokens.tolist() == tokens
    assert result.acceptance == info["acc"]
    assert cost_events(result.cost) == info["log"]
    return result


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("lossless", [True, False])
def test_jump_decoding_matches_tskip(base_dir: str, prompt, variant: str, lossless: bool) -> None:
    decoder, model = _pair(base_dir, **VARIANTS[variant])
    for cycle_length in (1, 3, 6):
        _compare(decoder, model, prompt, cycle_length, lossless)


@pytest.mark.parametrize("adaptation", ["frozen", "lora"])
@pytest.mark.parametrize("std", [0.01, 0.02])
def test_partial_acceptance_matches_tskip(base_dir: str, prompt, adaptation: str, std: float) -> None:
    decoder, model = _pair(base_dir, std, near_identity=True, m=0, u=0, r=6, mode=adaptation, lora_rank=4)
    result = _compare(decoder, model, prompt, 4, lossless=True)
    accepted = [a for _, a in result.acceptance]
    assert 0 < sum(accepted) < 3 * len(accepted)  # some drafts accepted, some rejected
    _compare(decoder, model, prompt, 4, lossless=True, tau=0.3)
    _compare(decoder, model, prompt, 4, lossless=False)


@pytest.mark.parametrize("lossless", [True, False])
def test_confidence_stop_matches_tskip(base_dir: str, prompt, lossless: bool) -> None:
    decoder, model = _pair(base_dir, m=2, u=4, r=6)
    # the thresholds bracket the drafts' confidence, so drafting stops early in some cycles and not in others
    wasted = [
        _compare(decoder, model, prompt, 6, lossless, tau=tau).cost.count("draft_wasted") for tau in (0.05, 0.2, 0.5)
    ]
    assert wasted[0] < wasted[1] <= wasted[2]


@pytest.mark.parametrize("variant", ["frozen", "readout"])
def test_sampling_matches_tskip(base_dir: str, prompt, variant: str) -> None:
    decoder, model = _pair(base_dir, **VARIANTS[variant])
    for cycle_length in (1, 4):
        _compare(decoder, model, prompt, cycle_length, lossless=False, sample=(0.8, 0.9))
        _compare(decoder, model, prompt, cycle_length, lossless=False, tau=0.2, sample=(1.2, 1.0))
