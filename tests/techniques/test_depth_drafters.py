"""Dynamic-depth techniques as self-speculative drafters: ``DraftWith(<depth config>)``.

The technique's policy acts on the draft path only and the unchanged model verifies, so greedy decoding equals plain
greedy decoding exactly (fp32) while the drafts skip layers. The verify pass is a plain pass over every layer at the
drafted positions: it writes the K/V of the layers a draft step skipped, so later steps never read draft K/V.
"""

from __future__ import annotations

import pytest
import torch

import economize
from economize import DecodeConfig
from economize.techniques.draft_with import DraftWith
from economize.techniques.dynamic_depth import CalmConfig, GateSkipConfig, LoRADropConfig, SkipDecodeConfig
from helpers import VOCAB, make_tiny_model, perturb, reference_greedy

NEW = [30, 21, 26]

#: A router method, an exit method, a schedule and a periodic method, each set to skip layers on the tiny model
#: (and whether its parameters must be perturbed for that).
CASES = {
    "gateskip": (GateSkipConfig(gate_bias=0.0, gate_init_std=0.5), False),
    "calm": (CalmConfig(confidence="state", threshold=0.5), False),
    "calm-strided": (CalmConfig(confidence="state", threshold=0.5, exit_stride=2), False),
    "skipdecode": (SkipDecodeConfig(min_layers=2, decay=20, skip="top"), False),
    "loradrop": (LoRADropConfig(period=2), True),
}


def _prompts(lengths: tuple[int, ...] = (12, 7, 20)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in lengths]


def _drafter(case: str, cycle_length: int = 4) -> tuple[torch.nn.Module, economize.EfficientModel]:
    base = make_tiny_model()
    config, needs_noise = CASES[case]
    model = economize.apply(base, DraftWith(config), decoding=DecodeConfig(cycle_length=cycle_length))
    if needs_noise:
        perturb(model.techniques["draft_with"].held[config.technique], 0.3)
    return base, model


def _skipped(model: economize.EfficientModel, case: str) -> dict[str, object]:
    technique = model.techniques["draft_with"].held[CASES[case][0].technique]
    return technique.depth_stats()


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
def test_depth_drafter_is_lossless_and_skips_layers(case: str, batch: bool) -> None:
    """One row runs the step engine (DepthSteps), a batch the policy's eager path; both equal plain greedy."""
    base, model = _drafter(case)
    prompts = _prompts() if batch else _prompts()[:1]
    new = NEW if batch else NEW[:1]
    results = model.decode_batch(prompts, new)
    for prompt, result, n in zip(prompts, results, new):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, n))
    drafted = sum(k for r in results for k, _ in r.acceptance)
    accepted = sum(a for r in results for _, a in r.acceptance)
    assert drafted > 0 and accepted < drafted  # the draft path differs from the verifier
    stats = _skipped(model, case)
    assert int(stats["skipped_layer_tokens"]) > 0
    assert set(stats["phases"]) <= {"draft"}  # the policy never acts on the verify or prefill passes


def test_verify_refills_the_kv_of_skipped_layers() -> None:
    """Long generations with many skipped draft layers stay exact: an accepted draft token whose skipped layers kept
    no K/V would change every later verify pass, so equality with plain greedy needs the verify pass to refill them."""
    base, model = _drafter("gateskip", cycle_length=6)
    prompt = _prompts()[2]
    result = model.decode_batch([prompt], [60])[0]
    assert torch.equal(result.tokens, reference_greedy(base, prompt, 60))
    assert sum(a for _, a in result.acceptance) > 0
    assert int(_skipped(model, "gateskip")["skipped_layer_tokens"]) > 0
