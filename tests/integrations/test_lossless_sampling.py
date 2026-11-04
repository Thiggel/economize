"""Lossless sampling at temperature 1: jump drafting with rejection sampling must sample the target's distribution.

This is the property vLLM's ``rejection_sample_method="standard"`` relies on when it verifies jump drafts with the
drafter's full-vocabulary ``q`` (RL rollouts, experiment E13). It is checked on the economize core in float32: the
empirical distribution of the first generated tokens over many seeds is compared with the exact distribution of
the target, and the tolerance is the spread of the same statistic for samples drawn from the exact distribution.
The test skips if the ``"rejection"`` verifier is unregistered.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from economize.errors import ConfigError
from helpers import PROMPT_LEN, VOCAB

SEEDS = 600
POSITIONS = (0, 1, 2)


def _require_rejection() -> None:
    from economize.decoding.verifiers import VERIFIERS

    if "rejection" not in VERIFIERS:
        pytest.skip(f"the core has no 'rejection' verifier yet (verifiers: {sorted(VERIFIERS.names())})")


@torch.no_grad()
def _exact_marginals(base: Any, prompt: torch.Tensor) -> list[torch.Tensor]:
    """Exact marginals of generated tokens 0, 1, 2 at temperature 1 (enumerating tokens 0 and 1)."""
    p0 = base(prompt[None]).logits[0, -1].float().softmax(-1)
    first = torch.cat([prompt.expand(VOCAB, -1), torch.arange(VOCAB)[:, None]], dim=1)
    p1_given_0 = base(first).logits[:, -1].float().softmax(-1)  # [x0, x1]
    joint01 = p0[:, None] * p1_given_0
    p2 = torch.zeros(VOCAB)
    for x0 in range(VOCAB):
        rows = torch.cat([first[x0].expand(VOCAB, -1), torch.arange(VOCAB)[:, None]], dim=1)
        p2 += (joint01[x0][:, None] * base(rows).logits[:, -1].float().softmax(-1)).sum(0)
    return [p0, joint01.sum(0), p2]


def _tv(counts: torch.Tensor, probs: torch.Tensor) -> float:
    return 0.5 * float((counts / counts.sum() - probs).abs().sum())


def test_rejection_sampling_is_lossless_at_temperature_one(jump_model: Callable[..., Any]) -> None:
    _require_rejection()
    import economize
    from economize.decoding.selectors import TopPSampler

    model = jump_model(cycle_length=4, seed=3)
    base = model.base_model
    with torch.no_grad():
        base.model.norm.weight.mul_(4.0)  # a peaked target: few likely tokens, so few samples resolve it
    prompt = torch.randint(1, VOCAB, (PROMPT_LEN,), generator=torch.Generator().manual_seed(5))
    exact = _exact_marginals(base, prompt)

    decoding = economize.DecodeConfig(cycle_length=4, verifier="rejection")
    counts = torch.zeros(len(POSITIONS), VOCAB)
    for seed in range(SEEDS):
        tokens = model.decode(prompt, len(POSITIONS), decoding=decoding, selector=TopPSampler(seed=seed)).tokens
        for position in POSITIONS:
            counts[position, int(tokens[position])] += 1

    generator = torch.Generator().manual_seed(0)
    for position in POSITIONS:
        reference = [_tv(torch.bincount(torch.multinomial(exact[position], SEEDS, True, generator=generator),
                                        minlength=VOCAB).float(), exact[position]) for _ in range(200)]
        tolerance = 1.5 * max(reference)
        assert _tv(counts[position], exact[position]) < tolerance, f"token {position} is not sampled losslessly"


# ---- seeded selector of generate -----------------------------------------------------------------------------------
def test_processed_selector_hands_out_its_sampling_stream() -> None:
    from economize.decoding.capabilities import SeededSelector, selector_generator
    from economize.integrations._generation_args import ProcessedSelector

    logits, history = torch.randn(32, generator=torch.Generator().manual_seed(0)), torch.tensor([1, 2])
    default = ProcessedSelector(None, do_sample=True)
    assert isinstance(default, SeededSelector)
    assert selector_generator(default, torch.device("cpu")) is torch.default_generator  # transformers' global RNG
    torch.manual_seed(5)
    first = [default(logits, history) for _ in range(8)]
    torch.manual_seed(5)
    assert [int(torch.multinomial(torch.softmax(logits, -1), 1)[0]) for _ in range(8)] == first

    private = torch.Generator().manual_seed(7)
    seeded = ProcessedSelector(None, do_sample=True, generator=private)
    assert seeded.generator_for(torch.device("cpu")) is private
    with pytest.raises(ConfigError, match="on cpu but the logits are on cuda"):
        seeded.generator_for(torch.device("cuda", 0))
