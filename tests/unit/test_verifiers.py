"""Verifiers and selectors: batched paths equal per-row paths, and rejection sampling samples the target.

The distribution tests use Pearson's chi-square statistic. Bins whose expected count is below 5 are merged, and the
statistic is turned into an approximate standard normal score with the Wilson-Hilferty transform; a correct sampler
stays below ``Z_MAX`` (a false alarm has probability ~3e-5). Each test also runs a negative control, keeping every
draft (``accept_all``), which must fail the same check, so the check is known to have power.
"""

from __future__ import annotations

import math

import pytest
import torch

from economize import ConfigError, DecodeConfig
from economize.arch import adapter_for
from economize.decoding import (
    VERIFIERS,
    AcceptAllVerifier,
    CycleDecoder,
    Draft,
    DraftRequest,
    GreedySelector,
    GreedyVerifier,
    RejectionVerifier,
    TopPSampler,
    TypicalVerifier,
)
from economize.runtime import LayerRunner
from helpers import VOCAB, ScriptedDrafter, ScriptedTreeDrafter, make_tiny_model, reference_greedy

Z_MAX = 4.0


def chi_square_z(counts: torch.Tensor, probs: torch.Tensor) -> float:
    """Wilson-Hilferty normal score of Pearson's chi-square statistic of ``counts`` against ``probs``."""
    n = float(counts.sum())
    expected = probs.double() * n
    order = expected.argsort(descending=True)
    big = order[expected[order] >= 5]
    small = order[expected[order] < 5]
    observed = torch.cat([counts.double()[big], counts.double()[small].sum().view(1)])
    expected = torch.cat([expected[big], expected[small].sum().view(1)])
    if float(expected[-1]) < 5:  # fold a tiny remainder into the smallest kept bin
        observed = torch.cat([observed[:-2], observed[-2:].sum().view(1)])
        expected = torch.cat([expected[:-2], expected[-2:].sum().view(1)])
    df = len(expected) - 1
    statistic = float(((observed - expected) ** 2 / expected).sum())
    return ((statistic / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))


# ---- selectors ---------------------------------------------------------------------------------------------------
def test_select_many_equals_call_for_one_row() -> None:
    logits = torch.randn(20, 50, generator=torch.Generator().manual_seed(0))
    greedy = GreedySelector()
    assert greedy.select_many(logits).tolist() == [greedy(row, torch.zeros(0)) for row in logits]
    one, many = TopPSampler(1.3, 0.9, seed=4), TopPSampler(1.3, 0.9, seed=4)
    assert [one(row, torch.zeros(0)) for row in logits] == [int(many.select_many(row[None])[0]) for row in logits]


def test_sampler_distribution_is_what_it_samples() -> None:
    logits = torch.randn(12, generator=torch.Generator().manual_seed(1))
    sampler = TopPSampler(0.8, 0.7, seed=0)
    dist = sampler.distribution(logits, torch.zeros(0))
    assert float(dist.sum()) == pytest.approx(1.0) and int((dist > 0).sum()) < 12  # the nucleus cuts the tail
    counts = torch.bincount(torch.tensor([sampler(logits, torch.zeros(0)) for _ in range(4000)]), minlength=12)
    assert chi_square_z(counts, dist) < Z_MAX


# ---- verifiers ---------------------------------------------------------------------------------------------------
def _logits(*probs: list[float]) -> torch.Tensor:
    """Logits of the given distributions, one row each, padded to 10 tokens with probability 1e-9."""
    table = torch.full((len(probs), 10), 1e-9)
    for i, row in enumerate(probs):
        table[i, : len(row)] = torch.tensor(row)
    return table.log()


def test_registered_verifiers() -> None:
    assert {"greedy", "rejection", "accept_all", "typical"} <= set(VERIFIERS.names())
    assert VERIFIERS.get("rejection") is RejectionVerifier


@pytest.mark.parametrize("verifier_class", [GreedyVerifier, AcceptAllVerifier])
def test_verify_batch_equals_verify_per_row(verifier_class) -> None:
    generator = torch.Generator().manual_seed(2)
    logits = torch.randn(4, 4, 9, generator=generator)
    choice = logits.argmax(-1)
    lengths = [3, 0, 2, 3]
    tokens = choice[:, :3].clone()
    tokens[2, 1] = (tokens[2, 1] + 1) % 9  # row 2: second draft wrong
    tokens[3, 0] = (tokens[3, 0] + 1) % 9  # row 3: first draft wrong
    for b, k in enumerate(lengths):
        tokens[b, k:] = -1
    draft = Draft(tokens, lengths)
    seq = torch.zeros(4, 10, dtype=torch.long)
    verifier = verifier_class(DecodeConfig())
    accepted, following = verifier.verify_batch(logits, draft, GreedySelector(), seq, [5] * 4)
    expected = [verifier.verify(logits[b, : k + 1], tokens[b, :k].tolist(), None, GreedySelector(), seq[b], 5)
                for b, k in enumerate(lengths)]
    assert list(zip(accepted.tolist(), following.tolist())) == expected
    if verifier_class is GreedyVerifier:
        assert accepted.tolist() == [3, 0, 1, 0]


def test_rejection_needs_a_distribution() -> None:
    class Plain:
        def __call__(self, logits, history):
            return 0

    with pytest.raises(ConfigError, match="distribution"):
        RejectionVerifier(DecodeConfig()).draft(torch.zeros(4), Plain(), torch.zeros(0))


def test_rejection_without_draft_distributions_treats_the_drafts_as_fixed() -> None:
    """``probs=None`` means one-hot drafts: a draft the target cannot produce is always rejected and never resampled,
    one it always produces is always kept."""
    verifier = RejectionVerifier(DecodeConfig())
    selector = TopPSampler(seed=0)
    logits = _logits([0.5, 0.5], [1.0], [1.0])  # token 2 has probability ~1e-9 at step 0
    history = torch.zeros(3, dtype=torch.long)
    for _ in range(50):
        accepted, token = verifier.verify(logits, [2, 0], None, selector, history, 1)
        assert accepted == 0 and token in (0, 1)
    certain = _logits([1.0], [1.0], [0.0, 1.0])
    assert verifier.verify(certain, [0, 0], None, selector, history, 1) == (2, 1)


@pytest.mark.parametrize("name", ["rejection", "accept_all"])
@pytest.mark.parametrize("batch", [False, True], ids=["verify", "verify_batch"])
def test_rejection_of_fixed_drafts_outputs_the_target_distribution(name: str, batch: bool) -> None:
    """Fixed drafts (no draft distributions) against ``p0``, ``p1``, ``p2``: the first two output tokens follow
    ``p0`` and ``p1``. The drafts are the most likely tokens of ``q``, so keeping them (``accept_all``) is far off."""
    generator = torch.Generator().manual_seed(3)
    target = torch.randn(3, 10, generator=generator) * 1.5
    drafts = [int(target[0].argmax()), int(target[1].argsort()[-2])]
    verifier = VERIFIERS.get(name)(DecodeConfig(verifier=name))
    selector = TopPSampler(seed=11)
    trials = 6000
    if batch:
        logits = target.expand(trials, -1, -1)
        draft = Draft(torch.tensor(drafts).expand(trials, -1).clone(), [2] * trials)
        seq = torch.zeros(trials, 4, dtype=torch.long)
        accepted_t, following = verifier.verify_batch(logits, draft, selector, seq, [1] * trials)
        results = list(zip(accepted_t.tolist(), following.tolist()))
    else:
        history = torch.zeros(3, dtype=torch.long)
        results = [verifier.verify(target, drafts, None, selector, history, 1) for _ in range(trials)]
    counts = torch.zeros(2, 10)
    for accepted, token in results:
        output = [*drafts[:accepted], token]
        counts[0, output[0]] += 1
        if len(output) > 1:
            counts[1, output[1]] += 1
    p = target.softmax(-1)
    scores = [chi_square_z(counts[i], p[i]) for i in range(2)]
    if name == "rejection":
        assert max(scores) < Z_MAX, scores
    else:
        assert min(scores) > 3 * Z_MAX, scores


@pytest.mark.parametrize("name", ["rejection", "accept_all"])
def test_rejection_sampling_outputs_the_target_distribution(name: str) -> None:
    """Two drafts from ``q0``, ``q1`` against exact distributions ``p0``, ``p1``, ``p2`` (independent of the
    history): the first two output tokens follow ``p0`` and ``p1``."""
    generator = torch.Generator().manual_seed(3)
    target = torch.randn(3, 10, generator=generator) * 1.5
    drafter = torch.randn(2, 10, generator=generator) * 1.5
    verifier = VERIFIERS.get(name)(DecodeConfig(verifier=name))
    sampler = RejectionVerifier(DecodeConfig())  # draws the drafts from q in both cases
    selector = TopPSampler(seed=11)
    counts = torch.zeros(2, 10)
    for _ in range(6000):
        drafts, probs = [], []
        for i in range(2):
            token, q = sampler.draft(drafter[i], selector, torch.zeros(0))
            drafts.append(token)
            probs.append(q)
        history = torch.zeros(2, dtype=torch.long)
        accepted, token = verifier.verify(target, drafts, torch.stack(probs), selector, history, 0)
        output = [*drafts[:accepted], token]
        counts[0, output[0]] += 1
        if len(output) > 1:
            counts[1, output[1]] += 1
    p = target.softmax(-1)
    scores = [chi_square_z(counts[i], p[i]) for i in range(2)]
    if name == "rejection":
        assert max(scores) < Z_MAX, scores
    else:
        assert min(scores) > 3 * Z_MAX, scores  # keeping the drafts samples q, not p


@pytest.mark.parametrize("name", ["rejection", "accept_all"])
@torch.no_grad()
def test_speculative_sampling_of_a_batch_is_lossless(name: str) -> None:
    """Many copies of one prompt decoded as a batch: the first drafted-and-verified token follows the model."""
    model = make_tiny_model(seed=2)
    model.model.norm.weight.mul_(4.0)  # a peaked distribution, resolved by fewer samples
    prompt = torch.randint(1, VOCAB, (9,), generator=torch.Generator().manual_seed(6))
    p0 = model(prompt[None]).logits[0, -1].softmax(-1)
    following = torch.cat([prompt.expand(VOCAB, -1), torch.arange(VOCAB)[:, None]], 1)
    p1 = (p0[:, None] * model(following).logits[:, -1].softmax(-1)).sum(0)  # marginal of the second token

    rows = 1500
    script = torch.cat([prompt, torch.full((8,), int(p1.argmax()))])  # drafts lean towards a likely token
    decoder = CycleDecoder(
        LayerRunner(adapter_for(model)), ScriptedDrafter([script] * rows, scale=2.0),
        DecodeConfig(cycle_length=4, verifier=name),
    )
    results = decoder.decode_batch([prompt] * rows, 2, selector=TopPSampler(seed=5))
    first = torch.bincount(torch.tensor([int(r.tokens[0]) for r in results]), minlength=VOCAB)
    second = torch.bincount(torch.tensor([int(r.tokens[1]) for r in results]), minlength=VOCAB)
    assert chi_square_z(first, p0) < Z_MAX  # the prefill token is sampled exactly in both cases
    if name == "rejection":
        assert chi_square_z(second, p1) < Z_MAX
    else:
        assert chi_square_z(second, p1) > 3 * Z_MAX


class FixedDrafter(ScriptedDrafter):
    """Writes its scripted tokens straight into the sequence buffer and reports no distributions (a deterministic
    drafter such as n-gram lookup)."""

    def propose(self, request: DraftRequest) -> Draft:
        width = max(request.budgets, default=0)
        tokens = torch.full((request.batch_size, width), -1, dtype=torch.long)
        for b, row in enumerate(self.rows.tolist()):
            end, n = request.ends[b], request.budgets[b]
            tokens[b, :n] = (self.table[row, end : end + n] + self.offset) % VOCAB
            request.seq[b, end : end + n] = tokens[b, :n]
        return Draft(tokens.to(request.device), list(request.budgets))


@pytest.mark.parametrize("name", ["rejection", "accept_all"])
@torch.no_grad()
def test_speculative_sampling_of_fixed_drafts_is_lossless(name: str) -> None:
    """End to end in fp32: a drafter without distributions, verified by rejection sampling at T=1, samples the
    model's first two tokens exactly."""
    model = make_tiny_model(seed=2)
    model.model.norm.weight.mul_(4.0)
    prompt = torch.randint(1, VOCAB, (9,), generator=torch.Generator().manual_seed(6))
    p0 = model(prompt[None]).logits[0, -1].softmax(-1)
    following = torch.cat([prompt.expand(VOCAB, -1), torch.arange(VOCAB)[:, None]], 1)
    p1 = (p0[:, None] * model(following).logits[:, -1].softmax(-1)).sum(0)

    rows = 1500
    script = torch.cat([prompt, torch.full((8,), int(p1.argmax()))])
    decoder = CycleDecoder(
        LayerRunner(adapter_for(model)), FixedDrafter([script] * rows), DecodeConfig(cycle_length=4, verifier=name)
    )
    results = decoder.decode_batch([prompt] * rows, 2, selector=TopPSampler(seed=5))
    second = torch.bincount(torch.tensor([int(r.tokens[1]) for r in results]), minlength=VOCAB)
    if name == "rejection":
        assert sum(a for r in results for _, a in r.acceptance) > 0.2 * rows  # the drafts are kept often
        assert chi_square_z(second, p1) < Z_MAX
    else:
        assert chi_square_z(second, p1) > 3 * Z_MAX


# ---- typical acceptance ------------------------------------------------------------------------------------------
def test_typical_acceptance_follows_the_entropy() -> None:
    """A confident distribution keeps only likely drafts, a flat one keeps unlikely drafts too."""
    verifier = TypicalVerifier(DecodeConfig(verifier="typical"))
    peaked, flat = [0.9, 0.05, 0.05], [0.1] * 10
    history = torch.zeros(4, dtype=torch.long)
    selector = TopPSampler(seed=0)
    cut = verifier.cutoff(_logits(peaked, flat).softmax(-1))
    assert cut[0] == pytest.approx(0.09) and cut[1] == pytest.approx(0.3 * math.exp(-math.log(10)), rel=1e-3)
    for _ in range(20):  # a rejected draft is followed by a token from the typical set: only token 0 here
        assert verifier.verify(_logits(peaked, flat), [1], None, selector, history, 1) == (0, 0)
    accepted, _ = verifier.verify(_logits(flat, peaked), [7], None, selector, history, 1)
    assert accepted == 1
    loose = TypicalVerifier(DecodeConfig(verifier="typical", verifier_options={"threshold": 0.01}))
    assert loose.verify(_logits(peaked, peaked), [1], None, selector, history, 1)[0] == 1


def test_typical_verify_batch_equals_verify_per_row() -> None:
    generator = torch.Generator().manual_seed(4)
    logits = torch.randn(64, 4, 12, generator=generator) * 2.0
    tokens = torch.randint(0, 12, (64, 3), generator=generator)
    lengths = torch.randint(0, 4, (64,), generator=generator).tolist()
    for b, k in enumerate(lengths):
        tokens[b, k:] = -1
    verifier = TypicalVerifier(DecodeConfig(verifier="typical"))
    seq = torch.zeros(64, 10, dtype=torch.long)
    for selector in (GreedySelector(), TopPSampler(seed=1)):
        accepted, following = verifier.verify_batch(logits, Draft(tokens, lengths), selector, seq, [5] * 64)
        expected = [verifier.verify(logits[b, : k + 1], tokens[b, :k].tolist(), None, selector, seq[b], 5)
                    for b, k in enumerate(lengths)]
        assert accepted.tolist() == [a for a, _ in expected]
        assert 0 < sum(accepted.tolist()) < sum(lengths)  # some drafts are kept, some are not
        if selector.greedy:
            assert following.tolist() == [t for _, t in expected]


@pytest.mark.parametrize("tree", [False, True], ids=["chain", "tree"])
def test_typical_acceptance_with_a_greedy_selector_is_greedy(tree: bool) -> None:
    model = make_tiny_model(seed=1)
    generator = torch.Generator().manual_seed(2)
    prompts = [torch.randint(1, VOCAB, (n,), generator=generator) for n in (8, 5, 11)]
    greedy = [reference_greedy(model, p, 20) for p in prompts]
    scripts = [torch.cat([p, g]) for p, g in zip(prompts, greedy)]
    for script, prompt in zip(scripts, prompts):
        script[prompt.numel() + 3 :: 4] = (script[prompt.numel() + 3 :: 4] + 1) % VOCAB  # some wrong drafts
    drafter = ScriptedTreeDrafter(scripts) if tree else ScriptedDrafter(scripts)
    decoder = CycleDecoder(LayerRunner(adapter_for(model)), drafter, DecodeConfig(cycle_length=4, verifier="typical"))
    results = decoder.decode_batch(prompts, 14)
    assert [r.tokens.tolist() for r in results] == [g[:14].tolist() for g in greedy]
    assert all(r.mean_accepted > 0.5 for r in results)


def test_typical_sampling_decodes_a_tree() -> None:
    model = make_tiny_model(seed=1)
    prompt = torch.randint(1, VOCAB, (8,), generator=torch.Generator().manual_seed(2))
    script = torch.cat([prompt, reference_greedy(model, prompt, 30)])
    decoder = CycleDecoder(LayerRunner(adapter_for(model)), ScriptedTreeDrafter([script, script]),
                           DecodeConfig(cycle_length=4, verifier="typical"))
    results = decoder.decode_batch([prompt, prompt], 24, selector=TopPSampler(seed=3))
    assert all(r.tokens.numel() == 24 for r in results)
    assert all(r.mean_accepted > 0 for r in results)


def test_verifier_options_are_checked() -> None:
    with pytest.raises(ConfigError, match="unknown verifier_options"):
        TypicalVerifier(DecodeConfig(verifier="typical", verifier_options={"posterior_threshold": 0.1}))
    with pytest.raises(ConfigError, match="unknown verifier_options"):
        GreedyVerifier(DecodeConfig(verifier_options={"threshold": 0.1}))
    verifier = TypicalVerifier(DecodeConfig(verifier="typical", verifier_options={"alpha": 0.5}))
    assert (verifier.threshold, verifier.alpha) == (0.09, 0.5)


# ---- configuration -------------------------------------------------------------------------------------------------
def test_lossless_contradicting_the_verifier_is_refused() -> None:
    model = make_tiny_model(seed=1)
    runner = LayerRunner(adapter_for(model))
    drafter = ScriptedDrafter([torch.arange(1, 20)])
    with pytest.raises(ConfigError, match="lossless=True contradicts"):
        CycleDecoder(runner, drafter, DecodeConfig(lossless=True, verifier="accept_all"))
    assert CycleDecoder(runner, drafter, DecodeConfig(verifier="accept_all")).verifier.commits_drafts
    assert DecodeConfig().verifier_name == "greedy"
    assert DecodeConfig(lossless=False).verifier_name == "accept_all"
