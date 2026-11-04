"""Speculative sampling on chains and draft trees samples exactly from the model (fp32, tiny model, CPU).

The method of ``tests/sample_fp32.py`` in the research code. Many copies of one prompt are decoded as one batch
with a seeded sampler, and the first ``G`` tokens of every row are checked twice:

1. Randomized probability integral transform. With the model's exact conditional ``p`` of a token ``x`` (tokens
   ordered by probability, ties by id), ``u = F(x-) + U p(x)`` with ``U ~ Uniform(0, 1)`` is i.i.d. uniform when
   the sampler is exact (Rosenblatt). A Kolmogorov-Smirnov test over all rows and positions must not reject.
2. The total variation distance between the joint distribution of the first two (and three) tokens and that of
   plain autoregressive samples stays below the 99.9% quantile of the same distance over random splits of the
   pooled samples (a permutation test).

The drafts come from a perturbed copy of the model, so they agree with the model often but not always. A drafter
that samples its tree but hides the distributions (the verifier then treats the drafts as deterministic) must fail
the same checks, which shows that they have power. At temperature ``1e-6`` decoding equals greedy decoding.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from economize import DecodeConfig
from economize.arch import adapter_for
from economize.decoding import CycleDecoder, Draft, Drafter, DraftRequest, TopPSampler
from economize.runtime import LayerRunner
from helpers import VOCAB, make_tiny_model, perturb, reference_greedy

ALPHA = 1e-3  # level of every test; each case runs a handful of tests, so a false alarm is rare


# ---- statistics ----------------------------------------------------------------------------------------------------
def ks_p(u: np.ndarray) -> float:
    """Asymptotic Kolmogorov-Smirnov p-value of ``u`` against Uniform(0, 1)."""
    u = np.sort(u)
    n = len(u)
    i = np.arange(1, n + 1)
    d = max((i / n - u).max(), (u - (i - 1) / n).max())
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    return min(1.0, max(0.0, 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))))


@torch.no_grad()
def pit(model: torch.nn.Module, sampler: TopPSampler, prompt: torch.Tensor, samples: torch.Tensor) -> np.ndarray:
    """Randomized PIT of every token of ``samples`` ``[n, G]`` under the exact conditionals of ``sampler``."""
    ids = torch.cat([prompt.expand(len(samples), -1), samples[:, :-1]], 1)
    logits = model(ids).logits[:, prompt.numel() - 1 :]
    p = sampler.distributions(logits).double()
    px = p.gather(-1, samples[..., None])
    vocab = torch.arange(p.shape[-1])
    before = (p * ((p > px) | ((p == px) & (vocab < samples[..., None])))).sum(-1)
    noise = torch.rand(px.shape[:-1], dtype=torch.float64, generator=torch.Generator().manual_seed(99))
    return (before + noise * px[..., 0]).flatten().numpy()


def tv_bound(a: np.ndarray, b: np.ndarray, permutations: int = 1000) -> tuple[float, float]:
    """TV distance between the empirical distributions of keys ``a`` and ``b`` (same size) and its 99.9% quantile
    over random splits of the pooled keys."""
    _, index = np.unique(np.concatenate([a, b]), return_inverse=True)
    m, n = index.max() + 1, len(a)

    def tv(i: np.ndarray) -> float:
        return 0.5 * np.abs(np.bincount(i[:n], minlength=m) - np.bincount(i[n:], minlength=m)).sum() / n

    rng = np.random.default_rng(0)
    null = [tv(index[rng.permutation(2 * n)]) for _ in range(permutations)]
    return tv(index), float(np.quantile(null, 0.999))


def joint_keys(samples: torch.Tensor, k: int) -> np.ndarray:
    return sum(samples[:, i].numpy().astype(np.int64) * VOCAB ** (k - 1 - i) for i in range(k))


# ---- a drafter that samples chains and trees from a second model ----------------------------------------------------
class SampledDrafter(Drafter):
    """Drafts from a draft model by full recomputation (no cache).

    ``branching[level]`` children hang under every node of the previous level. ``mode`` chooses how siblings are
    drawn from the draft model's distribution ``q``: ``"iid"`` (with replacement, every child from ``q``),
    ``"without"`` (without replacement, child ``i`` from ``q`` renormalized without the earlier siblings) or
    ``"topk"`` (the most likely tokens, deterministic, no distributions). ``chain=True`` drafts one sampled chain
    (``branching`` ignored). ``report`` chooses the distributions the draft reports: ``"exact"`` (the one every
    child was drawn from; none for ``"topk"``), ``"none"`` (the verifier then treats the drafts as fixed, which is
    also exact) or ``"unconditional"`` (``q`` for every child, wrong unless the children are i.i.d. samples).
    """

    source_layers = (0,)
    layer_span = (1, 1)

    def __init__(
        self,
        model: torch.nn.Module,
        sampler: TopPSampler,
        branching: tuple[int, ...] = (1,),
        mode: str = "iid",
        chain: bool = False,
        report: str = "exact",
        seed: int = 0,
    ) -> None:
        self.model, self.sampler = model, sampler
        self.branching, self.mode, self.chain, self.report = branching, mode, chain, report
        self.generator = torch.Generator().manual_seed(seed)

    def begin(self, runner, cache, batch_size: int) -> None:
        pass

    def observe(self, exact, columns: torch.Tensor) -> None:
        pass

    def reorder(self, rows: torch.Tensor) -> None:
        pass

    def width(self, level: int) -> int:
        """Children per node at depth ``level``."""
        return 1 if self.chain else self.branching[min(level, len(self.branching) - 1)]

    def draft_capacity(self, cycle_length: int) -> int:
        return sum(math.prod(self.width(i) for i in range(level + 1)) for level in range(cycle_length - 1))

    @torch.no_grad()
    def _q(self, prefixes: list[list[int]]) -> torch.Tensor:
        """The draft distribution after every prefix, ``[n, V]`` (prefixes grouped by length)."""
        out = torch.empty(len(prefixes), VOCAB)
        groups: dict[int, list[int]] = {}
        for i, prefix in enumerate(prefixes):
            groups.setdefault(len(prefix), []).append(i)
        for members in groups.values():
            logits = self.model(torch.tensor([prefixes[i] for i in members])).logits[:, -1]
            out[members] = self.sampler.distributions(logits)
        return out

    def propose(self, request: DraftRequest) -> Draft:
        rows = range(request.batch_size)
        history = [request.seq[b, : request.ends[b]].tolist() for b in rows]
        tokens: list[list[int]] = [[] for _ in rows]
        parents: list[list[int]] = [[] for _ in rows]
        probs: list[list[torch.Tensor]] = [[] for _ in rows]
        frontier = [[(-1, history[b])] for b in rows]  # (node, prefix) to expand
        depth = max(request.budgets)
        for level in range(depth):
            work = [(b, node, prefix) for b in rows if level < request.budgets[b] for node, prefix in frontier[b]]
            if not work:
                break
            q_all = self._q([prefix for _, _, prefix in work])
            frontier = [[] for _ in rows]
            width = self.width(level)
            for (b, node, prefix), q in zip(work, q_all):
                for x, q_i in self._children(q, width):
                    tokens[b].append(x)
                    parents[b].append(node)
                    probs[b].append(q if self.report == "unconditional" else q_i)
                    frontier[b].append((len(tokens[b]) - 1, [*prefix, x]))
                request.costs[b].record("draft", width, len(prefix) - 1, *self.layer_span)
        lengths = [len(t) for t in tokens]
        width = max(lengths, default=0)
        token_table = torch.full((len(rows), width), -1, dtype=torch.long)
        parent_table = torch.full((len(rows), width), -1, dtype=torch.long)
        prob_table = torch.zeros(len(rows), width, VOCAB)
        for b in rows:
            n = lengths[b]
            token_table[b, :n] = torch.tensor(tokens[b], dtype=torch.long)
            parent_table[b, :n] = torch.tensor(parents[b], dtype=torch.long)
            if n:
                prob_table[b, :n] = torch.stack(probs[b])
            if self.chain:
                request.seq[b, request.ends[b] : request.ends[b] + n] = token_table[b, :n]
        hidden = self.report == "none" or (self.mode == "topk" and self.report == "exact")
        return Draft(token_table, lengths, None if self.chain else parent_table, None if hidden else prob_table)

    def _children(self, q: torch.Tensor, width: int) -> list[tuple[int, torch.Tensor]]:
        if self.mode == "topk":
            return [(int(x), q) for x in q.topk(width).indices]
        if self.mode == "iid":
            drawn = torch.multinomial(q, width, replacement=True, generator=self.generator)
            return [(int(x), q) for x in drawn]
        out, rest = [], q.clone()
        for _ in range(width):
            if rest.sum() <= 0:  # the support is exhausted (nucleus sampling)
                break
            q_i = rest / rest.sum()
            x = int(torch.multinomial(q_i, 1, generator=self.generator))
            out.append((x, q_i))
            rest[x] = 0.0
        return out


# ---- fixtures ------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def models() -> tuple[torch.nn.Module, torch.nn.Module, torch.Tensor]:
    torch.backends.cuda.matmul.allow_tf32 = False
    model = make_tiny_model(seed=2)
    with torch.no_grad():
        model.model.norm.weight.mul_(3.0)  # peaked distributions, so a few thousand samples resolve them
    draft = copy.deepcopy(model)
    perturb(draft, std=0.01, seed=1)
    prompt = torch.randint(1, VOCAB, (9,), generator=torch.Generator().manual_seed(6))
    return model, draft, prompt


ROWS, G = 3000, 3


def _decode(model, drafter, prompt, cycle_length, sampler, verifier="rejection"):
    config = DecodeConfig(cycle_length=cycle_length, verifier=verifier) if drafter else DecodeConfig(cycle_length=1)
    decoder = CycleDecoder(LayerRunner(adapter_for(model)), drafter, config)
    results = decoder.decode_batch([prompt] * ROWS, G, selector=sampler)
    assert all(r.tokens.numel() == G for r in results)
    return torch.stack([r.tokens for r in results]), results


@pytest.fixture(scope="module")
def reference(models) -> dict[tuple[float, float], torch.Tensor]:
    """Plain autoregressive samples for every (temperature, top_p) the tests use."""
    model, _, prompt = models
    return {key: _decode(model, None, prompt, 1, TopPSampler(*key, seed=123))[0] for key in [(1.0, 1.0), (0.8, 0.9)]}


def _check(models, reference, samples, sampler_key) -> tuple[float, list[tuple[float, float]]]:
    model, _, prompt = models
    p_value = ks_p(pit(model, TopPSampler(*sampler_key), prompt, samples))
    tvs = [tv_bound(joint_keys(samples, k), joint_keys(reference[sampler_key], k)) for k in (2, 3)]
    return p_value, tvs


# ---- exactness -----------------------------------------------------------------------------------------------------
CASES = {
    "chain": dict(chain=True),
    "tree iid": dict(branching=(3, 2), mode="iid"),
    "tree without replacement": dict(branching=(3, 2), mode="without"),
    "tree iid, distributions hidden": dict(branching=(3, 2), mode="iid", report="none"),
    "tree top-k (deterministic)": dict(branching=(2, 2), mode="topk"),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("sampler_key", [(1.0, 1.0), (0.8, 0.9)], ids=["T1", "T0.8-p0.9"])
def test_speculative_sampling_is_exact(models, reference, case: str, sampler_key) -> None:
    model, draft_model, prompt = models
    sampler = TopPSampler(*sampler_key, seed=7)
    drafter = SampledDrafter(draft_model, TopPSampler(*sampler_key), seed=8, **CASES[case])
    samples, results = _decode(model, drafter, prompt, 3, sampler)
    accepted = sum(a for r in results for _, a in r.acceptance) / sum(len(r.acceptance) for r in results)
    assert accepted > 0.3, accepted  # the drafts agree with the model often enough to test acceptance
    p_value, tvs = _check(models, reference, samples, sampler_key)
    assert p_value > ALPHA, (p_value, tvs)
    for tv, bound in tvs:
        assert tv <= bound, (p_value, tvs)


def test_misreported_draft_distributions_are_detected(models, reference) -> None:
    """Top-k drafts reported as samples from ``q`` are over-accepted: both checks reject, so they have power."""
    model, draft_model, prompt = models
    drafter = SampledDrafter(draft_model, TopPSampler(), branching=(2, 2), mode="topk", report="unconditional")
    samples, _ = _decode(model, drafter, prompt, 3, TopPSampler(seed=7))
    p_value, tvs = _check(models, reference, samples, (1.0, 1.0))
    assert p_value < ALPHA and all(tv > bound for tv, bound in tvs), (p_value, tvs)


@pytest.mark.parametrize("case", list(CASES))
def test_near_zero_temperature_is_greedy(models, case: str) -> None:
    model, draft_model, _ = models
    generator = torch.Generator().manual_seed(3)
    prompts = [torch.randint(1, VOCAB, (n,), generator=generator) for n in (7, 12, 9)]
    drafter = SampledDrafter(draft_model, TopPSampler(1e-6), seed=1, **CASES[case])
    decoder = CycleDecoder(LayerRunner(adapter_for(model)), drafter, DecodeConfig(cycle_length=4, verifier="rejection"))
    results = decoder.decode_batch(prompts, 24, selector=TopPSampler(1e-6, seed=0))
    for prompt, result in zip(prompts, results):
        assert torch.equal(result.tokens, reference_greedy(model, prompt, 24))
