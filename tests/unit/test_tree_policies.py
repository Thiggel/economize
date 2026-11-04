"""The Sequoia and SpecExec draft trees (``DecodeConfig(draft_policy="sequoia" | "specexec")``).

Both policies wrap any :class:`economize.decoding.TreeStepDrafter` (EAGLE-3, jump drafting, the self-drafters, FR-Spec
around any of them), like ``dynamic_tree``. Greedy tree verification is lossless, so in float32 the outputs equal
plain greedy decoding for every drafter and every tree shape. The tree constructions are checked against independent
references: Sequoia's dynamic program against a brute-force search over all trees and a Monte Carlo estimate, the
filling of a static tree and SpecExec's best-first search against pure-Python walks over a drafter whose logits
depend only on the last token. The CUDA graph and kernel tests need ``-m gpu``.
"""

from __future__ import annotations

import heapq
import itertools
import math
from typing import Any

import numpy as np
import pytest
import torch

import economize
from economize import ConfigError, DecodeConfig, JumpConfig
from economize.compat.tskip import cost_events
from economize.decoding import (
    POLICIES,
    AcceptanceProfile,
    CostKind,
    SearchOptions,
    SequoiaDrafter,
    SpecExecDrafter,
    StaticTree,
    measure_acceptance,
    sequoia_tree,
)
from economize.decoding.cost import LayerFootprint
from economize.decoding.sequoia import expected_tokens, probe_paths
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.early_exit import EarlyExitConfig
from economize.techniques.frspec import FRSpecConfig
from economize.techniques.vendor_mtp import VendorMTPConfig
from helpers import (
    FAMILY_MODELS,
    VOCAB,
    family_skip_reason,
    make_family_model,
    make_identity_jump,
    make_tiny_model,
    perturb,
    reference_greedy,
)

CUDA = torch.cuda.is_available()
MAX_NEW = 24
ACCEPTANCE = (0.6, 0.2, 0.1)
SEQUOIA = {"acceptance": ACCEPTANCE, "total_tokens": 10, "depth": 4}
SPECEXEC = {"depth": 4, "width": 2, "topk": 3, "total_tokens": 10}
OPTIONS = {"sequoia": SEQUOIA, "specexec": SPECEXEC}
DRAFTER_CLASS = {"sequoia": SequoiaDrafter, "specexec": SpecExecDrafter}


def _ids(size: int, seed: int = 5) -> list[int]:
    return torch.randperm(VOCAB, generator=torch.Generator().manual_seed(seed))[:size].tolist()


DRAFTERS = {
    "jump": lambda: JumpConfig(target_layer=2, source_layer=4),
    "jump_lora": lambda: JumpConfig(target_layer=2, source_layer=4, adaptation="lora", lora_rank=4),
    "jump_exit": lambda: JumpConfig(target_layer=1, source_layer=3, exit_layer=4, adaptation="full"),
    "eagle3": lambda: Eagle3Config(),
    "eagle3_vocab": lambda: Eagle3Config(draft_vocab=96),
    "early_exit": lambda: EarlyExitConfig(exit_layer=3),
    "early_exit_full": lambda: EarlyExitConfig(exit_layer=3, adaptation="full"),
    "frspec_jump": lambda: FRSpecConfig(drafter=JumpConfig(target_layer=2, source_layer=4), vocab_size=64,
                                        token_ids=tuple(_ids(64))),
}


def _build(name: str, device: str = "cpu", std: float = 0.05) -> economize.EfficientModel:
    config = DRAFTERS[name]()
    model = economize.apply(make_tiny_model().to(device), config)
    for technique in model.techniques.values():
        if isinstance(config, Eagle3Config) and config.draft_vocab is not None:
            technique.set_draft_vocab(torch.tensor(_ids(config.draft_vocab, seed=3), device=device))
        perturb(technique, std, seed=1)
    return model


def _prompts(device: str = "cpu", lengths: tuple[int, ...] = (13, 9, 11)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    return [torch.randint(3, VOCAB, (n,), generator=generator).to(device) for n in lengths]


def _tree(policy: str, cycle_length: int = 5, **config: Any) -> DecodeConfig:
    config.setdefault("draft_policy_options", OPTIONS[policy])
    return DecodeConfig(cycle_length=cycle_length, draft_policy=policy, **config)


def _accepted(results: list[Any]) -> float:
    cycles = [a for r in results for _, a in r.acceptance]
    return sum(cycles) / len(cycles)


# ---- losslessness ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("policy", sorted(OPTIONS))
@pytest.mark.parametrize("name", sorted(DRAFTERS))
@torch.no_grad()
def test_the_tree_policies_are_lossless_in_float32(name: str, policy: str) -> None:
    model = _build(name)
    prompts = _prompts()
    greedy = [reference_greedy(model.base_model, p, MAX_NEW).tolist() for p in prompts]
    results = model.decode_batch(prompts, MAX_NEW, decoding=_tree(policy))
    assert [r.tokens.tolist() for r in results] == greedy
    single = model.decode(prompts[1], MAX_NEW, decoding=_tree(policy))
    assert single.tokens.tolist() == greedy[1]
    assert isinstance(model.decoder(_tree(policy)).active_drafter, DRAFTER_CLASS[policy])


@pytest.mark.parametrize("options", [
    {"acceptance": (0.9,), "total_tokens": 1, "depth": 1},
    {"acceptance": (0.3, 0.3, 0.2, 0.1), "total_tokens": 20, "depth": 2},
    {"acceptance": ((0.5, 0.3), (0.8, 0.1), (0.2, 0.1)), "total_tokens": 12, "depth": 6},
    {"paths": ((0,), (2,), (0, 0), (0, 3), (2, 1), (0, 0, 0), (0, 3, 1))},  # ranks with gaps
    {"paths": [[0], [0, 0], [0, 0, 0]]},  # a chain, given as lists (as JSON gives it)
], ids=["single", "wide", "per_depth", "gaps", "chain"])
@torch.no_grad()
def test_sequoia_is_lossless_for_any_tree(options: dict[str, Any]) -> None:
    model = _build("eagle3")
    prompts = _prompts()
    greedy = [reference_greedy(model.base_model, p, MAX_NEW).tolist() for p in prompts]
    decoding = DecodeConfig(cycle_length=7, draft_policy="sequoia", draft_policy_options=options)
    assert [r.tokens.tolist() for r in model.decode_batch(prompts, MAX_NEW, decoding=decoding)] == greedy


@pytest.mark.parametrize("options", [dict(depth=1, width=1, topk=1, total_tokens=1),
                                     dict(depth=2, width=5, topk=5, total_tokens=30),
                                     dict(depth=6, width=1, topk=2, total_tokens=11),
                                     dict(depth=5, width=4, topk=6, total_tokens=40)])
@torch.no_grad()
def test_specexec_is_lossless_for_any_budget(options: dict[str, int]) -> None:
    model = _build("eagle3")
    prompts = _prompts()
    greedy = [reference_greedy(model.base_model, p, MAX_NEW).tolist() for p in prompts]
    decoding = DecodeConfig(cycle_length=options["depth"] + 1, draft_policy="specexec", draft_policy_options=options)
    assert [r.tokens.tolist() for r in model.decode_batch(prompts, MAX_NEW, decoding=decoding)] == greedy


@pytest.mark.parametrize("policy", sorted(OPTIONS))
@torch.no_grad()
def test_an_exact_drafter_accepts_the_whole_depth(policy: str) -> None:
    """A drafter that computes exactly what the model computes ranks the greedy token first, and both trees hold the
    rank-0 path to the full depth, so every cycle accepts the whole depth."""
    base = make_tiny_model()
    base.lm_head.weight.data *= 30
    model = make_identity_jump(base)
    prompt = _prompts()[0]
    result = model.decode(prompt, MAX_NEW, decoding=_tree(policy))
    assert result.tokens.tolist() == reference_greedy(model.base_model, prompt, MAX_NEW).tolist()
    accepted = [a for _, a in result.acceptance]
    assert accepted[:-1] == [4] * (len(accepted) - 1) and accepted


@pytest.mark.parametrize("policy", sorted(OPTIONS))
@pytest.mark.parametrize("name", ["jump", "jump_exit", "eagle3", "early_exit", "frspec_jump"])
@torch.no_grad()
def test_a_chain_tree_drafts_as_the_chain(name: str, policy: str) -> None:
    """A Sequoia tree of rank-0 nodes and a SpecExec search of width and topk 1 are the chain, so they draft and
    accept exactly what the chain drafter does."""
    model = _build(name, std=0.02)
    prompts = _prompts()
    chain = model.decode_batch(prompts, MAX_NEW, decoding=DecodeConfig(cycle_length=5))
    options = ({"paths": ((0,), (0, 0), (0, 0, 0), (0, 0, 0, 0))} if policy == "sequoia"
               else {"depth": 4, "width": 1, "topk": 1, "total_tokens": 4})
    tree = model.decode_batch(prompts, MAX_NEW, decoding=_tree(policy, draft_policy_options=options))
    assert [r.acceptance for r in tree] == [r.acceptance for r in chain]
    assert any(a for r in chain for _, a in r.acceptance)


@pytest.mark.parametrize("policy", sorted(OPTIONS))
@torch.no_grad()
def test_the_trees_accept_at_least_as_much_as_the_chain(policy: str) -> None:
    model = _build("eagle3", std=0.2)
    prompts = _prompts()
    chain = model.decode_batch(prompts, MAX_NEW, decoding=DecodeConfig(cycle_length=5))
    options = ({"acceptance": (0.4, 0.2, 0.1, 0.05, 0.05), "total_tokens": 30, "depth": 4} if policy == "sequoia"
               else {"depth": 4, "width": 6, "topk": 6, "total_tokens": 30})
    tree = model.decode_batch(prompts, MAX_NEW, decoding=_tree(policy, draft_policy_options=options))
    assert _accepted(tree) >= _accepted(chain) > 0


@pytest.mark.parametrize("family", sorted(FAMILY_MODELS))
@torch.no_grad()
def test_the_paper_recipe_is_lossless_on_every_family(family: str) -> None:
    """Jump + exit with a trained window copy, as at Gemma-4-31B, under both tree policies on every family."""
    reason = family_skip_reason(family)
    if reason is not None:
        pytest.skip(reason)
    base = make_family_model(family)
    recurrent = economize.runtime.LayerRunner(economize.arch.adapter_for(base)).has_recurrent_layers
    prompts = [torch.tensor(_ids(n, seed=n)) for n in (13, 9, 11)]
    greedy = [reference_greedy(base, p, MAX_NEW).tolist() for p in prompts]
    model = economize.apply(base, JumpConfig(target_layer=1, source_layer=2, exit_layer=2, adaptation="full"))
    try:
        for technique in model.techniques.values():
            perturb(technique, 0.05, seed=1)
        for policy in OPTIONS:
            if recurrent:
                with pytest.raises((ConfigError, economize.errors.IncompatibleTechniquesError)):
                    model.decode_batch(prompts, MAX_NEW, decoding=_tree(policy))
                continue
            results = model.decode_batch(prompts, MAX_NEW, decoding=_tree(policy))
            assert [r.tokens.tolist() for r in results] == greedy, policy
    finally:
        model.detach()


# ---- cost accounting ------------------------------------------------------------------------------------------------
@torch.no_grad()
def test_sequoia_logs_one_draft_call_per_depth_with_children() -> None:
    model = _build("jump")
    result = model.decode(_prompts()[0], MAX_NEW, decoding=_tree("sequoia"))
    tree = sequoia_tree(ACCEPTANCE, 10, 4)
    widths = [0] * (tree.depth + 1)
    for i, children in enumerate(tree.children()[:-1]):
        widths[len(tree.paths[i])] += bool(children)
    drafts = [e for e in result.cost.events if e.kind == CostKind.DRAFT]
    verifies = [e for e in result.cost.events if e.kind == CostKind.EXACT][1:]
    assert {(e.step, e.tokens) for e in drafts} <= {(0, 1)} | {(d, w) for d, w in enumerate(widths) if w}
    assert {e.tokens for e in verifies} == {tree.size + 1}
    assert sum(e.tokens for e in drafts) == (1 + tree.forwarded) * len(verifies)


@torch.no_grad()
def test_specexec_logs_one_draft_call_of_width_per_search_step() -> None:
    model = _build("jump")
    result = model.decode(_prompts()[0], MAX_NEW, decoding=_tree("specexec"))
    drafts = [e for e in result.cost.events if e.kind == CostKind.DRAFT]
    verifies = [e for e in result.cost.events if e.kind == CostKind.EXACT][1:]
    assert {(e.step, e.tokens) for e in drafts} == {(0, 1)} | {(s, SPECEXEC["width"]) for s in (1, 2, 3)}
    assert {e.tokens for e in verifies} == {SPECEXEC["total_tokens"] + 1}
    assert sum(e.tokens for e in drafts) == (1 + 3 * SPECEXEC["width"]) * len(verifies)


# ---- Sequoia's tree construction ------------------------------------------------------------------------------------
def _brute_force(p: tuple[float, ...], size: int, depth: int) -> float:
    """The best expected tokens over all trees of at most ``size`` nodes and depth ``depth`` whose sibling ranks are
    contiguous from 0 (an optimal tree has this form: a lower rank is at least as likely)."""
    paths = [q for level in range(1, depth + 1) for q in itertools.product(range(len(p)), repeat=level)]
    best, frontier, seen = 1.0, [frozenset()], {frozenset()}
    while frontier:
        grown = []
        for tree in frontier:
            if tree:
                best = max(best, expected_tokens(StaticTree.from_paths(sorted(tree)), p))
            if len(tree) == size:
                continue
            for q in paths:
                closed = (len(q) == 1 or q[:-1] in tree) and (q[-1] == 0 or (*q[:-1], q[-1] - 1) in tree)
                if q not in tree and closed and (bigger := tree | {q}) not in seen:
                    seen.add(bigger)
                    grown.append(bigger)
        frontier = grown
    return best


@pytest.mark.parametrize("p", [(0.6, 0.2, 0.1), (0.3, 0.3, 0.3), (0.9, 0.05, 0.0)])
def test_the_dynamic_program_finds_the_best_tree(p: tuple[float, ...]) -> None:
    for size, depth in itertools.product(range(1, 7), range(1, 5)):
        tree = sequoia_tree(p, size, depth)
        assert tree.size <= size and tree.depth <= depth
        assert tree.expected == pytest.approx(expected_tokens(tree, p), abs=1e-12)
        assert tree.expected == pytest.approx(_brute_force(p, size, depth), abs=1e-12), (size, depth)


# The acceptance-rate vector shipped with the official code (Infini-AI-Lab/Sequoia, acceptance-rate-vector.pt at
# 0ff3ff7, entries 1..32 rounded to 6 digits)
OFFICIAL_ACCEPTANCE = (0.634183, 0.107946, 0.056972, 0.022489, 0.01949, 0.014993, 0.004498, 0.002999, 0.011994,
                       0.004498, 0.007496, 0.004498, 0.005997, 0.002999, 0.001499, 0.002999, 0.001499, 0.002999, 0.0,
                       0.002999, 0.002999, 0.002999, 0.0, 0.001499, 0.0, 0.001499, 0.0, 0.0, 0.0, 0.001499, 0.001499,
                       0.001499)


def _official_dp(p: tuple[float, ...], budget: int, depth: int) -> list[list[float]]:
    """The dynamic program of the official ``tree_search.py`` (its loop, on floats): entry ``[m][l]`` is the best
    expected tokens of a tree of exactly ``m`` nodes (the root included) over ``l`` levels (the root's included)."""
    p = (0.0, *p)  # the official vector is indexed by rank + 1
    branches = len(p) - 1
    inf = float("-inf")
    t = [[[inf] * (branches + 1) for _ in range(depth + 1)] for _ in range(budget + 1)]
    for level in range(1, depth + 1):
        t[1][level][0] = 1.0
    for m in range(2, budget + 1):
        for level in range(2, depth + 1):
            t[m][level][1] = 1 + p[1] * max(t[m - 1][level - 1])
            for b in range(2, branches + 1):
                best = inf
                for y in range(1, m):  # as there, a NaN (0 * -inf of a rate 0) never wins
                    best = max(best, t[y][level][b - 1] + p[b] * max(t[m - y][level - 1]))
                t[m][level][b] = best
    return [[max(t[m][level]) for level in range(depth + 1)] for m in range(budget + 1)]


def test_the_dynamic_program_matches_the_official_code() -> None:
    p = OFFICIAL_ACCEPTANCE
    official = _official_dp(p, 33, 7)
    for m, level in itertools.product(range(2, 34), range(2, 8)):
        best = max(official[size][level] for size in range(1, m + 1))  # ours takes at most m - 1 draft nodes
        assert sequoia_tree(p, m - 1, level - 1).expected == pytest.approx(best, abs=1e-12), (m, level)


def test_the_expected_tokens_match_a_simulation() -> None:
    """Accepted tokens of a cycle simulated with the exact model's rank drawn from ``p`` after every accepted node."""
    p = (0.5, 0.25, 0.1)
    tree = sequoia_tree(p, 12, 5)
    nodes = {path: i for i, path in enumerate(tree.paths)}
    rng = np.random.default_rng(0)
    trials, total = 40_000, 0
    for _ in range(trials):
        path: tuple[int, ...] = ()
        total += 1
        while True:
            rank = rng.choice(len(p) + 1, p=[*p, 1 - sum(p)])
            if rank == len(p) or (*path, rank) not in nodes:
                break
            path += (rank,)
            total += 1
    assert total / trials == pytest.approx(tree.expected, abs=0.02)


def test_the_hardware_aware_choice_maximizes_tokens_per_second() -> None:
    p = (0.7, 0.15, 0.05)
    verify = [0.010 + 0.002 * max(i - 8, 0) for i in range(33)]  # flat up to 9 tokens, then linear
    tree = sequoia_tree(p, 32, 8, verify_latency=verify, draft_latency=0.001)
    rates = {(d, n): sequoia_tree(p, n, d).expected / (verify[n] + d * 0.001)
             for d in range(1, 9) for n in range(1, 33)}
    assert tree.expected / (verify[tree.size] + tree.depth * 0.001) == pytest.approx(max(rates.values()))
    assert tree.size < 32
    flat = sequoia_tree(p, 32, 8, verify_latency=[0.01] * 33)
    assert flat.expected == pytest.approx(sequoia_tree(p, 32, 8).expected)


def test_static_trees_and_profiles() -> None:
    tree = StaticTree.from_paths([(1,), (0,), (0, 0), (1, 2)])
    assert tree.paths == ((0,), (1,), (0, 0), (1, 2)) and tree.parents == (-1, -1, 0, 1)
    assert (tree.size, tree.depth, tree.forwarded, tree.ranks) == (4, 2, 2, (0, 1, 0, 2))
    for bad in ([], [(0, 0)], [(-1,)]):
        with pytest.raises(ConfigError):
            StaticTree.from_paths(bad)
    assert probe_paths(2, 3) == ((0,), (1,), (2,), (0, 0), (0, 1), (0, 2))
    profile = AcceptanceProfile()
    profile.add(0, 3, 0)
    profile.add(1, 3, 2)
    profile.add(0, 3, None)
    profile.add(2, 0, None)  # a leaf: no trial
    assert profile.rates() == pytest.approx((1 / 3, 0.0, 1 / 3))
    assert profile.rates(per_depth=True) == (pytest.approx((0.5, 0.0, 0.0)), pytest.approx((0.0, 0.0, 1.0)))


@torch.no_grad()
def test_measure_acceptance_feeds_the_policy() -> None:
    base = make_tiny_model()
    base.lm_head.weight.data *= 30
    exact = measure_acceptance(make_identity_jump(base), _prompts(), MAX_NEW, depth=3, width=4)
    assert exact[0] == pytest.approx(1.0) and sum(exact[1:]) == 0.0
    model = _build("eagle3", std=0.2)
    rates = measure_acceptance(model, _prompts(), MAX_NEW, depth=3, width=4)
    per_depth = measure_acceptance(model, _prompts(), MAX_NEW, depth=3, width=4, per_depth=True)
    assert len(rates) == 4 and all(0.0 <= r <= 1.0 for r in rates) and sum(rates) <= 1.0 and rates[0] > 0
    assert all(len(row) == 4 for row in per_depth)
    prompts = _prompts()
    greedy = [reference_greedy(model.base_model, p, MAX_NEW).tolist() for p in prompts]
    decoding = _tree("sequoia", draft_policy_options={"acceptance": rates, "total_tokens": 16})
    assert [r.tokens.tolist() for r in model.decode_batch(prompts, MAX_NEW, decoding=decoding)] == greedy


# ---- the trees a proposal builds, against pure-Python references ----------------------------------------------------
class _TableDrafter:
    """A stateless drafter whose logits after a token are a fixed random row of a table."""

    source_layers = (0,)
    layer_span = (0, 0)
    footprint = LayerFootprint(0, 0)

    def __init__(self, vocab: int = 16, seed: int = 0) -> None:
        self.table = torch.randn(vocab, vocab, generator=torch.Generator().manual_seed(seed)) * 2

    def tree_begin(self, request: Any) -> None:
        pass

    def tree_state(self) -> tuple[torch.Tensor, ...]:
        return ()

    def step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: Any) -> torch.Tensor:
        return self.table[tokens]

    def tree_step(self, tokens: torch.Tensor, parents: torch.Tensor, base: torch.Tensor,
                  *states: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        return self.table[tokens], ()

    def ranked(self, token: int) -> list[tuple[float, int]]:
        logp = torch.log_softmax(self.table[token].double(), -1)
        order = logp.argsort(descending=True).tolist()
        return [(float(logp[t]), t) for t in order]


class _Request:
    def __init__(self, roots: list[int]) -> None:
        self.tokens = torch.tensor(roots)
        self.positions = torch.zeros(len(roots), dtype=torch.long)
        self.ends = [5] * len(roots)
        self.costs: list[Any] = []
        self.batch_size = len(roots)
        self.device = torch.device("cpu")


def _token_paths(draft: Any, row: int) -> set[tuple[int, ...]]:
    tokens, parents = draft.tokens[row].tolist(), draft.parents[row].tolist()
    out = []
    for i, token in enumerate(tokens):
        assert parents[i] < i  # topological order
        out.append((out[parents[i]] if parents[i] >= 0 else ()) + (token,))
    assert len(set(out)) == len(out)
    return set(out)


def test_sequoia_fills_every_node_with_the_token_of_its_rank() -> None:
    inner = _TableDrafter()
    tree = StaticTree.from_paths(((0,), (1,), (3,), (0, 0), (0, 2), (1, 0), (0, 0, 0), (0, 2, 1), (0, 0, 0, 0)))
    drafter = SequoiaDrafter(inner, tree)  # type: ignore[arg-type]
    roots = [1, 7]
    draft = drafter.propose(_Request(roots))  # type: ignore[arg-type]
    for row, root in enumerate(roots):
        expected = set()
        for path in tree.paths:
            token, tokens = root, ()
            for rank in path:
                token = inner.ranked(token)[rank][1]
                tokens += (token,)
            expected.add(tokens)
        assert _token_paths(draft, row) == expected


@pytest.mark.parametrize("options", [SearchOptions(4, 2, 3, 10), SearchOptions(5, 3, 4, 20, decay=1.0),
                                     SearchOptions(3, 1, 1, 3), SearchOptions(6, 4, 4, 25, decay=0.5)])
def test_specexec_expands_the_best_candidates_whatever_their_depth(options: SearchOptions) -> None:
    inner = _TableDrafter(seed=1)
    drafter = SpecExecDrafter(inner, options)  # type: ignore[arg-type]
    roots = [2, 9, 4]
    draft = drafter.propose(_Request(roots))  # type: ignore[arg-type]
    for row, root in enumerate(roots):
        pool: list[tuple[float, tuple[int, ...]]] = []  # (score, tokens of the path)
        unexpanded: list[tuple[float, int]] = []

        def add(pool: list[tuple[float, tuple[int, ...]]], unexpanded: list[tuple[float, int]], score: float,
                path: tuple[int, ...], token: int) -> None:
            for logp, child in inner.ranked(token)[:options.topk]:
                pool.append((score + max(logp, -1e4) + math.log(options.decay), (*path, child)))
                heapq.heappush(unexpanded, (-pool[-1][0], len(pool) - 1))

        add(pool, unexpanded, 0.0, (), root)
        for _ in range(options.steps):
            best = [heapq.heappop(unexpanded)[1] for _ in range(options.width)]
            for i in best:
                add(pool, unexpanded, pool[i][0], pool[i][1], pool[i][1][-1])
        chosen = sorted(pool, key=lambda c: (-c[0], len(c[1])))[:options.total_tokens]
        assert _token_paths(draft, row) == {path for _, path in chosen}


# ---- configuration --------------------------------------------------------------------------------------------------
def test_the_policies_are_registered_with_their_defaults() -> None:
    assert {"sequoia", "specexec"} <= set(POLICIES.names())
    default = SearchOptions.from_config(DecodeConfig(cycle_length=7, draft_policy="specexec"))
    assert (default.depth, default.width, default.topk, default.total_tokens, default.decay) == (6, 8, 8, 60, 0.95)
    assert SearchOptions(4, 2, 3, 10).candidates == 3 + 3 * 2 * 3
    policy = POLICIES.get("sequoia")(DecodeConfig(cycle_length=7, draft_policy="sequoia",
                                                  draft_policy_options={"acceptance": ACCEPTANCE}))
    assert policy.tree.size <= 32 and policy.tree.depth <= 6 and not policy.record


@pytest.mark.parametrize("bad", [
    {}, {"acceptance": ACCEPTANCE, "paths": ((0,),)}, {"acceptance": (0.7, 0.5)}, {"acceptance": (1.2,)},
    {"acceptance": ()}, {"acceptance": ACCEPTANCE, "total_tokens": 0}, {"acceptance": ACCEPTANCE, "depth": 2.5},
    {"acceptance": ACCEPTANCE, "draft_latency": 0.1}, {"acceptance": ACCEPTANCE, "verify_latency": (0.1,)},
    {"paths": ((0,),), "depth": 3}, {"paths": ((0, 0),)}, {"acceptance": ACCEPTANCE, "record_acceptance": 1},
    {"acceptance": ACCEPTANCE, "width": 3},
])
def test_bad_sequoia_options_are_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        POLICIES.get("sequoia")(DecodeConfig(cycle_length=5, draft_policy="sequoia", draft_policy_options=bad))


@pytest.mark.parametrize("bad", [{"depth": 0}, {"width": 4}, {"topk": 0}, {"total_tokens": 1000}, {"depth": 2.5},
                                 {"breadth": 2}, {"width": True}, {"decay": 0.0}, {"decay": 1.5}, {"decay": "x"}])
def test_bad_specexec_options_are_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        SearchOptions.from_config(DecodeConfig(cycle_length=5, draft_policy="specexec",
                                               draft_policy_options={**SPECEXEC, **bad}))


def test_list_options_are_stored_as_tuples_and_hash() -> None:
    a = DecodeConfig(draft_policy="sequoia", draft_policy_options={"paths": [[0], [0, 0]], "record_acceptance": True})
    b = DecodeConfig.from_dict(a.to_dict())
    assert a.draft_policy_options["paths"] == ((0,), (0, 0))
    assert a == b and hash(a) == hash(b)
    assert hash(_tree("sequoia")) != hash(_tree("sequoia", draft_policy_options={**SEQUOIA, "total_tokens": 9}))


@pytest.mark.parametrize("policy", sorted(OPTIONS))
@pytest.mark.parametrize(
    ("config", "decoding"),
    [
        (JumpConfig(target_layer=2, source_layer=4), dict(verifier="rejection")),
        (JumpConfig(target_layer=2, source_layer=4), dict(lossless=False)),
        (VendorMTPConfig(), {}),
    ],
    ids=["rejection", "lossy", "mtp"],
)
def test_unsupported_combinations_are_rejected(config: Any, decoding: dict[str, Any], policy: str) -> None:
    model = economize.apply(make_tiny_model(), config)
    with pytest.raises(ConfigError):
        model.decode(_prompts()[0], 4, decoding=_tree(policy, **decoding))


# ---- CUDA graphs and the economize kernel ---------------------------------------------------------------------------
@pytest.fixture
def no_native_overrides() -> Any:
    """Turn off torch's native triton overrides of aten ops for one test (see ``test_dynamic_tree.py``)."""
    try:
        from torch._native import registry
    except ImportError:
        yield
        return
    ops = sorted({op for op, _ in getattr(registry, "_graphs", {})}) or ["bmm"]
    registry.deregister_op_overrides(disable_op_symbols=ops)
    try:
        yield
    finally:
        registry.reenable_op_overrides(enable_op_symbols=ops)


@pytest.mark.parametrize("policy", sorted(OPTIONS))
@pytest.mark.parametrize("name", ["jump", "jump_exit", "eagle3", "early_exit", "frspec_jump"])
@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
@torch.no_grad()
def test_the_tree_policies_under_cuda_graphs_are_lossless_and_equal_eager(name: str, policy: str,
                                                                          no_native_overrides: Any) -> None:
    model = _build(name, "cuda")
    prompts = _prompts("cuda")
    greedy = [reference_greedy(model.base_model, p, MAX_NEW).tolist() for p in prompts]
    eager = model.decode_batch(prompts, MAX_NEW, decoding=_tree(policy))
    summary = [(r.tokens.tolist(), r.acceptance, cost_events(r.cost)) for r in eager]
    for _ in range(3):  # eager warm-up, capture, replay
        graphed = model.decode_batch(prompts, MAX_NEW, decoding=_tree(policy, cuda_graphs=True))
        assert [r.tokens.tolist() for r in graphed] == greedy
        assert [(r.tokens.tolist(), r.acceptance, cost_events(r.cost)) for r in graphed] == summary
    drafter = model.decoder(_tree(policy, cuda_graphs=True)).active_drafter
    assert drafter._root_fn._graphs
    if policy == "sequoia":
        assert len(drafter._level_fn._graphs) >= 1
    else:
        assert len(drafter._search_fn._graphs) >= SPECEXEC["depth"] - 1 and drafter._select_fn._graphs


@pytest.mark.parametrize("policy", sorted(OPTIONS))
@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graphs"])
@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
@torch.no_grad()
def test_the_tree_policies_run_on_the_economize_attention_kernel(policy: str, graphs: bool,
                                                                 no_native_overrides: Any) -> None:
    from economize.attention import install, kernel_usage, usage_summary

    install()
    base = make_tiny_model().to("cuda").eval()
    prompts = _prompts("cuda")
    greedy = [reference_greedy(base, p, MAX_NEW).tolist() for p in prompts]
    base.set_attn_implementation("economize")
    model = economize.apply(base, DRAFTERS["jump_exit"]())
    for technique in model.techniques.values():
        perturb(technique, 0.05, seed=1)
    options = ({"acceptance": (0.5, 0.2, 0.1, 0.05), "total_tokens": 48, "depth": 6} if policy == "sequoia"
               else {"depth": 6, "width": 8, "topk": 8, "total_tokens": 60})
    decoding = _tree(policy, cycle_length=7, cuda_graphs=graphs, draft_policy_options=options)
    kernel_usage(reset=True)
    for _ in range(3 if graphs else 1):
        assert [r.tokens.tolist() for r in model.decode_batch(prompts, MAX_NEW, decoding=decoding)] == greedy
    summary = usage_summary()
    assert summary["calls"].get("kernel", 0) > 0, summary
    assert set(summary["fallback_reasons"]) <= {"long_query", "no_mask"}, summary
