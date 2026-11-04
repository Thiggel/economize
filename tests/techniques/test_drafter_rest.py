"""REST (retrieval-based speculative decoding): the continuation trie, the proposal and lossless decoding."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import economize
from economize import ConfigError, DecodeConfig, TopPSampler
from economize.core.config import nested_dict
from economize.decoding import CostLog, DraftRequest, FixedLength, GreedyVerifier, RejectionVerifier
from economize.decoding.selectors import GreedySelector
from economize.techniques.rest import RestConfig, RestDrafter
from economize.techniques.rest.trie import build_trie
from economize.techniques.retrieval import SuffixIndex
from helpers import VOCAB, make_tiny_model, reference_greedy


def _copying_model(seed: int = 0) -> torch.nn.Module:
    model = make_tiny_model(seed=seed)
    model.model.embed_tokens.weight.mul_(20.0)
    return model


def _prompts(seed: int = 1, lengths: tuple[int, ...] = (12, 7, 20)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in lengths]


def _request(seq: torch.Tensor, ends: list[int], budgets: list[int], verifier=None) -> DraftRequest:
    config = DecodeConfig()
    return DraftRequest(seq, ends, budgets, selector=GreedySelector(), verifier=verifier or GreedyVerifier(config),
                        policy=FixedLength(config), costs=[CostLog() for _ in ends], layer_span=(1, 0))


def _datastore(tmp_path, base: torch.nn.Module, noise: int = 30) -> str:
    """An index of the model's greedy continuations of the test prompts (from other starts: different prompts with
    the same tail) and of random records, so retrieved drafts are partly right."""
    records = []
    for prompt in _prompts():
        records.append(torch.cat([prompt, reference_greedy(base, prompt, 45)]).tolist())
    generator = torch.Generator().manual_seed(7)
    records += [torch.randint(1, VOCAB, (40,), generator=generator).tolist() for _ in range(noise)]
    path = tmp_path / "rest_store"
    SuffixIndex.build(records, depth=32).save(path)
    return str(path)


# ---- reference implementation ----------------------------------------------------------------------------------
def _continuations(records: list[list[int]], context: list[int], max_span: int, min_span: int, depth: int
                   ) -> list[tuple[int, ...]]:
    """The continuations (up to ``depth`` tokens, within the record) of the longest suffix of ``context`` with an
    occurrence, by the definition (every occurrence)."""
    for span in range(min(max_span, len(context)), min_span - 1, -1):
        suffix = context[-span:]
        out = [tuple(r[i + span : i + span + depth]) for r in records for i in range(len(r) - span + 1)
               if r[i : i + span] == suffix]
        if out:
            return out
    return []


def _node_counts(continuations: list[tuple[int, ...]]) -> Counter:
    return Counter(c[:j] for c in continuations for j in range(1, len(c) + 1))


def _reference_path(counts: Counter) -> list[int]:
    path: tuple[int, ...] = ()
    while True:
        children = [(-n, p[-1]) for p, n in counts.items() if len(p) == len(path) + 1 and p[:-1] == path]
        if not children:
            return list(path)
        path = (*path, min(children)[1])


def _reference_top(counts: Counter, size: int) -> set[tuple[int, ...]]:
    return set(sorted(counts, key=lambda p: (-counts[p], len(p), p))[:size])


def _prefixes(tokens: list[int], parents: list[int]) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for token, parent in zip(tokens, parents):
        assert parent < len(out)  # topological order
        out.append((out[parent] if parent >= 0 else ()) + (token,))
    return out


# ---- configuration -------------------------------------------------------------------------------------------------
def test_config_defaults_validation_and_round_trip() -> None:
    config = RestConfig(datastore="store", max_span=8, tree=False)
    assert RestConfig.from_dict(config.to_dict()) == config
    default = RestConfig(datastore="store")
    assert (default.max_span, default.min_span, default.max_occurrences, default.max_depth, default.tree_size,
            default.tree) == (16, 2, 5000, 10, 64, True)
    for bad in (dict(datastore=""), dict(min_span=0), dict(min_span=5, max_span=4), dict(max_depth=0),
                dict(tree_size=0), dict(max_occurrences=0), dict(occurrences="last")):
        with pytest.raises(ConfigError):
            RestConfig(**{"datastore": "store", **bad})


# ---- the trie ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(4))
def test_trie_counts_every_prefix(seed: int) -> None:
    rng = np.random.default_rng(seed)
    table = rng.integers(0, 3, (60, 5))
    for i, cut in enumerate(rng.integers(1, 6, 60)):
        table[i, cut:] = -1
    table = table[np.lexsort(table.T[::-1])]
    trie = build_trie(table)
    rows = [tuple(t for t in r if t >= 0) for r in table.tolist()]
    counts = _node_counts(rows)
    prefixes = _prefixes(trie.tokens.tolist(), trie.parents.tolist())
    assert dict(zip(prefixes, trie.counts.tolist())) == dict(counts)
    assert trie.depths.tolist() == [len(p) for p in prefixes]
    assert trie.path() == _reference_path(counts)
    for size in (1, 5, 20, 1000):
        tokens, parents = trie.top(size)
        assert set(_prefixes(tokens, parents)) == _reference_top(counts, size)


def test_trie_of_an_empty_table() -> None:
    trie = build_trie(np.zeros((0, 4), dtype=np.int64))
    assert len(trie) == 0 and trie.path() == [] and trie.top(4) == ([], [])


# ---- the proposal --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("tree", [False, True], ids=["chain", "tree"])
def test_proposals_equal_the_definition(tree: bool) -> None:
    rng = np.random.default_rng(0)
    records = [rng.integers(0, 4, rng.integers(5, 40)).tolist() for _ in range(80)]
    index = SuffixIndex.build(records, depth=24)
    config = RestConfig(datastore="unused", max_span=6, min_span=2, max_depth=5, tree_size=12)
    drafter = RestDrafter(config, index, VOCAB)
    contexts = [rng.integers(0, 4, rng.integers(1, 12)) for _ in range(40)] + [np.array([9, 9, 9])]
    budgets = [int(b) for b in rng.integers(1, 7, len(contexts))]
    for context, budget, proposal in zip(contexts, budgets, drafter.proposals(contexts, budgets, tree)):
        continuations = _continuations(records, context.tolist(), 6, 2, min(budget, 5))
        counts = _node_counts(continuations)
        if proposal is None:  # too short to search
            assert len(context) < 2
        elif not tree:
            assert proposal.parents is None and proposal.tokens == _reference_path(counts)
        else:
            assert set(_prefixes(proposal.tokens, proposal.parents)) == _reference_top(counts, 12)


def test_occurrences_are_capped_first_or_spread() -> None:
    index = SuffixIndex.build([[1, 2, 3]] * 20 + [[1, 2, 4]] * 30, depth=8)
    (lo,), (hi,) = index.find([[1, 2]])
    paths = {}
    for rule in ("first", "spread"):
        config = RestConfig(datastore="unused", max_occurrences=10, max_depth=1, occurrences=rule, tree=False)
        paths[rule] = RestDrafter(config, index, VOCAB).draft(2, int(lo), int(hi), 1, tree=False).tokens
    assert paths == {"first": [3], "spread": [4]}  # the first 10 in suffix order all continue with 3


def test_propose_writes_chains_and_returns_trees_with_draft_costs() -> None:
    index = SuffixIndex.build([[5, 6, 7, 8, 9], [5, 6, 7, 2], [5, 6, 7, 8, 1]], depth=16)
    seq = torch.zeros(2, 20, dtype=torch.long)
    seq[0, :4] = torch.tensor([3, 4, 5, 6])  # [5 6] -> 7 (3), 7 8 (2), then ties: the lower token
    seq[1, :3] = torch.tensor([10, 11, 12])  # no match
    chain = RestDrafter(RestConfig(datastore="unused", tree=False), index, VOCAB)
    request = _request(seq, [4, 3], [3, 3])
    draft = chain.propose(request)
    assert draft.lengths == [3, 0] and draft.parents is None and draft.tokens[0].tolist() == [7, 8, 1]
    assert seq[0, 4:7].tolist() == [7, 8, 1] and request.lengths == [3, 0]
    assert [[e.kind for e in c.events] for c in request.costs] == [["draft"], ["draft"]]
    assert all(e.first_layer > e.last_layer for c in request.costs for e in c.events)  # no layer runs
    tree = RestDrafter(RestConfig(datastore="unused", tree_size=4), index, VOCAB)
    request = _request(seq.clone(), [4, 3], [3, 3])
    draft = tree.propose(request)
    assert draft.parents is not None and draft.lengths == [4, 0] and request.lengths == [0, 0]
    assert set(_prefixes(draft.tokens[0, :4].tolist(), draft.parents[0, :4].tolist())) == {
        (7,), (7, 8), (7, 2), (7, 8, 1)}


def test_rejection_verifier_gets_point_masses_for_chains() -> None:
    index = SuffixIndex.build([[1, 2, 3, 4, 5]], depth=8)
    seq = torch.zeros(1, 12, dtype=torch.long)
    seq[0, :2] = torch.tensor([1, 2])
    drafter = RestDrafter(RestConfig(datastore="unused", tree=False), index, VOCAB)
    draft = drafter.propose(_request(seq, [2], [3], verifier=RejectionVerifier(DecodeConfig())))
    assert draft.probs is not None and draft.probs[0].argmax(-1).tolist() == [3, 4, 5]


# ---- decoding ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cycle_length", [2, 4, 6])
@pytest.mark.parametrize("tree", [False, True], ids=["chain", "tree"])
def test_greedy_rest_decoding_equals_plain_greedy(tmp_path, cycle_length: int, tree: bool) -> None:
    base = _copying_model()
    config = RestConfig(datastore=_datastore(tmp_path, base), tree=tree, tree_size=16)
    model = economize.apply(base, config, decoding=DecodeConfig(cycle_length=cycle_length))
    prompts = [torch.cat([torch.randint(1, VOCAB, (3,)), p[-6:]]) for p in _prompts()]  # unseen heads
    results = model.decode_batch(prompts, [40, 25, 33])
    for prompt, result, new in zip(prompts, results, [40, 25, 33]):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, new))
    assert sum(a for r in results for _, a in r.acceptance) > 15


def test_cost_log_lists_exact_passes_and_free_drafts(tmp_path) -> None:
    base = _copying_model()
    model = economize.apply(base, RestConfig(datastore=_datastore(tmp_path, base)),
                            decoding=DecodeConfig(cycle_length=5))
    result = model.decode(_prompts()[0], 30)
    kinds = {event.kind for event in result.cost.events}
    assert kinds == {"exact", "draft"}
    for event in result.cost.events:
        if event.kind == "draft":
            assert event.first_layer > event.last_layer


def test_rejection_at_near_zero_temperature_is_greedy(tmp_path) -> None:
    base = _copying_model()
    model = economize.apply(base, RestConfig(datastore=_datastore(tmp_path, base), tree_size=8))
    prompts = _prompts()
    config = DecodeConfig(cycle_length=4, verifier="rejection")
    results = model.decode_batch(prompts, 30, decoding=config, selector=TopPSampler(1e-6, seed=0))
    for prompt, result in zip(prompts, results):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 30))
    assert sum(a for r in results for _, a in r.acceptance) > 10


def test_save_and_load_round_trip(tmp_path) -> None:
    base = _copying_model()
    model = economize.apply(base, RestConfig(datastore=_datastore(tmp_path, base), max_span=8, tree=False),
                            decoding=DecodeConfig(cycle_length=6))
    economize.save(model, tmp_path / "ckpt")
    loaded = economize.load(_copying_model(), tmp_path / "ckpt")
    technique = loaded.drafting_technique
    assert technique is not None and technique.config == model.drafting_technique.config


def test_config_only_checkpoint_attaches(tmp_path) -> None:
    """The layout of scripts/xb31_make_ckpts.py: a configuration without weights, attached as bench31 does."""
    base = _copying_model()
    config = RestConfig(datastore=_datastore(tmp_path, base), tree_size=8)
    path = tmp_path / "ckpt"
    path.mkdir()
    meta = {"economize_version": economize.__version__, "base_model": None, "techniques": [nested_dict(config)]}
    (path / "economize_config.json").write_text(json.dumps(meta))
    save_file({}, path / "economize_model.safetensors")
    ckpt = economize.read_checkpoint(path)
    model = economize.attach(base, economize.Checkpoint(list(ckpt.configs), ckpt.states))
    assert model.drafting_technique.config == config
    prompt = _prompts()[2]
    result = model.decode(prompt, 20, decoding=DecodeConfig(cycle_length=5, lossless=True))
    assert torch.equal(result.tokens, reference_greedy(base, prompt, 20))
