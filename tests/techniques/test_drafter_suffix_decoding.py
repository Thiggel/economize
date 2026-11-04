"""SuffixDecoding: drafts equal a reference suffix tree (Arctic Inference's rules), and decoding is lossless."""

from __future__ import annotations

import heapq
import itertools
import json
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import economize
from economize import ConfigError, DecodeConfig, TopPSampler
from economize.core.config import nested_dict
from economize.decoding import CostLog, DraftRequest, FixedLength, GreedyVerifier
from economize.decoding.selectors import GreedySelector
from economize.techniques import retrieval
from economize.techniques.retrieval import SuffixIndex
from economize.techniques.suffix_decoding import SuffixDecodingConfig, SuffixDecodingDrafter
from economize.techniques.suffix_decoding.local import suffix_matches
from helpers import VOCAB, make_tiny_model, reference_greedy


def _copying_model(seed: int = 0) -> torch.nn.Module:
    model = make_tiny_model(seed=seed)
    model.model.embed_tokens.weight.mul_(20.0)
    return model


def _prompts(seed: int = 1, lengths: tuple[int, ...] = (12, 7, 20)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    prompts = [torch.randint(1, VOCAB, (n,), generator=generator) for n in lengths]
    prompts[-1][10:] = prompts[-1][:10]  # a prompt that repeats itself
    return prompts


# ---- reference: a suffix tree of dictionaries, speculated as Arctic Inference does -------------------------------
@dataclass
class _Node:
    count: int = 0
    children: dict[int, _Node] = field(default_factory=dict)


def _tree(sequences: list[list[int]], max_depth: int) -> _Node:
    root = _Node()
    for seq in sequences:
        for i in range(len(seq)):
            node = root
            for token in seq[i : i + max_depth]:
                node = node.children.setdefault(token, _Node())
                node.count += 1
    return root


def _match(root: _Node, pattern: list[int]) -> _Node | None:
    node = root
    for token in pattern:
        node = node.children.get(token)
        if node is None:
            return None
    return node


def _ranked(node: _Node, prob: float, min_prob: float) -> list[tuple[float, int, _Node]]:
    out = []
    for token, child in sorted(node.children.items(), key=lambda kv: (-kv[1].count, kv[0])):
        p = prob * child.count / node.count
        if p < min_prob:
            break
        out.append((p, token, child))
    return out


def _speculate(node: _Node, max_tokens: int, depth: int, min_prob: float, tree: bool
               ) -> tuple[list[int], list[int], float]:
    tokens, parents, probs = [], [], []
    if not tree:
        prob = 1.0
        while len(tokens) < min(max_tokens, depth):
            ranked = _ranked(node, prob, min_prob)
            if not ranked:
                break
            prob, token, node = ranked[0]
            tokens.append(token)
            parents.append(len(tokens) - 2)
            probs.append(prob)
        return tokens, parents, sum(probs)
    order = itertools.count()
    heap = [(-1.0, next(order), -1, -1, 0, node)]
    while heap and len(tokens) < max_tokens:
        negative, _, parent, token, level, node = heapq.heappop(heap)
        index = -1
        if token >= 0:
            index = len(tokens)
            tokens.append(token)
            parents.append(parent)
            probs.append(-negative)
        if level < depth and len(tokens) < max_tokens:
            for p, child_token, child in _ranked(node, -negative, min_prob):
                heapq.heappush(heap, (-p, next(order), index, child_token, level + 1, child))
    return tokens, parents, sum(probs)


def _reference(records: list[list[int]] | None, seq: list[int], config: SuffixDecodingConfig, budget: int,
               tree: bool) -> tuple[list[int], list[int]]:
    c = config
    cap = c.max_spec_tokens or budget
    if not tree:
        cap = min(cap, budget)

    def speculate(root: _Node) -> tuple[list[int], list[int], float]:
        context = seq[-c.max_depth :]
        best: tuple[list[int], list[int], float] = ([], [], 0.0)
        for m in range(1, len(context)):
            node = _match(root, context[-m:])
            if node is None:
                break
            max_tokens = min(cap, int(m * c.max_spec_factor + c.max_spec_offset + 1e-6))
            draft = _speculate(node, max_tokens, budget, c.min_token_prob, tree)
            if draft[2] >= best[2]:
                best = draft
        return best

    best = speculate(_tree(records, c.max_depth)) if records else ([], [], 0.0)
    if c.local:
        local = speculate(_tree([seq], c.max_depth))
        if local[2] >= best[2]:
            best = local
    return best[0], best[1]


# ---- configuration and the local structure -------------------------------------------------------------------------
def test_config_defaults_validation_and_round_trip() -> None:
    config = SuffixDecodingConfig(datastore="store", tree=True, max_spec_tokens=16)
    assert SuffixDecodingConfig.from_dict(config.to_dict()) == config
    default = SuffixDecodingConfig()
    assert (default.max_depth, default.max_spec_factor, default.max_spec_offset, default.min_token_prob,
            default.tree, default.local, default.datastore) == (64, 1.0, 0.0, 0.1, False, True, None)
    for bad in (dict(local=False), dict(max_depth=1), dict(max_spec_factor=-1.0), dict(min_token_prob=1.5),
                dict(max_spec_tokens=0)):
        with pytest.raises(ConfigError):
            SuffixDecodingConfig(**bad)


def test_suffix_matches_measure_the_common_suffix() -> None:
    seq = np.array([1, 2, 3, 9, 2, 3, 5, 1, 2, 3])
    ends, lengths = suffix_matches(seq, 5)
    assert ends.tolist() == [2, 5, 9] and lengths.tolist() == [3, 2, 5]  # the context itself: the limit


# ---- drafts equal the reference ------------------------------------------------------------------------------------
_CONFIGS = [
    dict(),
    dict(max_spec_factor=2.0, max_spec_offset=1.0, min_token_prob=0.05),
    dict(min_token_prob=0.3, max_spec_tokens=6),
    dict(max_spec_factor=0.5, max_spec_offset=2.0, max_spec_tokens=4),
]


@pytest.mark.parametrize("tree", [False, True], ids=["path", "tree"])
@pytest.mark.parametrize("structures", ["global", "local", "both"])
@pytest.mark.parametrize("settings", range(len(_CONFIGS)))
def test_drafts_equal_the_reference_suffix_tree(monkeypatch, tree: bool, structures: str, settings: int) -> None:
    monkeypatch.setattr(retrieval.index, "GATHER_LIMIT", 8)  # also the sampled children of large ranges
    rng = np.random.default_rng(settings)
    max_depth = 10
    records = [rng.integers(0, 4, rng.integers(2, 40)).tolist() for _ in range(60)]
    index = SuffixIndex.build(records, depth=max_depth) if structures != "local" else None
    config = SuffixDecodingConfig(datastore="unused" if index is not None else None, local=structures != "global",
                                  max_depth=max_depth, tree=tree, **_CONFIGS[settings])
    drafter = SuffixDecodingDrafter(config, index, VOCAB)
    contexts = [rng.integers(0, 4, rng.integers(1, 30)) for _ in range(40)]
    budgets = [int(b) for b in rng.integers(1, 9, len(contexts))]
    for context, budget, proposal in zip(contexts, budgets, drafter.proposals(contexts, budgets, tree)):
        tokens, parents = _reference(records if index is not None else None, context.tolist(), config, budget, tree)
        assert proposal.tokens == tokens
        assert proposal.parents == (parents if tree else None)


def test_propose_records_one_free_draft_event_per_row() -> None:
    config = SuffixDecodingConfig()
    seq = torch.zeros(2, 24, dtype=torch.long)
    seq[0, :9] = torch.tensor([4, 5, 6, 7, 8, 4, 5, 6, 7])  # [4 5 6 7] -> 8 4 5 6 (probability 1 each)
    seq[1, :4] = torch.tensor([1, 2, 3, 9])
    request = DraftRequest(seq, [9, 4], [3, 3], selector=GreedySelector(), verifier=GreedyVerifier(DecodeConfig()),
                           policy=FixedLength(DecodeConfig()), costs=[CostLog(), CostLog()], layer_span=(1, 0))
    draft = SuffixDecodingDrafter(config, None, VOCAB).propose(request)
    assert draft.lengths == [3, 0] and draft.tokens[0].tolist() == [8, 4, 5]
    assert [[e.kind for e in c.events] for c in request.costs] == [["draft"], ["draft"]]


# ---- decoding ------------------------------------------------------------------------------------------------------
def _datastore(tmp_path, base: torch.nn.Module) -> str:
    responses = [reference_greedy(base, p, 45).tolist() for p in _prompts(seed=1)]
    generator = torch.Generator().manual_seed(7)
    responses += [torch.randint(1, VOCAB, (40,), generator=generator).tolist() for _ in range(20)]
    SuffixIndex.build(responses, depth=64).save(tmp_path / "suffix_store")
    return str(tmp_path / "suffix_store")


@pytest.mark.parametrize("cycle_length", [2, 4, 6])
@pytest.mark.parametrize("tree", [False, True], ids=["path", "tree"])
@pytest.mark.parametrize("structures", ["local", "both"])
def test_greedy_suffix_decoding_equals_plain_greedy(tmp_path, cycle_length: int, tree: bool, structures: str) -> None:
    base = _copying_model()
    datastore = _datastore(tmp_path, base) if structures == "both" else None
    config = SuffixDecodingConfig(datastore=datastore, tree=tree, max_spec_tokens=8 if tree else None)
    model = economize.apply(base, config, decoding=DecodeConfig(cycle_length=cycle_length))
    prompts = _prompts()
    results = model.decode_batch(prompts, [40, 25, 33])
    for prompt, result, new in zip(prompts, results, [40, 25, 33]):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, new))
    assert sum(a for r in results for _, a in r.acceptance) > 15
    assert {e.kind for r in results for e in r.cost.events} == {"exact", "draft"}


def test_rejection_at_near_zero_temperature_is_greedy(tmp_path) -> None:
    base = _copying_model()
    model = economize.apply(base, SuffixDecodingConfig(datastore=_datastore(tmp_path, base), tree=True,
                                                       max_spec_tokens=6))
    prompts = _prompts()
    config = DecodeConfig(cycle_length=4, verifier="rejection")
    results = model.decode_batch(prompts, 30, decoding=config, selector=TopPSampler(1e-6, seed=0))
    for prompt, result in zip(prompts, results):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 30))
    assert sum(a for r in results for _, a in r.acceptance) > 10


def test_save_and_load_round_trip(tmp_path) -> None:
    base = _copying_model()
    config = SuffixDecodingConfig(datastore=_datastore(tmp_path, base), max_spec_factor=2.0, min_token_prob=0.2)
    model = economize.apply(base, config, decoding=DecodeConfig(cycle_length=6))
    economize.save(model, tmp_path / "ckpt")
    loaded = economize.load(_copying_model(), tmp_path / "ckpt")
    technique = loaded.drafting_technique
    assert technique is not None and technique.config == config


def test_config_only_checkpoint_attaches(tmp_path) -> None:
    """The layout of scripts/xb31_make_ckpts.py: a configuration without weights, attached as bench31 does."""
    base = _copying_model()
    config = SuffixDecodingConfig(datastore=_datastore(tmp_path, base), tree=True, max_spec_tokens=8)
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
