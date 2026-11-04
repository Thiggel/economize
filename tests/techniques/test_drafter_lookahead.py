"""Lookahead decoding: the window and pool against the official update rules, the one-pass tree and lossless
decoding."""

from __future__ import annotations

import random

import pytest
import torch

import economize
from economize import ConfigError, DecodeConfig, TopPSampler
from economize.decoding import CostLog, DraftRequest, FixedLength, GreedyVerifier
from economize.decoding.selectors import GreedySelector
from economize.techniques.lookahead import (
    JacobiWindow,
    LookaheadConfig,
    LookaheadDrafter,
    NgramPool,
    guess_tree,
    window_capacity,
)
from helpers import VOCAB, make_tiny_model, reference_greedy


def _copying_model(seed: int = 0) -> torch.nn.Module:
    """A tiny model with large tied embeddings: its greedy continuations repeat earlier tokens, so n-grams hit."""
    model = make_tiny_model(seed=seed)
    model.model.embed_tokens.weight.mul_(20.0)
    return model


def _prompts(seed: int = 1, lengths: tuple[int, ...] = (12, 7, 20)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    prompts = [torch.randint(1, VOCAB, (n,), generator=generator) for n in lengths]
    prompts[-1][10:] = prompts[-1][:10]
    return prompts


def _request(seq: torch.Tensor, ends: list[int], budgets: list[int]) -> DraftRequest:
    config = DecodeConfig()
    return DraftRequest(seq, ends, budgets, selector=GreedySelector(), verifier=GreedyVerifier(config),
                        policy=FixedLength(config), costs=[CostLog() for _ in ends], layer_span=(1, 0))


# ---- a direct port of the official update rules (jacobi_greedy_search_multilevel, update_token_map) ---------------
def _official_update_token_map(token_map: dict, lst_token: int, past_tokens: list, new_results: list, n: int,
                               w: int, g: int) -> None:
    def add(key, tup):
        grams = token_map.setdefault(key, [])
        if tup in grams:
            grams.remove(tup)
            grams.append(tup)
        elif len(grams) < g:
            grams.append(tup)
        else:
            token_map[key] = [*grams[1:], tup]

    add(lst_token, (*(past_tokens[ll][0] for ll in range(1, n - 1)), new_results[0]))
    for i in range(1, w):
        add(past_tokens[0][i - 1], (*(past_tokens[ll][i] for ll in range(1, n - 1)), new_results[i]))


def _official_step(past_tokens: list, inp: list[int], fill_level: int, n: int) -> int:
    """The window update of one official step from the predictions ``inp`` at the last present level."""
    if past_tokens[1] is None:
        past_tokens[0] = past_tokens[0][1:]
        past_tokens[1] = inp
        return 1
    if past_tokens[n - 2] is None:
        for level in range(fill_level + 1):
            past_tokens[level] = past_tokens[level][1:]
        past_tokens[fill_level + 1] = inp[1:]
        return fill_level + 1
    past_tokens[0] = past_tokens[1][1:]
    for level in range(1, n - 2):
        past_tokens[level] = past_tokens[level + 1][:]
    past_tokens[n - 2] = inp
    return fill_level


# ---- configuration -------------------------------------------------------------------------------------------------
def test_config_validation_and_round_trip() -> None:
    config = LookaheadConfig(window=7, ngram=4, guesses=5, pool_from_prompt=True, seed=3)
    assert LookaheadConfig.from_dict(config.to_dict()) == config
    assert LookaheadConfig().pool_size == 15 and config.pool_size == 5
    for bad in (dict(window=0), dict(ngram=2), dict(ngram=1.5), dict(guesses=-1), dict(seed=-1)):
        with pytest.raises(ConfigError):
            LookaheadConfig(**bad)


# ---- pool and window -----------------------------------------------------------------------------------------------
def test_pool_keeps_the_most_recent_grams_per_key() -> None:
    pool = NgramPool(2)
    for gram in [(1, 2), (3, 4), (1, 2), (5, 6)]:
        pool.add(7, gram)
    assert pool.get(7) == [(1, 2), (5, 6)] and pool.get(8) == [] and 7 in pool and len(pool) == 1
    empty = NgramPool(0)
    empty.add(7, (1, 2))
    assert empty.get(7) == []


def test_guess_tree_merges_shared_prefixes() -> None:
    tokens, parents = guess_tree([(1, 2, 3), (1, 2, 4), (5, 6, 7), (1, 9, 3)], 3)
    assert tokens == [1, 2, 3, 4, 5, 6, 7, 9, 3]
    assert parents == [-1, 0, 1, 1, -1, 4, 5, 0, 7]
    assert guess_tree([(1, 2, 3), (1, 2, 4)], 2) == ([1, 2], [-1, 0])
    assert guess_tree([(1, 2)], 0) == ([], [])


@pytest.mark.parametrize("w,n,g", [(15, 5, 15), (7, 5, 7), (4, 3, 2), (1, 3, 1), (2, 8, 3)])
def test_window_and_pool_follow_the_official_updates(w: int, n: int, g: int) -> None:
    rng = random.Random(w * 100 + n)
    initial = [rng.randrange(VOCAB) for _ in range(w + n - 3)]
    past_tokens = [initial[:]] + [None] * (n - 2)
    token_map: dict = {}
    fill_level = 0
    window, pool = JacobiWindow(initial, n - 1), NgramPool(g)
    for _ in range(3 * n):
        _, _, last = window.layout(0)
        predictions = [rng.randrange(8) for _ in last]  # a small vocabulary, so keys and grams repeat
        root = rng.randrange(8)
        full = past_tokens[n - 2] is not None
        if full:
            _official_update_token_map(token_map, root, past_tokens, predictions, n, w, g)
        fill_level = _official_step(past_tokens, predictions, fill_level, n)
        pool.add_all(window.advance(predictions, root))
        assert window.levels == [level for level in past_tokens if level is not None]
        assert window.full == (past_tokens[n - 2] is not None)
        assert len(pool) == len(token_map) and {k: pool.get(k) for k in token_map} == token_map
    assert window.full and window.size == (n - 1) * w - 1


def test_window_layout_is_the_official_mask() -> None:
    window = JacobiWindow([10, 11, 12, 13], 3)  # W = 3, N = 4: level 0 has W + N - 3 = 4 tokens
    window.advance([20, 21, 22, 23], root=0)
    tokens, parents, last = window.layout(5)
    # block 0 = [root, 11, 12, 13]; level 1 column j hangs below block 0 column j
    assert tokens == [11, 12, 13, 20, 21, 22, 23]
    assert parents == [-1, 5, 6, -1, 5, 6, 7] and last == [8, 9, 10, 11]
    window.advance([30, 31, 32, 33], root=0)
    tokens, parents, last = window.layout(0)
    assert tokens == [12, 13, 21, 22, 23, 31, 32, 33]
    assert parents == [-1, 0, -1, 0, 1, 2, 3, 4] and last == [5, 6, 7]
    assert window.full and window.size == 3 * 3 - 1


@pytest.mark.parametrize("w,n", [(15, 5), (7, 5), (2, 8), (1, 3)])
def test_window_capacity_bounds_every_window(w: int, n: int) -> None:
    window = JacobiWindow([0] * (w + n - 3), n - 1)
    sizes = []
    for _ in range(n):
        sizes.append(window.size)
        window.advance([0] * len(window.layout(0)[2]), root=0)
    assert max(sizes) == window_capacity(w, n)


# ---- the draft -----------------------------------------------------------------------------------------------------
def test_draft_puts_guesses_first_and_the_window_after_them() -> None:
    config = LookaheadConfig(window=3, ngram=4, guesses=2)
    drafter = LookaheadDrafter(config, VOCAB)
    drafter.begin(None, None, 1)
    seq = torch.zeros(1, 64, dtype=torch.long)
    seq[0, :6] = torch.tensor([1, 2, 3, 4, 5, 9])
    draft = drafter.propose(_request(seq, [6], [3]))
    assert draft.candidates == [0] and draft.lengths == [4]  # no full window yet: the window alone (W + N - 3)
    assert set(draft.tokens[0].tolist()) <= {1, 2, 3, 4, 5} and draft.parents[0].tolist() == [-1, 0, 1, 2]
    window = drafter._windows[0]
    window.levels = [[7, 8], [9, 9, 9], [9, 9, 9]]
    drafter._pools[0].add(9, (4, 5, 6))
    drafter._pools[0].add(9, (4, 7, 6))
    draft = drafter.propose(_request(seq, [6], [2]))  # a budget of two tokens cuts the n-grams
    assert draft.candidates == [3] and draft.lengths == [3 + 8]
    assert draft.tokens[0, :3].tolist() == [4, 5, 7] and draft.parents[0, :3].tolist() == [-1, 0, 0]
    assert draft.parents[0, 3:].tolist() == [-1, 3, -1, 3, 4, 5, 6, 7]  # the window never hangs below a guess
    assert draft.verifiable().lengths == [3] and draft.verifiable().width == 3


# ---- decoding ------------------------------------------------------------------------------------------------------
CONFIGS = [dict(window=4, ngram=4), dict(window=7, ngram=5, guesses=7), dict(window=2, ngram=3, guesses=1),
           dict(window=5, ngram=4, pool_from_prompt=True)]


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: "-".join(f"{k}{v}" for k, v in c.items()))
@pytest.mark.parametrize("cycle_length", [3, 5])
@pytest.mark.parametrize("copying", [True, False], ids=["copying", "random"])
def test_greedy_lookahead_decoding_equals_plain_greedy(config: dict, cycle_length: int, copying: bool) -> None:
    base = _copying_model() if copying else make_tiny_model()
    model = economize.apply(base, LookaheadConfig(**config), decoding=DecodeConfig(cycle_length=cycle_length))
    prompts = _prompts()
    results = model.decode_batch(prompts, [40, 25, 33])
    for prompt, result, new in zip(prompts, results, [40, 25, 33]):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, new))
        assert all(a <= k <= model.drafting_technique.config.pool_size * (cycle_length - 1)
                   for k, a in result.acceptance)
    if copying and config["window"] > 2:  # a two-column window rarely completes an n-gram that the output repeats
        assert sum(a for r in results for _, a in r.acceptance) > 10


def test_window_predictions_are_the_greedy_continuations_of_their_branch(monkeypatch) -> None:
    """Every window node sees the committed tokens, the first block up to its column and its column's earlier
    levels, at the positions after them: its prediction equals greedy decoding of that context."""
    base = _copying_model()
    model = economize.apply(base, LookaheadConfig(window=3, ngram=4), decoding=DecodeConfig(cycle_length=4))
    seen: list[tuple[list[int], list[int], list[int], list[int]]] = []
    propose, observe = LookaheadDrafter.propose, LookaheadDrafter.observe_logits

    def recording_propose(self, request):
        draft = propose(self, request)
        committed = request.seq[0, : request.ends[0]].tolist()
        k = draft.lengths[0]
        seen.append((committed, draft.tokens[0, :k].tolist(), draft.parents[0, :k].tolist(), []))
        return draft

    def recording_observe(self, logits):
        if self._outputs is not None:
            seen[-1][3].extend(logits[0].argmax(-1).tolist())
        observe(self, logits)

    monkeypatch.setattr(LookaheadDrafter, "propose", recording_propose)
    monkeypatch.setattr(LookaheadDrafter, "observe_logits", recording_observe)
    model.decode(_prompts()[0], 12)
    checked = 0
    for committed, tokens, parents, predicted in seen[:6]:
        for node in range(len(tokens)):
            path, i = [], node
            while i >= 0:
                path.append(tokens[i])
                i = parents[i]
            context = torch.tensor(committed + path[::-1])
            with torch.no_grad():
                expected = base(context[None]).logits[0, -1].argmax().item()
            assert predicted[1 + node] == expected
            checked += 1
    assert checked > 20


def test_cost_log_counts_every_node_of_the_pass() -> None:
    base = _copying_model()
    model = economize.apply(base, LookaheadConfig(window=4, ngram=4), decoding=DecodeConfig(cycle_length=4))
    result = model.decode(_prompts()[0], 30)
    assert {event.kind for event in result.cost.events} == {"exact"}
    for event in result.cost.events[1:]:
        assert (event.first_layer, event.last_layer) == (1, 6)
    assert len(result.cost) >= len(result.acceptance) + 1


def test_rejection_at_near_zero_temperature_is_greedy() -> None:
    base = _copying_model()
    model = economize.apply(base, LookaheadConfig(window=4, ngram=4))
    prompts = _prompts()
    config = DecodeConfig(cycle_length=4, verifier="rejection")
    results = model.decode_batch(prompts, 30, decoding=config, selector=TopPSampler(1e-6, seed=0))
    for prompt, result in zip(prompts, results):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 30))
    assert sum(a for r in results for _, a in r.acceptance) > 5


def test_typical_acceptance_with_a_greedy_selector_is_greedy() -> None:
    base = _copying_model()
    model = economize.apply(base, LookaheadConfig(window=4, ngram=4))
    prompts = _prompts()
    results = model.decode_batch(prompts, 30, decoding=DecodeConfig(cycle_length=4, verifier="typical"))
    for prompt, result in zip(prompts, results):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 30))


def test_lossy_verifier_is_rejected() -> None:
    model = economize.apply(make_tiny_model(), LookaheadConfig(window=4, ngram=4))
    with pytest.raises(ConfigError):
        model.decode(_prompts()[0], 8, decoding=DecodeConfig(cycle_length=4, lossless=False))


def test_save_and_load_round_trip(tmp_path) -> None:
    base = _copying_model()
    model = economize.apply(base, LookaheadConfig(window=6, ngram=4, guesses=3), decoding=DecodeConfig(cycle_length=4))
    economize.save(model, tmp_path)
    loaded = economize.load(_copying_model(), tmp_path)
    technique = loaded.drafting_technique
    assert technique is not None and technique.config == model.drafting_technique.config
    checkpoint = economize.read_checkpoint(tmp_path)
    attached = economize.attach(_copying_model(), economize.Checkpoint(list(checkpoint.configs), checkpoint.states))
    prompt = _prompts()[0]
    result = attached.decode(prompt, 20, decoding=DecodeConfig(cycle_length=4))
    assert torch.equal(result.tokens, reference_greedy(base, prompt, 20))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@torch.no_grad()
def test_graphed_lookahead_decoding_is_greedy_and_replayed() -> None:
    base = _copying_model().cuda()
    model = economize.apply(base, LookaheadConfig(window=5, ngram=4))
    decoding = DecodeConfig(cycle_length=4, cuda_graphs=True)
    prompts = [p.cuda() for p in _prompts()]
    greedy = [reference_greedy(base, p, 30).tolist() for p in prompts]
    for _ in range(3):  # eager warm-up, capture, replay
        results = model.decode_batch(prompts, 30, decoding=decoding)
        assert [r.tokens.tolist() for r in results] == greedy
        assert sum(a for r in results for _, a in r.acceptance) > 0
    graphs = model.decoder(decoding).graphs
    assert graphs is not None and graphs.num_graphs > 0
