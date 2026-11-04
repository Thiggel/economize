"""Prompt lookup (n-gram) drafting: the match rule, the proposal written into the request and lossless decoding."""

from __future__ import annotations

import pytest
import torch

import economize
from economize import ConfigError, DecodeConfig, TopPSampler
from economize.decoding import CostLog, DraftRequest, FixedLength, GreedyVerifier, RejectionVerifier
from economize.decoding.selectors import GreedySelector
from economize.techniques.lookup import LookupConfig, LookupDrafter, match_ngrams, one_hot_probs
from helpers import VOCAB, make_tiny_model, reference_greedy


def _brute_force(row: list[int], end: int, max_ngram: int, min_ngram: int, match: str = "earliest") -> tuple[int, int]:
    """(matched length, start of the copied tokens) by the definition: the longest suffix n-gram of ``row[:end]``
    with an earlier occurrence followed by at least one committed token, its first or most recent such occurrence."""
    for n in range(max_ngram, min_ngram - 1, -1):
        if n >= end:
            continue
        suffix = row[end - n : end]
        hits = [i for i in range(end - n) if row[i : i + n] == suffix]
        if hits:
            return n, (hits[-1] if match == "latest" else hits[0]) + n
    return 0, 0


def _copying_model(seed: int = 0) -> torch.nn.Module:
    """A tiny model with large tied embeddings: its greedy continuations repeat earlier tokens, so lookups hit."""
    model = make_tiny_model(seed=seed)
    model.model.embed_tokens.weight.mul_(20.0)
    return model


def _prompts(seed: int = 1, lengths: tuple[int, ...] = (12, 7, 20)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    prompts = [torch.randint(1, VOCAB, (n,), generator=generator) for n in lengths]
    prompts[-1][10:] = prompts[-1][:10]  # a prompt that repeats itself
    return prompts


def _request(seq: torch.Tensor, ends: list[int], budgets: list[int], verifier=None) -> DraftRequest:
    config = DecodeConfig()
    return DraftRequest(
        seq,
        ends,
        budgets,
        selector=GreedySelector(),
        verifier=verifier or GreedyVerifier(config),
        policy=FixedLength(config),
        costs=[CostLog() for _ in ends],
        layer_span=(1, 0),
    )


# ---- configuration -------------------------------------------------------------------------------------------------
def test_config_validation_and_round_trip() -> None:
    config = LookupConfig(max_ngram=4, min_ngram=2, max_proposal=3)
    assert LookupConfig.from_dict(config.to_dict()) == config
    assert LookupConfig().match == "earliest"
    for bad in (dict(min_ngram=0), dict(max_ngram=2, min_ngram=3), dict(max_proposal=0), dict(match="first")):
        with pytest.raises(ConfigError):
            LookupConfig(**bad)


# ---- matching ------------------------------------------------------------------------------------------------------
def test_match_prefers_the_longest_ngram_and_its_latest_occurrence() -> None:
    row = [5, 6, 7, 8, 5, 6, 7, 9, 4, 7, 3, 5, 6, 7]
    seq = torch.tensor([[*row, 0, 0]])
    matched, start = match_ngrams(seq, [len(row)], 3, 1, "latest")
    assert (matched.item(), start.item()) == (3, 7)  # [5, 6, 7] last seen at 4 (the suffix itself excluded)
    matched, start = match_ngrams(seq, [len(row)], 1, 1, "latest")
    assert (matched.item(), start.item()) == (1, 10)  # [7] last seen at 9
    matched, _ = match_ngrams(torch.tensor([[1, 2, 3, 4, 0]]), [4], 3, 1, "latest")
    assert matched.item() == 0


def test_match_prefers_the_longest_ngram_and_its_earliest_occurrence_by_default() -> None:
    row = [5, 6, 7, 8, 5, 6, 7, 9, 4, 7, 3, 5, 6, 7]
    seq = torch.tensor([[*row, 0, 0]])
    matched, start = match_ngrams(seq, [len(row)], 3, 1)
    assert (matched.item(), start.item()) == (3, 3)  # [5, 6, 7] first seen at 0
    matched, start = match_ngrams(seq, [len(row)], 1, 1)
    assert (matched.item(), start.item()) == (1, 3)  # [7] first seen at 2
    matched, start = match_ngrams(torch.tensor([[1, 2, 3, 4, 0]]), [4], 3, 1)
    assert matched.item() == 0 and start.item() == 0
    with pytest.raises(ValueError, match="match must be one of"):
        match_ngrams(seq, [len(row)], 3, 1, "first")


def test_match_ignores_tokens_past_the_end_of_each_row() -> None:
    seq = torch.tensor([[1, 2, 3, 1, 2, 3, 1, 2], [9, 1, 2, 9, 1, 2, 9, 9]])
    matched, start = match_ngrams(seq, [3, 6], 2, 1)
    assert matched.tolist() == [0, 2] and start[1].item() == 3
    matched, start = match_ngrams(seq, [5, 6], 2, 2)
    assert matched.tolist() == [2, 2] and start.tolist() == [2, 3]


@pytest.mark.parametrize("match", ["earliest", "latest"])
@pytest.mark.parametrize("max_ngram,min_ngram", [(3, 1), (4, 2), (2, 2), (1, 1)])
def test_match_equals_the_definition_on_random_sequences(max_ngram: int, min_ngram: int, match: str) -> None:
    generator = torch.Generator().manual_seed(max_ngram * 10 + min_ngram)
    seq = torch.randint(0, 4, (64, 30), generator=generator)
    ends = torch.randint(1, 30, (64,), generator=generator).tolist()
    matched, start = match_ngrams(seq, ends, max_ngram, min_ngram, match)
    for b, end in enumerate(ends):
        n, first = _brute_force(seq[b].tolist(), end, max_ngram, min_ngram, match)
        assert matched[b].item() == n
        if n:
            assert start[b].item() == first


# ---- the proposal --------------------------------------------------------------------------------------------------
def test_propose_writes_the_copied_tokens_after_each_row() -> None:
    seq = torch.zeros(3, 20, dtype=torch.long)
    seq[0, :9] = torch.tensor([1, 2, 3, 4, 5, 6, 1, 2, 3])  # [1 2 3] -> 4 5 6 1 2 ...
    seq[1, :5] = torch.tensor([7, 8, 9, 10, 11])  # no match
    seq[2, :6] = torch.tensor([4, 4, 4, 4, 4, 4])  # [4 4 4] at 2: one committed token follows
    request = _request(seq, [9, 5, 6], [4, 4, 4])
    draft = LookupDrafter(LookupConfig(max_ngram=3, match="latest"), VOCAB).propose(request)
    assert draft.lengths == [4, 0, 1] and draft.probs is None and draft.parents is None
    assert draft.tokens[0].tolist() == [4, 5, 6, 1]
    assert draft.tokens[1].tolist() == [-1] * 4 and draft.tokens[2].tolist() == [4, -1, -1, -1]
    assert seq[0, 9:13].tolist() == [4, 5, 6, 1] and seq[2, 6].item() == 4
    assert request.lengths == [4, 0, 1] and all(len(c) == 0 for c in request.costs)


def test_earliest_match_copies_the_longer_continuation() -> None:
    seq = torch.zeros(1, 20, dtype=torch.long)
    seq[0, :6] = 4  # [4 4 4] first at 0: three committed tokens follow it (the latest, at 2, has one)
    for match, length in (("earliest", 3), ("latest", 1)):
        request = _request(seq.clone(), [6], [4])
        draft = LookupDrafter(LookupConfig(max_ngram=3, match=match), VOCAB).propose(request)
        assert draft.lengths == [length] and draft.tokens[0, :length].tolist() == [4] * length


def test_proposal_respects_budgets_and_max_proposal() -> None:
    seq = torch.zeros(2, 24, dtype=torch.long)
    seq[:, :12] = torch.arange(1, 7).repeat(2)
    request = _request(seq, [12, 12], [5, 2])
    draft = LookupDrafter(LookupConfig(max_proposal=3), VOCAB).propose(request)
    assert draft.lengths == [3, 2] and draft.tokens[0].tolist() == [1, 2, 3]


def test_rejection_verifier_gets_point_masses() -> None:
    seq = torch.zeros(1, 16, dtype=torch.long)
    seq[0, :8] = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4])
    request = _request(seq, [8], [3], verifier=RejectionVerifier(DecodeConfig()))
    draft = LookupDrafter(LookupConfig(), VOCAB).propose(request)
    assert draft.probs is not None and draft.probs.shape == (1, 3, VOCAB)
    assert torch.equal(draft.probs, one_hot_probs(draft.tokens, VOCAB))
    assert draft.probs[0].sum(-1).tolist() == [1.0, 1.0, 1.0]


# ---- decoding ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cycle_length", [2, 4, 6])
@pytest.mark.parametrize("copying", [True, False], ids=["copying", "random"])
def test_greedy_lookup_decoding_equals_plain_greedy(cycle_length: int, copying: bool) -> None:
    base = _copying_model() if copying else make_tiny_model()
    model = economize.apply(base, LookupConfig(max_ngram=3), decoding=DecodeConfig(cycle_length=cycle_length))
    prompts = _prompts()
    results = model.decode_batch(prompts, [40, 25, 33])
    for prompt, result, new in zip(prompts, results, [40, 25, 33]):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, new))
    if copying:
        assert sum(a for r in results for _, a in r.acceptance) > 20


def test_cost_log_lists_only_exact_passes() -> None:
    base = _copying_model()
    model = economize.apply(base, LookupConfig(), decoding=DecodeConfig(cycle_length=5))
    result = model.decode(_prompts()[0], 30)
    kinds = {event.kind for event in result.cost.events}
    assert kinds == {"exact"}
    assert len(result.cost) >= len(result.acceptance) + 1  # the prefill, then one exact pass per cycle
    for event in result.cost.events[1:]:
        assert (event.first_layer, event.last_layer) == (1, 6)
    assert sum(a + 1 for _, a in result.acceptance) + 1 >= 30


def test_rejection_at_near_zero_temperature_is_greedy() -> None:
    base = _copying_model()
    model = economize.apply(base, LookupConfig())
    prompts = _prompts()
    config = DecodeConfig(cycle_length=4, verifier="rejection")
    results = model.decode_batch(prompts, 30, decoding=config, selector=TopPSampler(1e-6, seed=0))
    for prompt, result in zip(prompts, results):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 30))
    assert sum(a for r in results for _, a in r.acceptance) > 10


def test_save_and_load_round_trip(tmp_path) -> None:
    base = _copying_model()
    model = economize.apply(base, LookupConfig(max_ngram=4, min_ngram=2, max_proposal=3),
                         decoding=DecodeConfig(cycle_length=6))
    economize.save(model, tmp_path)
    loaded = economize.load(_copying_model(), tmp_path)
    technique = loaded.drafting_technique
    assert technique is not None and technique.config == model.drafting_technique.config
    assert loaded.decoding == model.decoding
    prompt = _prompts()[2]
    first, second = model.decode(prompt, 24), loaded.decode(prompt, 24)
    assert torch.equal(first.tokens, second.tokens) and first.acceptance == second.acceptance


def test_lookup_on_a_pruned_model_equals_its_greedy(calib) -> None:
    base = _copying_model()
    from economize.techniques.prune import PruneConfig

    model = economize.apply(base, PruneConfig(method="wanda", sparsity="2:4"), LookupConfig(),
                         decoding=DecodeConfig(cycle_length=4), calibration=calib)
    assert list(model.techniques) == ["prune", "lookup"]
    prompts = _prompts()
    for prompt, result in zip(prompts, model.decode_batch(prompts, 24)):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 24))
