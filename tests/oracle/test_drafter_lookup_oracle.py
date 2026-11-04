"""economize's prompt lookup against the research decoder's (``tskip.bench.GraphDecoder._lookup``, the n-gram drafter
of ``bench.py --ngram K:n[:nmin]``).

The research lookup runs inside a CUDA graph over a fixed buffer; it is called here unbound on a stand-in object
with the same buffers, on the CPU. The proposal length ``K`` of the research code is ``cycle_length - 1`` in economize.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import economize
from economize import DecodeConfig
from economize.decoding import CostLog, DraftRequest, FixedLength, GreedyVerifier
from economize.decoding.selectors import GreedySelector
from economize.techniques.lookup import LookupConfig, LookupDrafter
from helpers import VOCAB, make_tiny_model, reference_greedy

tskip_bench = pytest.importorskip("tskip.bench")

pytestmark = [pytest.mark.oracle, pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")]


def _tskip_lookup(row: torch.Tensor, end: int, K: int, nmax: int, nmin: int) -> tuple[int, int, list[int]]:
    """(matched n-gram length, proposal length, proposal) of the research lookup on ``row[:end]``."""
    T = row.numel()
    state = SimpleNamespace(
        seq=row.clone(),
        slen=torch.tensor(end),
        ar=torch.arange(T),
        tok=torch.zeros(K + 1, dtype=torch.long),
        nres=torch.zeros(2, dtype=torch.long),
    )
    tskip_bench.GraphDecoder._lookup(state, K, nmax, nmin)
    bn, ln = state.nres.tolist()
    return bn, ln, state.tok[1 : ln + 1].tolist()


def _request(seq: torch.Tensor, ends: list[int], budget: int) -> DraftRequest:
    config = DecodeConfig()
    room = torch.zeros(seq.shape[0], budget + 1, dtype=seq.dtype)  # room for the drafts, then the scratch column
    return DraftRequest(
        torch.cat([seq, room], 1),
        ends,
        [budget] * len(ends),
        selector=GreedySelector(),
        verifier=GreedyVerifier(config),
        policy=FixedLength(config),
        costs=[CostLog() for _ in ends],
        layer_span=(1, 0),
    )


@pytest.mark.parametrize("K,nmax,nmin", [(3, 3, 1), (5, 2, 2), (7, 4, 1), (1, 1, 1)])
def test_proposals_equal_the_research_lookup(K: int, nmax: int, nmin: int) -> None:
    generator = torch.Generator().manual_seed(K * 100 + nmax * 10 + nmin)
    seq = torch.randint(0, 5, (96, 40), generator=generator)
    ends = torch.randint(1, 40, (96,), generator=generator).tolist()
    request = _request(seq, ends, K)
    draft = LookupDrafter(LookupConfig(max_ngram=nmax, min_ngram=nmin, match="latest"), VOCAB).propose(request)
    for b, end in enumerate(ends):
        _bn, ln, proposal = _tskip_lookup(seq[b], end, K, nmax, nmin)
        assert draft.lengths[b] == ln, (b, end)
        assert draft.tokens[b, :ln].tolist() == proposal


@pytest.mark.parametrize("K", [2, 4])
def test_acceptance_history_equals_the_research_ngram_loop(K: int) -> None:
    """Greedy decoding with lookup drafts commits the same proposals, cycle by cycle, as ``GraphDecoder.ngram``
    (replayed here with the research lookup and the reference greedy continuation)."""
    base = make_tiny_model(seed=0)
    base.model.embed_tokens.weight.mul_(20.0)  # continuations that repeat themselves, so lookups hit
    generator = torch.Generator().manual_seed(5)
    prompt = torch.randint(1, VOCAB, (14,), generator=generator)
    new = 48
    greedy = reference_greedy(base, prompt, new + K + 1)

    buffer = torch.zeros(prompt.numel() + new + K + 2, dtype=torch.long)
    P = prompt.numel()
    buffer[:P], buffer[P] = prompt, greedy[0]
    length, produced, history = P + 1, 1, []
    while produced < new:
        _, ln, proposal = _tskip_lookup(buffer, length, K, 3, 1)
        target = greedy[length - P : length - P + ln].tolist()
        accepted = next((i for i, (x, y) in enumerate(zip(proposal, target)) if x != y), ln)
        buffer[length : length + accepted + 1] = greedy[length - P : length - P + accepted + 1]
        length, produced = length + accepted + 1, produced + accepted + 1
        history.append((ln, accepted))

    model = economize.apply(base, LookupConfig(max_ngram=3, min_ngram=1, match="latest"),
                         decoding=DecodeConfig(cycle_length=K + 1))
    result = model.decode(prompt, new)
    assert torch.equal(result.tokens, greedy[:new])
    assert result.acceptance == [(k, a) for k, a in history if k > 0]  # economize logs only cycles with drafts
    assert len(result.cost) == len(history) + 1  # the prefill, then one exact pass per cycle
    assert sum(a for _, a in history) > 10
