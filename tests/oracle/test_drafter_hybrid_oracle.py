"""economize's hybrid drafting against the decision rule of the research decoder (``tskip.bench.GraphDecoder.ngram``
with ``draft=("spec", N)``, i.e. ``bench.py --ngram K:n --hybrid N``).

Every cycle the research loop runs its lookup; on a match it verifies up to ``K`` copied tokens, otherwise it drafts
``N - 1`` jump steps. With an identity fallback (every draft step accepted) the research loop is replayed here on the
CPU with the research lookup and the reference greedy continuation. In economize, ``K`` is ``cycle_length - 1`` and
``N - 1`` is ``max_drafts``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import economize
from economize import DecodeConfig, JumpConfig
from economize.techniques.hybrid import HybridConfig
from helpers import VOCAB, make_tiny_model, reference_greedy

tskip_bench = pytest.importorskip("tskip.bench")

pytestmark = [pytest.mark.oracle, pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")]


def _tskip_lookup(row: torch.Tensor, end: int, K: int, nmax: int, nmin: int) -> tuple[int, int, list[int]]:
    T = row.numel()
    state = SimpleNamespace(seq=row.clone(), slen=torch.tensor(end), ar=torch.arange(T),
                            tok=torch.zeros(K + 1, dtype=torch.long), nres=torch.zeros(2, dtype=torch.long))
    tskip_bench.GraphDecoder._lookup(state, K, nmax, nmin)
    bn, ln = state.nres.tolist()
    return bn, ln, state.tok[1 : ln + 1].tolist()


def _replay(prompt: torch.Tensor, greedy: torch.Tensor, new: int, K: int, N: int, nmax: int, nmin: int):
    """Acceptance history (proposed, accepted) and hits of the research hybrid loop with an always-right drafter."""
    P = prompt.numel()
    buffer = torch.zeros(P + new + K + N + 2, dtype=torch.long)
    buffer[:P], buffer[P] = prompt, greedy[0]
    length, produced, history, hits = P + 1, 1, [], []
    while produced < new:
        bn, ln, proposal = _tskip_lookup(buffer, length, K, nmax, nmin)
        target = greedy[length - P : length - P + max(ln, N - 1)].tolist()
        if bn:
            accepted = next((i for i, (x, y) in enumerate(zip(proposal, target)) if x != y), ln)
        else:
            ln = accepted = N - 1  # the jump steps of ("spec", N), all accepted
        buffer[length : length + accepted + 1] = greedy[length - P : length - P + accepted + 1]
        length, produced = length + accepted + 1, produced + accepted + 1
        history.append((ln, accepted))
        hits.append(bool(bn))
    return history, hits


@pytest.mark.parametrize("K,N,nmax,nmin", [(4, 3, 3, 1), (3, 4, 2, 2), (5, 2, 1, 1), (4, 3, 3, 2)])
@torch.no_grad()
def test_cycles_follow_the_research_hybrid_loop(K: int, N: int, nmax: int, nmin: int) -> None:
    base = make_tiny_model(seed=0)
    base.model.embed_tokens.weight.mul_(10.0)  # continuations that partly repeat: lookups both hit and miss
    generator = torch.Generator().manual_seed(11)
    prompt = torch.randint(1, VOCAB, (16,), generator=generator)
    prompt[8:] = prompt[:8]
    new = 48
    greedy = reference_greedy(base, prompt, new + K + N + 1)
    history, hits = _replay(prompt, greedy, new, K, N, nmax, nmin)

    config = HybridConfig(fallback=JumpConfig(target_layer=0, source_layer=0), max_ngram=nmax, min_ngram=nmin,
                          max_drafts=N - 1, match="latest")
    model = economize.apply(base, config, decoding=DecodeConfig(cycle_length=K + 1))
    for name, parameter in model.techniques["hybrid"].fallback.predictor.named_parameters():
        if name != "out_gain":
            parameter.zero_()
    result = model.decode(prompt, new)
    assert torch.equal(result.tokens, greedy[:new])
    assert result.acceptance == history  # every cycle drafts (a lookup or N - 1 fallback steps)
    exact = sum(1 for event in result.cost.events if event.kind == "exact")
    assert exact == len(history) + 1  # the prefill, then one exact pass per cycle
    drafts = sum(1 for event in result.cost.events if event.kind == "draft")
    assert drafts == (N - 1) * sum(1 for hit in hits if not hit)
    assert any(hits) and not all(hits)
