"""PARD: one draft-LM call over the owed tokens and a run of mask tokens drafts a whole cycle, lossless."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import economize
from economize import ConfigError, DecodeConfig
from economize.arch import adapter_for
from economize.core.technique import TECHNIQUES
from economize.decoding import CostLog, DraftRequest, FixedLength, GreedyVerifier
from economize.decoding.selectors import GreedySelector
from economize.runtime import LayerRunner
from economize.techniques.draft_model import CATCHUP, DraftModelConfig
from economize.techniques.pard import PardConfig, PardDrafter, PardTechnique
from helpers import PROMPT_LEN, VOCAB, make_family_model, reference_greedy

MASK = 4
CAPACITY = 64


def _draft_lm(seed: int, mask: int | None = MASK) -> torch.nn.Module:
    draft = make_family_model("gemma4_text", seed=seed)
    if mask is not None:
        draft.config.pard_token = mask
    return draft


@pytest.fixture(scope="module")
def draft_dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """``same``: the model itself as the draft LM (its first draft of a cycle is always right); ``other``: another
    random model (mostly rejected drafts); ``plain``: a draft LM without ``pard_token`` in its config."""
    dirs = {}
    for name, seed, mask in (("same", 0, MASK), ("other", 1, MASK), ("plain", 1, None)):
        path = tmp_path_factory.mktemp(f"pard_{name}")
        _draft_lm(seed, mask).save_pretrained(path)
        dirs[name] = str(path)
    return dirs


def _prompts(seed: int = 1, lengths: tuple[int, ...] = (PROMPT_LEN, 7, 20)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(5, VOCAB, (n,), generator=generator) for n in lengths]


# ---- configuration -------------------------------------------------------------------------------------------------
def test_registration_and_config() -> None:
    assert TECHNIQUES.get("pard") is PardTechnique
    config = PardConfig(model="org/pard-draft", mask_token_id=MASK, max_parallel=4)
    assert PardConfig.from_dict(config.to_dict()) == config
    assert isinstance(config, DraftModelConfig)
    assert PardConfig(model="org/pard-draft").mask_token_id is None
    for bad in (dict(max_parallel=0), dict(max_parallel=2.0), dict(max_parallel=True), dict(mask_token_id=-1),
                dict(mask_token_id=False), dict(dtype="int8")):
        with pytest.raises(ConfigError):
            PardConfig(model="org/pard-draft", **bad)
    with pytest.raises(ConfigError):
        PardConfig()


def test_mask_token_comes_from_the_config_or_the_draft_lm(draft_dirs: dict[str, str]) -> None:
    base = make_family_model("gemma4_text")
    technique = economize.apply(base, PardConfig(model=draft_dirs["same"])).techniques["pard"]
    assert technique.mask_token_id == MASK and technique.drafter(None).mask_token_id == MASK
    technique = economize.apply(base, PardConfig(model=draft_dirs["same"], mask_token_id=6)).techniques["pard"]
    assert technique.drafter(None).mask_token_id == 6
    technique = economize.apply(base, PardConfig(model=draft_dirs["plain"])).techniques["pard"]
    with pytest.raises(ConfigError, match="no pard_token"):
        technique.drafter(None)
    technique = economize.apply(base, PardConfig(model=draft_dirs["plain"], mask_token_id=VOCAB)).techniques["pard"]
    with pytest.raises(ConfigError, match="outside the draft LM's vocabulary"):
        technique.drafter(None)


# ---- the parallel draft --------------------------------------------------------------------------------------------
def _request(seq: torch.Tensor, ends: list[int], budgets: list[int], drafter: PardDrafter) -> DraftRequest:
    config = DecodeConfig()
    return DraftRequest(seq, ends, budgets, selector=GreedySelector(), verifier=GreedyVerifier(config),
                        policy=FixedLength(config), costs=[CostLog() for _ in ends], footprint=drafter.footprint)


@torch.no_grad()
def _mask_scheme(draft_lm: torch.nn.Module, context: torch.Tensor, drafts: int) -> list[int]:
    """The documented scheme by a full forward of the draft LM: ``[context, mask x (drafts - 1)]`` at the positions
    ``0 .. len + drafts - 2``, the arg-max of the last ``drafts`` logits."""
    ids = torch.cat([context, torch.full((drafts - 1,), MASK, dtype=torch.long)]).view(1, -1)
    return draft_lm(ids).logits[0, -drafts:].argmax(-1).tolist()


@torch.no_grad()
@pytest.mark.parametrize("max_parallel", [1, 2, 4])
def test_one_call_equals_the_mask_scheme_across_cycles(max_parallel: int) -> None:
    """Every cycle's drafts equal the mask scheme on the committed tokens: in the first cycle (the prompt), after
    a rejection (one owed token), after a fully accepted cycle (two owed tokens), for rows with smaller budgets and
    after a row skipped a cycle. The masks of a cycle are never attended to later. One draft-LM call per cycle."""
    draft_lm = _draft_lm(1)
    runner = LayerRunner(adapter_for(draft_lm))
    drafter = PardDrafter(runner, VOCAB, MASK, max_parallel)
    calls = []
    run = drafter._run
    drafter._run = lambda *args, **kw: calls.append(args[0].shape) or run(*args, **kw)  # type: ignore[method-assign]
    drafter.begin(None, SimpleNamespace(capacity=CAPACITY), 2)
    generator = torch.Generator().manual_seed(0)
    seq = torch.zeros(2, CAPACITY + 1, dtype=torch.long)
    seq[:, :40] = torch.randint(5, VOCAB, (2, 40), generator=generator)
    ends = [9, 13]
    n = max_parallel
    cycles = [([n, n], [0, 0]), ([n, n], [n, 0]), ([n, max(n - 1, 1)], [1, n]), ([0, n], [0, 1]), ([n + 2, n], [0, 0])]
    for budgets, accepted in cycles:
        calls.clear()
        draft = drafter.propose(_request(seq, ends, budgets, drafter))
        assert len(calls) == 1
        for b in range(2):
            want = min(budgets[b], n)
            assert draft.lengths[b] == want
            if want:
                assert draft.tokens[b, :want].tolist() == _mask_scheme(draft_lm, seq[b, :ends[b]], want), (budgets, b)
        for b in range(2):
            if budgets[b] == 0:  # drafted elsewhere: commits other tokens
                seq[b, ends[b]:ends[b] + 3] = torch.randint(5, VOCAB, (3,), generator=generator)
                ends[b] += 3
                continue
            k = min(accepted[b], draft.lengths[b])
            seq[b, ends[b]:ends[b] + k] = draft.tokens[b, :k]
            seq[b, ends[b] + k] = (draft.tokens[b, k] + 1) % VOCAB if k < draft.lengths[b] else 7
            ends[b] += k + 1


@torch.no_grad()
def test_cost_log_counts_the_tokens_of_the_call() -> None:
    """The catch-up (owed tokens beyond the first) and one draft event per draft: the tokens of the one call."""
    drafter = PardDrafter(LayerRunner(adapter_for(_draft_lm(1))), VOCAB, MASK, 4)
    drafter.begin(None, SimpleNamespace(capacity=CAPACITY), 2)
    seq = torch.randint(5, VOCAB, (2, CAPACITY + 1), generator=torch.Generator().manual_seed(0))
    request = _request(seq, [9, 13], [4, 2], drafter)
    drafter.propose(request)
    for b, (end, drafts) in enumerate(((9, 4), (13, 2))):
        events = request.costs[b].events
        assert [(e.kind, e.tokens) for e in events] == [(CATCHUP, end - 1)] + [("draft", 1)] * drafts
        assert all(e.module == "draft_model" for e in events)


# ---- end to end ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("max_parallel", [1, 2, 4])
@pytest.mark.parametrize("name", ["same", "other"])
def test_greedy_equals_plain_greedy(draft_dirs: dict[str, str], name: str, max_parallel: int) -> None:
    """fp32 greedy decoding with PARD equals plain greedy decoding, for cycles that use all ``max_parallel``
    drafts, cycles longer than that (cut to it) and the confidence stop, with rows leaving the batch."""
    base = make_family_model("gemma4_text")
    model = economize.apply(base, PardConfig(model=draft_dirs[name], max_parallel=max_parallel))
    prompts = _prompts()
    for decoding in (DecodeConfig(cycle_length=max_parallel + 1), DecodeConfig(cycle_length=max_parallel + 3),
                     DecodeConfig(cycle_length=max_parallel + 1, min_confidence=0.6)):
        for limits in ([24, 24, 24], [24, 3, 11]):
            results = model.decode_batch(prompts, limits, decoding=decoding)
            for prompt, result, n in zip(prompts, results, limits):
                assert torch.equal(result.tokens, reference_greedy(base, prompt, n))
                assert all(k <= max_parallel for k, _ in result.acceptance)
    if name == "same":  # the model's own first draft is always accepted
        result = model.decode(prompts[0], 24, decoding=DecodeConfig(cycle_length=max_parallel + 1))
        assert all(a >= 1 for k, a in result.acceptance if k)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@torch.no_grad()
def test_pard_is_lossless_eager_and_graphed(draft_dirs: dict[str, str]) -> None:
    """With ``cuda_graphs`` the parallel call replays from a graph, also after rows leave the batch and after a
    longer generation replaces the draft LM's cache."""
    base = make_family_model("gemma4_text").cuda()
    model = economize.apply(base, PardConfig(model=draft_dirs["same"], max_parallel=4))
    # Seed 3: the default prompts reach a top-2 logit gap of 5e-6 at the 50th of 150 tokens, where any two kernels
    # (and Hugging Face's own cached generate) disagree; with seed 3 the smallest gap over 150 tokens is 1e-3.
    prompts = [p.cuda() for p in _prompts(seed=3)]
    for cuda_graphs in (False, True):
        decoding = DecodeConfig(cycle_length=5, cuda_graphs=cuda_graphs)
        for limits in ([24, 24, 24], [24, 24, 24], [24, 3, 11], [150, 24, 24]):
            results = model.decode_batch(prompts, limits, decoding=decoding)
            for prompt, result, n in zip(prompts, results, limits):
                assert torch.equal(result.tokens.cpu(), reference_greedy(base, prompt, n).cpu())
        drafter = model.decoder(decoding).active_drafter
        assert isinstance(drafter, PardDrafter)
        assert drafter._step_fn.enabled == cuda_graphs
        assert (drafter._parallel_graph is not None) == cuda_graphs
