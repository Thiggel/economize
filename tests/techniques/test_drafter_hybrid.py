"""Hybrid drafting: prompt lookup first, a fallback drafter for the rows without a match."""

from __future__ import annotations

import json

import pytest
import torch

import economize
from economize import ConfigError, DecodeConfig, JumpConfig, TopPSampler
from economize.arch import adapter_for
from economize.decoding import (
    CostLog,
    CycleDecoder,
    Draft,
    DraftRequest,
    FixedLength,
    GreedyVerifier,
    RejectionVerifier,
    Verifier,
    WrappableDrafter,
)
from economize.decoding.selectors import GreedySelector
from economize.runtime import LayerRunner
from economize.techniques.frspec import FRSpecDrafter
from economize.techniques.hybrid import HybridConfig, HybridDrafter, merge_drafts
from economize.techniques.lookup import LookupConfig, LookupDrafter
from helpers import VOCAB, ScriptedDrafter, ScriptedTreeDrafter, make_tiny_model, perturb, reference_greedy

JUMP = JumpConfig(target_layer=2, source_layer=4)


def _copying_model(seed: int = 0) -> torch.nn.Module:
    model = make_tiny_model(seed=seed)
    model.model.embed_tokens.weight.mul_(20.0)
    return model


def _prompts(seed: int = 1, lengths: tuple[int, ...] = (12, 7, 20)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    prompts = [torch.randint(1, VOCAB, (n,), generator=generator) for n in lengths]
    prompts[-1][10:] = prompts[-1][:10]
    return prompts


def _hybrid(base: torch.nn.Module, config: HybridConfig | None = None, cycle_length: int = 4, **kwargs):
    decoding = DecodeConfig(cycle_length=cycle_length)
    model = economize.apply(base, config or HybridConfig(fallback=JUMP), decoding=decoding, **kwargs)
    perturb(model.techniques["hybrid"].fallback, std=0.05, seed=3)
    return model


def _identity_hybrid(base: torch.nn.Module, **fields) -> economize.EfficientModel:
    """A hybrid whose fallback is the identity jump (``m = u = 0``, ``r = L``): every fallback draft is accepted."""
    model = economize.apply(base, HybridConfig(fallback=JumpConfig(target_layer=0, source_layer=0), **fields),
                         decoding=DecodeConfig(cycle_length=4))
    for name, parameter in model.techniques["hybrid"].fallback.predictor.named_parameters():
        if name != "out_gain":
            parameter.zero_()
    return model


def _request(seq: torch.Tensor, ends: list[int], budgets: list[int], verifier=None, selector=None) -> DraftRequest:
    config = DecodeConfig()
    return DraftRequest(seq, ends, budgets, selector=selector or GreedySelector(),
                        verifier=verifier or GreedyVerifier(config),
                        policy=FixedLength(config), costs=[CostLog() for _ in ends], layer_span=(1, 1))


# ---- configuration -------------------------------------------------------------------------------------------------
def test_config_round_trips_through_json() -> None:
    config = HybridConfig(fallback=JUMP, max_ngram=4, min_ngram=2, max_proposal=6, max_drafts=2, match="latest")
    data = json.loads(json.dumps(config.to_dict()))
    assert data["fallback"]["technique"] == "jump"
    assert HybridConfig.from_dict(data) == config


@pytest.mark.parametrize("bad", [
    dict(fallback=LookupConfig()),
    dict(fallback=JUMP, max_drafts=0),
    dict(fallback=JUMP, max_proposal=0),
    dict(fallback=JUMP, max_ngram=1, min_ngram=2),
    dict(fallback=JUMP, match="first"),
    dict(fallback={"technique": "jump"}),
])
def test_config_rejects_invalid_fields(bad: dict) -> None:
    with pytest.raises(ConfigError):
        HybridConfig(**bad)


def test_fallback_must_draft() -> None:
    from economize.techniques.prune import PruneConfig

    with pytest.raises(ConfigError, match="draft"):
        economize.apply(make_tiny_model(), HybridConfig(fallback=PruneConfig(method="magnitude", sparsity=0.5)))


# ---- merging and the per-row choice --------------------------------------------------------------------------------
def test_merge_drafts_takes_rows_from_either_side() -> None:
    first = Draft(torch.tensor([[4, 5, 6], [-1, -1, -1]]), [3, 0])
    second = Draft(torch.tensor([[9, -1], [7, 8]]), [1, 2], probs=torch.rand(2, 2, VOCAB))
    merged = merge_drafts(first, second, [False, True], VOCAB)
    assert merged.lengths == [3, 2] and merged.tokens.tolist() == [[4, 5, 6], [7, 8, -1]]
    assert merged.parents is None and merged.probs is not None
    assert torch.equal(merged.probs[0], torch.nn.functional.one_hot(torch.tensor([4, 5, 6]), VOCAB).float())
    assert torch.equal(merged.probs[1, :2], second.probs[1]) and merged.probs[1, 2].sum() == 0


def test_merge_drafts_turns_chains_into_trees_for_a_tree_fallback() -> None:
    first = Draft(torch.tensor([[4, 5, 6], [-1, -1, -1]]), [3, 0])
    second = Draft(torch.tensor([[-1, -1, -1, -1], [7, 8, 9, 3]]), [0, 4],
                   parents=torch.tensor([[-1] * 4, [-1, -1, 0, 0]]))
    merged = merge_drafts(first, second, [False, True], VOCAB)
    assert merged.is_tree and merged.lengths == [3, 4]
    assert merged.parents[0, :3].tolist() == [-1, 0, 1] and merged.parents[1].tolist() == [-1, -1, 0, 0]
    assert merged.tokens[0].tolist() == [4, 5, 6, -1]


def test_fallback_drafts_only_the_rows_without_a_match() -> None:
    seq = torch.zeros(3, 24, dtype=torch.long)
    seq[0, :9] = torch.tensor([1, 2, 3, 4, 5, 6, 1, 2, 3])  # hit: proposes 4 5 6
    seq[1, :5] = torch.tensor([7, 8, 9, 10, 11])  # miss
    seq[2, :6] = torch.tensor([12, 13, 14, 15, 16, 17])  # miss
    script = torch.arange(30) % VOCAB
    fallback = ScriptedDrafter([script] * 3)
    fallback.begin(None, None, 3)
    request = _request(seq, [9, 5, 6], [3, 3, 3])
    hybrid = HybridDrafter(LookupDrafter(LookupConfig(), VOCAB), fallback, max_drafts=2, vocab_size=VOCAB)
    draft = hybrid.propose(request)
    assert draft.lengths == [3, 2, 2] and request.lengths == [3, 2, 2]
    assert draft.tokens.tolist() == [[4, 5, 6], [5, 6, -1], [6, 7, -1]]
    assert seq[0, 9:12].tolist() == [4, 5, 6] and seq[1, 5:7].tolist() == [5, 6] and seq[2, 6:8].tolist() == [6, 7]
    assert [len(cost) for cost in request.costs] == [0, 2, 2]
    assert all(event.kind == "draft" for cost in request.costs for event in cost.events)


# ---- lossless decoding ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cycle_length", [3, 5])
@pytest.mark.parametrize("copying", [True, False], ids=["copying", "random"])
def test_greedy_hybrid_decoding_equals_plain_greedy(cycle_length: int, copying: bool) -> None:
    base = _copying_model() if copying else make_tiny_model()
    model = _hybrid(base, cycle_length=cycle_length)
    prompts = _prompts()
    results = model.decode_batch(prompts, [40, 25, 33])
    for prompt, result, new in zip(prompts, results, [40, 25, 33]):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, new))
    kinds = {event.kind for r in results for event in r.cost.events}
    assert kinds <= {"exact", "draft"} and "exact" in kinds
    if not copying:
        assert "draft" in kinds


def test_identity_fallback_is_always_accepted_and_lookups_are_logged() -> None:
    base = _copying_model()
    model = _identity_hybrid(base)
    prompts = _prompts()
    results = model.decode_batch(prompts, 30)
    for prompt, result in zip(prompts, results):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 30))
        draft_cycles = sum(1 for e in result.cost.events if e.kind == "draft") // 3
        assert draft_cycles <= len(result.acceptance)
        assert all(k <= 3 for k, _ in result.acceptance)
    # every cycle drafts, by lookup or by the fallback, and the identity fallback is never rejected
    lookups = [len(r.acceptance) - sum(1 for e in r.cost.events if e.kind == "draft") // 3 for r in results]
    assert sum(lookups) > 0


def test_max_drafts_caps_the_fallback() -> None:
    base = make_tiny_model()
    model = _identity_hybrid(base, max_drafts=1)
    prompts = _prompts()
    results = model.decode_batch(prompts, 30, decoding=DecodeConfig(cycle_length=5))
    for prompt, result in zip(prompts, results):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 30))
        drafts = [e for e in result.cost.events if e.kind == "draft"]
        assert 5 <= len(drafts) <= len(result.acceptance)  # at most one fallback step per cycle


def test_tree_fallback_is_greedy() -> None:
    base = _copying_model()
    prompts = _prompts()
    greedy = [reference_greedy(base, p, 40) for p in prompts]
    scripts = [torch.cat([p, g]) for p, g in zip(prompts, greedy)]
    drafter = HybridDrafter(LookupDrafter(LookupConfig(), VOCAB), ScriptedTreeDrafter(scripts, cut=(2,)), None, VOCAB)
    decoder = CycleDecoder(LayerRunner(adapter_for(base)), drafter, DecodeConfig(cycle_length=5))
    results = decoder.decode_batch(prompts, 32)
    assert [r.tokens.tolist() for r in results] == [g[:32].tolist() for g in greedy]


def test_rejection_at_near_zero_temperature_is_greedy() -> None:
    base = _copying_model()
    model = _hybrid(base)
    prompts = _prompts()
    config = DecodeConfig(cycle_length=4, verifier="rejection")
    results = model.decode_batch(prompts, 30, decoding=config, selector=TopPSampler(1e-6, seed=0))
    for prompt, result in zip(prompts, results):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 30))


def test_rejection_verifier_gets_point_masses_for_lookup_rows() -> None:
    seq = torch.zeros(2, 24, dtype=torch.long)
    seq[0, :8] = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4])
    seq[1, :5] = torch.tensor([7, 8, 9, 10, 11])
    fallback = ScriptedDrafter([torch.arange(30)] * 2, scale=2.0)
    fallback.begin(None, None, 2)
    request = _request(seq, [8, 5], [2, 2], verifier=RejectionVerifier(DecodeConfig()),
                       selector=TopPSampler(1.0, seed=0))
    draft = HybridDrafter(LookupDrafter(LookupConfig(), VOCAB), fallback, None, VOCAB).propose(request)
    assert draft.probs is not None and draft.probs.shape == (2, 2, VOCAB)
    assert draft.probs[0].max(-1).values.tolist() == [1.0, 1.0]
    assert 0 < draft.probs[1].max(-1).values.min() < 1  # the fallback's softmax


def test_lossy_hybrid_commits_drafts() -> None:
    base = _copying_model()
    model = _identity_hybrid(base)
    prompt = _prompts()[0]
    lossy = model.decode(prompt, 30, decoding=DecodeConfig(cycle_length=4, lossless=False))
    assert lossy.tokens.numel() == 30


# ---- persistence and combination -----------------------------------------------------------------------------------
def test_save_and_load_round_trip(tmp_path) -> None:
    base = _copying_model()
    model = _hybrid(base, HybridConfig(fallback=JUMP, max_ngram=2, max_drafts=2), cycle_length=5)
    economize.save(model, tmp_path)
    loaded = economize.load(_copying_model(), tmp_path)
    technique = loaded.techniques["hybrid"]
    assert technique.config == model.techniques["hybrid"].config
    for (name, a), (_, b) in zip(model.techniques["hybrid"].state_dict().items(), technique.state_dict().items()):
        assert name.startswith("fallback.") and torch.equal(a, b)
    prompt = _prompts()[2]
    first, second = model.decode(prompt, 24), loaded.decode(prompt, 24)
    assert torch.equal(first.tokens, second.tokens) and first.acceptance == second.acceptance


def test_hybrid_on_a_pruned_model_equals_its_greedy(calib) -> None:
    from economize.techniques.prune import PruneConfig

    base = _copying_model()
    model = economize.apply(
        base, PruneConfig(method="wanda", sparsity="2:4"), HybridConfig(fallback=JUMP),
        decoding=DecodeConfig(cycle_length=4), calibration=calib)
    perturb(model.techniques["hybrid"].fallback, std=0.05, seed=3)
    assert list(model.techniques) == ["prune", "hybrid"]
    prompts = _prompts()
    for prompt, result in zip(prompts, model.decode_batch(prompts, 24)):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 24))


def test_draft_modules_are_the_fallbacks() -> None:
    model = _hybrid(make_tiny_model())
    hybrid, fallback = model.techniques["hybrid"], model.techniques["hybrid"].fallback
    own = fallback.draft_modules() if hasattr(fallback, "draft_modules") else {}
    assert set(hybrid.draft_modules()) == {f"fallback.{name}" for name in own}


# ---- primary drafters ----------------------------------------------------------------------------------------------
def test_a_lookup_primary_equals_the_built_in_lookup() -> None:
    config = HybridConfig(fallback=JUMP, primary=LookupConfig(max_ngram=2, match="latest"))
    assert HybridConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    built_in = _hybrid(_copying_model(), HybridConfig(fallback=JUMP, max_ngram=2, match="latest"))
    explicit = _hybrid(_copying_model(), config)
    assert built_in.techniques["hybrid"].state_dict().keys() == explicit.techniques["hybrid"].state_dict().keys()
    explicit.techniques["hybrid"].load_state_dict(built_in.techniques["hybrid"].state_dict())
    for prompt in _prompts():
        a, b = built_in.decode(prompt, 24), explicit.decode(prompt, 24)
        assert torch.equal(a.tokens, b.tokens) and a.acceptance == b.acceptance and a.cost.events == b.cost.events


def test_a_primary_keeps_the_ngram_fields_at_their_defaults() -> None:
    with pytest.raises(ConfigError, match="n-gram"):
        HybridConfig(fallback=JUMP, primary=LookupConfig(), max_ngram=4)


def test_the_primary_must_be_a_partial_drafter() -> None:
    hybrid = HybridConfig(fallback=JUMP, primary=JumpConfig(target_layer=1, source_layer=3))
    model = economize.apply(make_tiny_model(), hybrid,
                         decoding=DecodeConfig(cycle_length=4))
    with pytest.raises(ConfigError, match="PartialDrafter"):
        model.decode(_prompts()[0], 8)


def test_a_registered_primary_is_used_without_editing_the_hybrid() -> None:
    """A new primary drafter is one registered technique; one that serves no row leaves every row to the fallback,
    whose tokens and cost log the hybrid then reproduces."""
    from dataclasses import dataclass
    from typing import ClassVar

    from economize.core.config import TechniqueConfig
    from economize.core.metadata import Family, Saving, TechniqueInfo
    from economize.core.technique import TECHNIQUES, Capability, Technique

    @dataclass(frozen=True)
    class NeverConfig(TechniqueConfig):
        technique: ClassVar[str] = "never_primary"

    class Never:
        def begin(self, runner, cache, batch_size): ...
        def observe(self, exact, columns): ...
        def reorder(self, rows): ...
        def propose_partial(self, request): return Draft.empty(len(request.ends), request.device)

    class NeverTechnique(Technique):
        name, config_class, capabilities = "never_primary", NeverConfig, Capability.DRAFTER
        info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True)

        def drafter(self, runner):
            return Never()

    TECHNIQUES.register("never_primary", NeverTechnique)
    try:
        hybrid = _hybrid(make_tiny_model(), HybridConfig(fallback=JUMP, primary=NeverConfig()))
        plain = economize.apply(make_tiny_model(), JUMP, decoding=DecodeConfig(cycle_length=4))
        plain.techniques["jump"].load_state_dict(hybrid.techniques["hybrid"].fallback.state_dict())
        for prompt in _prompts():
            a, b = hybrid.decode(prompt, 20), plain.decode(prompt, 20)
            assert torch.equal(a.tokens, b.tokens) and a.cost.events == b.cost.events
    finally:
        TECHNIQUES.unregister("never_primary")


# ---- drafter wrappers need a wrappable drafter ----
SCRIPT = [torch.arange(40) % VOCAB] * 2


def _allowed() -> torch.Tensor:
    allowed = torch.zeros(VOCAB, dtype=torch.bool)
    allowed[: VOCAB // 2] = True
    return allowed


class _Tree(ScriptedTreeDrafter):
    """A tree drafter that declares its trees and counts its proposals."""

    def __init__(self) -> None:
        super().__init__(SCRIPT)
        self.calls = 0

    def drafts_trees(self, verifier: Verifier) -> bool:
        return verifier.supports_trees

    def propose(self, request: DraftRequest) -> Draft:
        self.calls += 1
        return super().propose(request)


def test_wrappers_need_a_wrappable_drafter() -> None:
    assert isinstance(ScriptedDrafter(SCRIPT), WrappableDrafter)
    assert not isinstance(object(), WrappableDrafter)
    with pytest.raises(ConfigError, match="WrappableDrafter"):
        FRSpecDrafter(object(), _allowed())  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="WrappableDrafter"):
        HybridDrafter(LookupDrafter(LookupConfig(), VOCAB), object(), None, VOCAB)  # type: ignore[arg-type]


def test_wrappers_forward_drafts_trees() -> None:
    greedy = GreedyVerifier(DecodeConfig())
    hybrid = HybridDrafter(LookupDrafter(LookupConfig(), VOCAB), _Tree(), None, VOCAB)
    assert hybrid.drafts_trees(greedy)
    assert not HybridDrafter(LookupDrafter(LookupConfig(), VOCAB), ScriptedDrafter(SCRIPT), None,
                             VOCAB).drafts_trees(greedy)
    assert FRSpecDrafter(_Tree(), _allowed()).drafts_trees(greedy)
    assert not FRSpecDrafter(ScriptedDrafter(SCRIPT), _allowed()).drafts_trees(greedy)
