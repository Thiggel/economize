"""FR-Spec: drafts restricted to a draft vocabulary of frequent tokens, verified over the full vocabulary."""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

import economize
from economize import ConfigError, DecodeConfig, JumpConfig, TopPSampler
from economize.arch import ArchAdapter, adapter_for
from economize.compat.tskip import cost_events
from economize.core.technique import TECHNIQUES
from economize.decoding import (
    CostKind,
    CostLog,
    CycleDecoder,
    Draft,
    DraftRequest,
    FixedLength,
    GreedyVerifier,
    RejectionVerifier,
    Verifier,
)
from economize.decoding.selectors import GreedySelector
from economize.runtime import LayerRunner
from economize.techniques._drafting import DelegatingDrafter, DraftModules, LinearSlice, adapter_head_slice
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.early_exit import EarlyExitConfig
from economize.techniques.frspec import (
    FRSpecConfig,
    FRSpecDrafter,
    FRSpecTechnique,
    frequency_order,
    read_token_ids,
    restrict_logits,
    token_counts,
    widen_logits,
)
from economize.techniques.hybrid import HybridConfig
from economize.techniques.lookup import LookupConfig
from economize.techniques.lookup.proposal import chain_draft
from economize.techniques.vendor_mtp import VendorMTPConfig
from helpers import (
    VOCAB,
    ScriptedDrafter,
    ScriptedTreeDrafter,
    make_family_model,
    make_tiny_model,
    perturb,
    reference_greedy,
)

K = 64
JUMP = JumpConfig(target_layer=2, source_layer=4)
INNERS = {
    "jump": JUMP,
    "early_exit": EarlyExitConfig(exit_layer=3),
    "eagle3": Eagle3Config(),
}


def _prompts(seed: int = 1, lengths: tuple[int, ...] = (12, 7, 20)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in lengths]


def _ids(seed: int = 5, size: int = K) -> tuple[int, ...]:
    return tuple(torch.randperm(VOCAB, generator=torch.Generator().manual_seed(seed))[:size].tolist())


def _model(base: torch.nn.Module, inner: Any = JUMP, size: int = K, ids: tuple[int, ...] | None = None,
           cycle_length: int = 4, std: float = 0.05, **kwargs: Any) -> economize.EfficientModel:
    ids = _ids(size=size) if ids is None and "calibration" not in kwargs else ids
    model = economize.apply(base, FRSpecConfig(drafter=inner, vocab_size=size, token_ids=ids),
                         decoding=DecodeConfig(cycle_length=cycle_length), **kwargs)
    if std:
        perturb(model.techniques["frspec"].inner, std=std, seed=3)
    return model


def _identity(base: torch.nn.Module, ids: tuple[int, ...], cycle_length: int = 4) -> economize.EfficientModel:
    """FR-Spec around the identity jump (``m = u = 0``, ``r = L``): every draft is the model's own argmax over the
    draft vocabulary, so a draft is accepted exactly when the greedy token lies in it."""
    model = _model(base, JumpConfig(target_layer=0, source_layer=0), len(ids), ids, cycle_length, std=0.0)
    for name, parameter in model.techniques["frspec"].inner.predictor.named_parameters():
        if name != "out_gain":
            parameter.zero_()
    return model


def _request(seq: torch.Tensor, ends: list[int], budgets: list[int], verifier=None, selector=None) -> DraftRequest:
    config = DecodeConfig()
    return DraftRequest(seq, ends, budgets, selector=selector or GreedySelector(),
                        verifier=verifier or GreedyVerifier(config), policy=FixedLength(config),
                        costs=[CostLog() for _ in ends], layer_span=(1, 1))


# ---- configuration -------------------------------------------------------------------------------------------------
def test_registered_and_round_trips_through_json() -> None:
    assert TECHNIQUES.get("frspec") is FRSpecTechnique
    config = FRSpecConfig(drafter=JUMP, vocab_size=3, token_ids=[7, 1, 4])
    assert config.token_ids == (7, 1, 4)
    data = json.loads(json.dumps(config.to_dict()))
    assert data["drafter"]["technique"] == "jump"
    assert FRSpecConfig.from_dict(data) == config
    assert FRSpecConfig.from_dict(json.loads(json.dumps(FRSpecConfig(drafter=JUMP, vocab_size=3).to_dict()))) \
        == FRSpecConfig(drafter=JUMP, vocab_size=3)


@pytest.mark.parametrize("bad", [
    dict(drafter=LookupConfig(), vocab_size=8),
    dict(drafter=HybridConfig(fallback=JUMP), vocab_size=8),
    dict(drafter=FRSpecConfig(drafter=JUMP, vocab_size=8), vocab_size=8),
    dict(drafter=JUMP, vocab_size=0),
    dict(drafter=JUMP, vocab_size=2, token_ids=(1,)),
    dict(drafter=JUMP, vocab_size=2, token_ids=(1, 1)),
    dict(drafter=JUMP, vocab_size=2, token_ids=(1, -1)),
    dict(drafter={"technique": "jump"}, vocab_size=2),
], ids=["lookup", "hybrid", "nested", "empty", "length", "duplicate", "negative", "bad_nesting"])
def test_config_rejects_invalid_fields(bad: dict) -> None:
    with pytest.raises(ConfigError):
        FRSpecConfig(**bad)


@pytest.mark.parametrize("bad", [
    dict(drafter=JUMP, vocab_size=VOCAB + 1),
    dict(drafter=JUMP, vocab_size=2, token_ids=(1, VOCAB)),
    dict(drafter=JumpConfig(target_layer=2, source_layer=40), vocab_size=2),
])
def test_validation_against_the_model(bad: dict) -> None:
    with pytest.raises(ConfigError):
        economize.apply(make_tiny_model(), FRSpecConfig(**bad))


def test_inner_must_draft() -> None:
    from economize.techniques.prune import PruneConfig

    with pytest.raises(ConfigError, match="draft"):
        economize.apply(make_tiny_model(), FRSpecConfig(drafter=PruneConfig(method="magnitude", sparsity=0.5),
                                                     vocab_size=8, token_ids=tuple(range(8))))


def test_a_missing_vocabulary_raises() -> None:
    with pytest.raises(ConfigError, match="calibration"):
        economize.apply(make_tiny_model(), FRSpecConfig(drafter=JUMP, vocab_size=8))


def test_set_vocab_validates_and_sets_the_mask() -> None:
    technique = _model(make_tiny_model(), size=4, ids=(1, 2, 3, 4)).techniques["frspec"]
    technique.set_vocab([9, 0, 5, 7])
    assert technique.draft_ids.tolist() == [9, 0, 5, 7]
    assert technique.allowed().nonzero().flatten().tolist() == [0, 5, 7, 9]
    for bad in ([1, 2, 3], [1, 1, 2, 3], [1, 2, 3, VOCAB], [-1, 2, 3, 4]):
        with pytest.raises(ConfigError):
            technique.set_vocab(bad)


# ---- frequency ranking ---------------------------------------------------------------------------------------------
def test_frequency_order_ranks_by_count_and_breaks_ties_by_id() -> None:
    tokens = torch.tensor([[5, 3, 5, 9, 3, 5], [1, 9, 9, 0, 3, 7]])
    assert token_counts(tokens, 10).tolist() == [1, 1, 0, 3, 0, 3, 0, 1, 0, 3]
    assert frequency_order(tokens, 10).tolist() == [3, 5, 9, 0, 1, 7, 2, 4, 6, 8]
    assert frequency_order([tokens[0], tokens[1]], 10).tolist() == frequency_order(tokens, 10).tolist()
    with pytest.raises(ConfigError):
        token_counts(torch.tensor([10]), 10)


def test_calibration_data_picks_the_most_frequent_tokens(calib: torch.Tensor) -> None:
    model = _model(make_tiny_model(), calibration=calib)
    assert model.techniques["frspec"].draft_ids.tolist() == frequency_order(calib, VOCAB)[:K].tolist()


def test_explicit_ids_win_over_calibration_data(calib: torch.Tensor) -> None:
    model = _model(make_tiny_model(), ids=_ids(), calibration=calib)
    assert tuple(model.techniques["frspec"].draft_ids.tolist()) == _ids()


# ---- the restriction -----------------------------------------------------------------------------------------------
def test_restrict_logits_masks_and_pads() -> None:
    logits = torch.randn(2, 6)
    allowed = torch.tensor([True, False, True, False])
    out = restrict_logits(logits, allowed)
    assert torch.equal(out[:, [0, 2]], logits[:, [0, 2]])
    assert torch.isinf(out[:, [1, 3, 4, 5]]).all()


def test_restricted_steps_equal_the_argmax_over_the_draft_ids() -> None:
    base = make_tiny_model()
    model = _model(base)
    technique = model.techniques["frspec"]
    ids = technique.draft_ids
    allowed = technique.allowed()
    seq = torch.zeros(3, 40, dtype=torch.long)
    prompts = _prompts()
    for row, prompt in enumerate(prompts):
        seq[row, : prompt.numel()] = prompt
    script = [torch.arange(40) % VOCAB] * 3
    drafter = FRSpecDrafter(ScriptedDrafter(script), allowed)
    drafter.begin(None, None, 3)
    tokens = torch.tensor([3, 4, 5])
    logits = drafter.step(tokens, torch.tensor([3, 4, 5]), None)
    inner = ScriptedDrafter(script)
    inner.begin(None, None, 3)
    reference = inner.step(tokens, torch.tensor([3, 4, 5]), None)
    assert torch.equal(logits, restrict_logits(reference, allowed))
    noise = torch.randn(5, VOCAB, generator=torch.Generator().manual_seed(0))
    assert torch.equal(ids[noise[:, ids].argmax(-1)], restrict_logits(noise, allowed).argmax(-1))
    request = _request(seq, [p.numel() for p in prompts], [3, 3, 3])
    draft = drafter.propose(request)
    drafted = draft.tokens[draft.tokens >= 0]
    assert drafted.numel() == 9 and allowed[drafted].all()


def test_rejection_drafts_have_no_mass_outside_the_vocabulary() -> None:
    allowed = torch.zeros(VOCAB, dtype=torch.bool)
    allowed[torch.tensor(_ids())] = True
    drafter = FRSpecDrafter(ScriptedDrafter([torch.arange(40) % VOCAB] * 2, scale=2.0), allowed)
    drafter.begin(None, None, 2)
    seq = torch.zeros(2, 24, dtype=torch.long)
    request = _request(seq, [5, 8], [3, 3], verifier=RejectionVerifier(DecodeConfig()),
                       selector=TopPSampler(1.0, seed=0))
    draft = drafter.propose(request)
    assert draft.probs is not None and draft.probs.shape == (2, 3, VOCAB)
    assert draft.probs[..., ~allowed].abs().max() == 0
    torch.testing.assert_close(draft.probs.sum(-1), torch.ones(2, 3))
    assert allowed[draft.tokens].all()


def test_tree_drafters_are_rejected() -> None:
    base = make_tiny_model()
    prompts = _prompts()
    scripts = [torch.cat([p, reference_greedy(base, p, 30)]) for p in prompts]
    drafter = FRSpecDrafter(ScriptedTreeDrafter(scripts, cut=(2,)), torch.ones(VOCAB, dtype=torch.bool))
    decoder = CycleDecoder(LayerRunner(adapter_for(base)), drafter, DecodeConfig(cycle_length=5))
    with pytest.raises(ConfigError, match="chain"):
        decoder.decode_batch(prompts, 16)


# ---- lossless decoding ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cycle_length", [2, 4, 6])
@pytest.mark.parametrize("inner", INNERS)
def test_greedy_equals_plain_greedy(inner: str, cycle_length: int) -> None:
    base = make_tiny_model()
    model = _model(base, INNERS[inner], cycle_length=cycle_length)
    prompts = _prompts()
    results = model.decode_batch(prompts, [32, 20, 27])
    for prompt, result, new in zip(prompts, results, [32, 20, 27]):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, new))
    kinds = {e.kind for r in results for e in r.cost.events}
    assert {CostKind.EXACT, CostKind.DRAFT} <= kinds <= {CostKind.EXACT, CostKind.DRAFT, CostKind.DRAFT_OBSERVE}


def test_drafts_are_accepted_exactly_when_the_greedy_token_is_in_the_vocabulary() -> None:
    base = make_tiny_model()
    prompt = _prompts()[0]
    greedy = reference_greedy(base, prompt, 40)
    inside = tuple(dict.fromkeys(greedy.tolist()))
    model = _identity(base, inside)
    result = model.decode(prompt, 40)
    assert torch.equal(result.tokens, greedy)
    assert all(a == k for k, a in result.acceptance)
    outside = tuple(i for i in range(VOCAB) if i not in set(greedy[1::2].tolist()))
    result = _identity(base, outside).decode(prompt, 40)
    assert torch.equal(result.tokens, greedy)
    allowed = torch.zeros(VOCAB, dtype=torch.bool)
    allowed[torch.tensor(outside)] = True
    position = 1  # the first token comes from the prefill's exact pass
    for drafted, accepted in result.acceptance:
        run = 0
        while run < drafted and position + run < greedy.numel() and allowed[greedy[position + run]]:
            run += 1
        assert accepted == run
        position += accepted + 1


@pytest.mark.parametrize("inner", INNERS)
def test_rejection_at_near_zero_temperature_is_greedy(inner: str) -> None:
    base = make_tiny_model()
    model = _model(base, INNERS[inner])
    prompts = _prompts()
    config = DecodeConfig(cycle_length=4, verifier="rejection")
    results = model.decode_batch(prompts, 24, decoding=config, selector=TopPSampler(1e-6, seed=0))
    for prompt, result in zip(prompts, results):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 24))


def test_sampling_runs_and_commits_tokens() -> None:
    model = _model(make_tiny_model())
    result = model.decode(_prompts()[0], 24, decoding=DecodeConfig(cycle_length=4, verifier="rejection"),
                          selector=TopPSampler(1.0, seed=0))
    assert result.tokens.numel() == 24


def test_lossy_decoding_commits_only_draft_vocabulary_tokens() -> None:
    model = _model(make_tiny_model(), cycle_length=4)
    allowed = model.techniques["frspec"].allowed()
    result = model.decode(_prompts()[0], 30, decoding=DecodeConfig(cycle_length=4, lossless=False))
    drafted = [t for i, t in enumerate(result.tokens.tolist()) if i % 4 != 0]  # every 4th token is exact
    assert allowed[torch.tensor(drafted)].all()


@pytest.mark.parametrize("inner", INNERS)
def test_the_full_vocabulary_equals_the_inner_drafter(inner: str) -> None:
    """With ``K = V`` the wrapper changes nothing: tokens, acceptance and the cost log equal the inner drafter's."""
    base = make_tiny_model()
    wrapped = _model(base, INNERS[inner], size=VOCAB, ids=tuple(range(VOCAB)))
    plain = economize.apply(make_tiny_model(), INNERS[inner], decoding=DecodeConfig(cycle_length=4))
    plain.techniques[inner].load_state_dict(wrapped.techniques["frspec"].inner.state_dict())
    for prompt in _prompts():
        a, b = wrapped.decode(prompt, 30), plain.decode(prompt, 30)
        assert torch.equal(a.tokens, b.tokens) and a.acceptance == b.acceptance
        assert cost_events(a.cost) == cost_events(b.cost)
        assert a.cost.events == b.cost.events


# ---- persistence and combination -----------------------------------------------------------------------------------
def test_save_and_load_round_trip(tmp_path, calib: torch.Tensor) -> None:
    model = _model(make_tiny_model(), calibration=calib)
    economize.save(model, tmp_path)
    loaded = economize.load(make_tiny_model(), tmp_path)  # the draft ids come from the saved state
    technique = loaded.techniques["frspec"]
    assert technique.config == model.techniques["frspec"].config
    state = model.techniques["frspec"].state_dict()
    assert state.keys() == technique.state_dict().keys() and "draft_ids" in state
    for name, value in technique.state_dict().items():
        assert torch.equal(value, state[name]), name
    prompt = _prompts()[2]
    first, second = model.decode(prompt, 24), loaded.decode(prompt, 24)
    assert torch.equal(first.tokens, second.tokens) and first.acceptance == second.acceptance


def test_frspec_on_a_pruned_model_equals_its_greedy(calib: torch.Tensor) -> None:
    from economize.techniques.prune import PruneConfig

    base = make_tiny_model()
    model = economize.apply(base, PruneConfig(method="wanda", sparsity="2:4"), FRSpecConfig(drafter=JUMP, vocab_size=K),
                         decoding=DecodeConfig(cycle_length=4), calibration=calib)
    perturb(model.techniques["frspec"].inner, std=0.05, seed=3)
    assert list(model.techniques) == ["prune", "frspec"]
    assert model.techniques["frspec"].draft_ids.tolist() == frequency_order(calib, VOCAB)[:K].tolist()
    for prompt, result in zip(_prompts(), model.decode_batch(_prompts(), 24)):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 24))


def test_frspec_inside_a_hybrid_fallback() -> None:
    base = make_tiny_model()
    model = economize.apply(base, HybridConfig(fallback=FRSpecConfig(drafter=JUMP, vocab_size=K, token_ids=_ids())),
                         decoding=DecodeConfig(cycle_length=4))
    perturb(model.techniques["hybrid"].fallback.inner, std=0.05, seed=3)
    for prompt, result in zip(_prompts(), model.decode_batch(_prompts(), 24)):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 24))


def test_draft_modules_and_graph_safety_follow_the_inner_technique() -> None:
    for inner in INNERS:
        technique = _model(make_tiny_model(), INNERS[inner]).techniques["frspec"]
        inner_technique = technique.inner
        own = (inner_technique.draft_modules() if isinstance(inner_technique, DraftModules)
               else dict(inner_technique.named_modules()))  # the whole tree, as for an unwrapped technique
        assert set(technique.draft_modules()) == {f"inner.{name}" for name in own if name}
        assert technique.graph_safe == technique.inner.graph_safe
        assert "vocab_size=64" in repr(technique)


# ---- sliced heads --------------------------------------------------------------------------------------------------
SLICEABLE = {
    "jump": JUMP,
    "early_exit": EarlyExitConfig(exit_layer=3, head="readout"),
    "vendor_mtp": VendorMTPConfig(num_modules=2, train_steps=2),
}


class _Recording(DelegatingDrafter):
    """Records the tokens of every proposal of the wrapped drafter."""

    def __init__(self, inner: FRSpecDrafter) -> None:
        super().__init__(inner, "test")
        self.drafts: list[torch.Tensor] = []

    def propose(self, request: DraftRequest):
        draft = self.inner.propose(request)
        self.drafts.append(draft.tokens.clone())
        return draft


def test_widen_logits_places_the_columns_at_the_ids() -> None:
    ids = torch.tensor([5, 1, 3])
    out = widen_logits(torch.tensor([[1.0, 2.0, 3.0]]), ids, 6)
    assert out[0, 5] == 1 and out[0, 1] == 2 and out[0, 3] == 3
    assert torch.isinf(out[0, [0, 2, 4]]).all()


def test_the_adapter_head_slice_equals_the_columns_of_the_full_logits() -> None:
    adapter = adapter_for(make_tiny_model())
    ids = torch.tensor(_ids())
    hidden = torch.randn(2, 3, adapter.hidden_size)
    head = adapter_head_slice(adapter, ids)
    assert head is not None
    # a GEMM over a row subset of the head may round differently from the full GEMM, hence fp32 tolerances
    torch.testing.assert_close(head(hidden), adapter.logits(hidden)[..., ids])


@pytest.mark.parametrize("family", ["gemma2", "gemma3_text", "granite"])
def test_the_head_slice_applies_the_family_logit_transform(family: str) -> None:
    """Gemma's softcap and Granite's ``logits_scaling`` act on every logit on their own
    (:meth:`~economize.arch.ArchAdapter.transform_logits`), so these heads slice too, with the same numbers."""
    adapter = adapter_for(make_family_model(family))
    assert type(adapter).transform_logits is not ArchAdapter.transform_logits
    ids = torch.tensor(_ids())
    hidden = torch.randn(2, 3, adapter.hidden_size)
    head = adapter_head_slice(adapter, ids)
    assert head is not None
    full = adapter.logits(hidden)
    torch.testing.assert_close(head(hidden), full[..., ids])
    raw = adapter.lm_head(adapter.final_norm(hidden))
    assert not torch.equal(full, raw)  # the transform is applied


def test_heads_that_cannot_be_sliced_fall_back_to_masking() -> None:
    class Patched(torch.nn.Linear):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return super().forward(x) * 2

    assert not LinearSlice.supports(Patched(4, 8))
    base = make_tiny_model()
    base.lm_head.__class__ = Patched
    model = _model(base)
    drafter = model.techniques["frspec"].drafter(model.runner)
    assert not drafter.sliced


@pytest.mark.parametrize("inner", SLICEABLE)
def test_sliced_heads_draft_the_tokens_of_masked_logits(inner: str) -> None:
    """A drafter whose head computes the draft ids only proposes the same tokens as one whose full logits are
    masked, and decoding stays lossless."""
    base = make_tiny_model()
    model = _model(base, SLICEABLE[inner])
    technique = model.techniques["frspec"]
    sliced = technique.drafter(model.runner)
    assert sliced.sliced
    masked = FRSpecDrafter(technique.inner.drafter(model.runner), technique.allowed())
    assert not masked.sliced
    prompts, budgets = _prompts(), [24, 18, 21]
    drafts = []
    for drafter in (sliced, masked):
        recording = _Recording(drafter)
        decoder = CycleDecoder(model.runner, recording, DecodeConfig(cycle_length=4))
        results = decoder.decode_batch(prompts, budgets)
        for prompt, result, new in zip(prompts, results, budgets):
            assert torch.equal(result.tokens, reference_greedy(base, prompt, new))
        drafts.append(recording.drafts)
    assert len(drafts[0]) == len(drafts[1]) > 0
    for a, b in zip(*drafts):
        assert torch.equal(a, b)


def test_published_token_lists_load(tmp_path) -> None:
    """``freq_32768.pt`` of thunlp/LLaMA3-Instruct-8B-FR-Spec is a saved Python list of ids sorted by id."""
    ids = list(_ids())
    torch.save(sorted(ids), tmp_path / "freq.pt")
    torch.save(torch.tensor(ids), tmp_path / "tensor.pt")
    assert read_token_ids(tmp_path / "freq.pt") == tuple(sorted(ids))
    assert read_token_ids(tmp_path / "tensor.pt") == tuple(ids)
    torch.save([1, 1, 2], tmp_path / "bad.pt")
    with pytest.raises(ConfigError):
        read_token_ids(tmp_path / "bad.pt")
    base = make_tiny_model()
    model = _model(base, ids=read_token_ids(tmp_path / "freq.pt"))
    assert torch.equal(model.techniques["frspec"].draft_ids, torch.tensor(sorted(ids)))


# ---- FR-Spec checks the contract of the drafter it wraps ----
SCRIPT = [torch.arange(40) % VOCAB] * 2


def _allowed() -> torch.Tensor:
    allowed = torch.zeros(VOCAB, dtype=torch.bool)
    allowed[: VOCAB // 2] = True
    return allowed


def _budget_request(budgets: list[int]) -> DraftRequest:
    config = DecodeConfig()
    seq = torch.zeros(len(budgets), 32, dtype=torch.long)
    return DraftRequest(seq, [4] * len(budgets), budgets, selector=GreedySelector(), verifier=GreedyVerifier(config),
                        policy=FixedLength(config), costs=[CostLog() for _ in budgets], layer_span=(1, 1))


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


class _Copying(ScriptedDrafter):
    """Writes ``VOCAB - 1`` (outside the draft vocabulary) into the sequence itself, like prompt lookup."""

    def propose(self, request: DraftRequest) -> Draft:
        tokens = torch.full((request.batch_size, 2), VOCAB - 1)
        return chain_draft(request, tokens, [2] * request.batch_size)


class _Mixed(ScriptedDrafter):
    """Drafts one token through ``extend``, then copies one more for row 0 into the sequence itself."""

    def propose(self, request: DraftRequest) -> Draft:
        request.extend(self.step(request.tokens, request.positions, request.valid))
        request.seq[0, request.ends[0] + 1] = VOCAB - 1
        request.lengths[0] += 1
        return request.draft()


def test_frspec_refuses_a_tree_drafter_before_it_drafts() -> None:
    inner = _Tree()
    drafter = FRSpecDrafter(inner, _allowed())
    drafter.begin(None, None, 2)
    with pytest.raises(ConfigError, match="tree"):
        drafter.propose(_budget_request([3, 3]))
    assert inner.calls == 0


@pytest.mark.parametrize("inner", [_Copying, _Mixed])
def test_frspec_rejects_tokens_that_bypass_extend(inner: type[ScriptedDrafter]) -> None:
    drafter = FRSpecDrafter(inner(SCRIPT), _allowed())
    drafter.begin(None, None, 2)
    with pytest.raises(ConfigError, match="extend"):
        drafter.propose(_budget_request([3, 3]))


def test_frspec_accepts_a_drafter_that_extends() -> None:
    drafter = FRSpecDrafter(ScriptedDrafter(SCRIPT), _allowed())
    drafter.begin(None, None, 2)
    draft = drafter.propose(_budget_request([3, 2]))
    assert draft.lengths == [3, 2] and _allowed()[draft.tokens[draft.tokens >= 0]].all()
