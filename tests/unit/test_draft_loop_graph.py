"""Whole-proposal CUDA graphs (``DecodeConfig.draft_loop_graph``) and the fused greedy LM head.

A chain drafter records its ``k`` greedy draft steps as one graph and picks every step's token with the fused head
kernel (:mod:`economize.techniques._drafting.greedy_head`). Both are lossless: in float32 the outputs equal plain
greedy decoding and the drafts, acceptances and cost logs equal the per-step graphed path.

The kernel tests run anywhere (the CPU falls back to the reference); the decoding tests need ``-m gpu``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import economize
from economize import ConfigError, DecodeConfig, JumpConfig
from economize.compat.tskip import cost_events
from economize.decoding import CostLog, DraftRequest, FixedLength, GreedyVerifier
from economize.decoding.selectors import GreedySelector
from economize.techniques._drafting import ChainGraphs, FusedGreedyHead, LogitsGreedyHead, greedy_select
from economize.techniques._drafting.greedy_head import LogitTransform, greedy_select_reference
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.frspec import FRSpecConfig
from helpers import VOCAB, make_tiny_model, perturb, reference_greedy

CUDA = torch.cuda.is_available()
gpu = [pytest.mark.gpu, pytest.mark.skipif(not CUDA, reason="needs a CUDA device")]
DEVICE = "cuda"
MAX_NEW = 20


# ---- the fused head -------------------------------------------------------------------------------------------------
def _head_case(rows: int, vocab: int, dim: int, dtype: torch.dtype, device: str, seed: int = 0) -> tuple[Any, ...]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, dim, generator=generator).to(device, dtype)
    weight = (torch.randn(vocab, dim, generator=generator) * dim**-0.5).to(device, dtype)
    bias = torch.randn(vocab, generator=generator).to(device, dtype)
    return x, weight, bias


TRANSFORMS = {"none": LogitTransform(), "softcap": LogitTransform(cap=30.0), "both": LogitTransform(8.0, 30.0)}


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=gpu)])
@pytest.mark.parametrize("rows", [1, 3, 17])
@pytest.mark.parametrize("transform", TRANSFORMS)
@torch.no_grad()
def test_fused_head_equals_the_reference_in_float32(device: str, rows: int, transform: str) -> None:
    x, weight, bias = _head_case(rows, 3000, 96, torch.float32, device)
    for b in (None, bias):
        tokens, top = greedy_select(x, weight, b, transform=TRANSFORMS[transform])
        ref_tokens, ref_top = greedy_select_reference(x, weight, b, transform=TRANSFORMS[transform])
        assert torch.equal(tokens, ref_tokens)
        torch.testing.assert_close(top, ref_top, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("sort", [True, False])
@torch.no_grad()
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=gpu)])
def test_fused_head_over_a_draft_vocabulary_returns_model_ids(device: str, sort: bool) -> None:
    x, weight, bias = _head_case(5, 2000, 64, torch.float32, device)
    ids = torch.randperm(2000, generator=torch.Generator().manual_seed(1))[:300].to(device)
    ids = ids.sort().values if sort else ids
    head = FusedGreedyHead(lambda h: h, weight[ids], bias[ids], token_ids=ids)
    tokens, top = head.select(x)
    widened = torch.full((5, 2000), float("-inf"), device=device)
    widened[:, ids] = x @ weight[ids].T + bias[ids]
    assert torch.equal(tokens, widened.argmax(-1))  # ties go to the smallest model id, as over the full vocabulary
    torch.testing.assert_close(top, torch.softmax(widened, -1).amax(-1), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=gpu)])
@torch.no_grad()
def test_ties_go_to_the_first_index(device: str) -> None:
    x = torch.ones(2, 16, device=device)
    weight = torch.zeros(1000, 16, device=device)
    weight[[700, 5, 900]] = 1.0
    tokens, top = greedy_select(x, weight)
    assert tokens.tolist() == [5, 5]
    torch.testing.assert_close(top, torch.softmax(x @ weight.T, -1).amax(-1))


@pytest.mark.parametrize("rows", [1, 4, 16])
@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
@torch.no_grad()
def test_fused_head_in_bfloat16_is_within_rounding(rows: int) -> None:
    x, weight, _ = _head_case(rows, 32768, 512, torch.bfloat16, DEVICE, seed=2)
    transform = TRANSFORMS["softcap"]
    tokens, top = greedy_select(x, weight, transform=transform)
    logits = (x.float() @ weight.float().T)
    logits = (torch.tanh(logits / 30.0) * 30.0)
    picked = logits.gather(1, tokens[:, None])[:, 0]
    assert (logits.amax(-1) - picked).max() < 0.25  # a near-tie at most a bf16 step apart
    ref_tokens, ref_top = greedy_select_reference(x, weight, transform=transform)
    assert (tokens == ref_tokens).float().mean() >= 0.9
    torch.testing.assert_close(top, ref_top, rtol=0.05, atol=0.02)


def test_logits_head_matches_greedy_selection() -> None:
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(50, 8, generator=generator)
    ids = torch.tensor([40, 3, 17, 9])
    head = LogitsGreedyHead(lambda h: h @ weight[ids].T, token_ids=ids)
    h = torch.randn(6, 8, generator=generator)
    tokens, top = head.select(h)
    widened = torch.full((6, 50), float("-inf"))
    widened[:, ids] = h @ weight[ids].T
    assert torch.equal(tokens, widened.argmax(-1))
    torch.testing.assert_close(top, torch.softmax(widened, -1).amax(-1))


# ---- DraftRequest: k steps at once ----------------------------------------------------------------------------------
def _request(budgets: list[int], flag: bool = True, record_confidence: bool = False) -> DraftRequest:
    seq = torch.zeros(len(budgets), 32, dtype=torch.long)
    seq[:, :4] = torch.arange(1, 5)
    config = DecodeConfig()
    return DraftRequest(seq, [4] * len(budgets), budgets, selector=GreedySelector(), verifier=GreedyVerifier(config),
                        policy=FixedLength(config), costs=[CostLog() for _ in budgets], layer_span=(1, 1),
                        record_confidence=record_confidence, draft_loop_graph=flag)


def test_graphed_chain_steps_needs_the_flag_and_equal_budgets() -> None:
    assert _request([3, 3]).graphed_chain_steps() == 3
    assert _request([3, 3], flag=False).graphed_chain_steps() is None
    assert _request([3, 2]).graphed_chain_steps() is None
    assert _request([0, 0]).graphed_chain_steps() is None
    request = _request([3, 3])
    request.extend(torch.randn(2, VOCAB))
    assert request.graphed_chain_steps() is None  # rows already drafted


def test_extend_tokens_equals_k_extend_calls() -> None:
    steps, generator = 3, torch.Generator().manual_seed(0)
    logits = [torch.randn(2, VOCAB, generator=generator) for _ in range(steps)]
    stepwise, chained = _request([steps] * 2, record_confidence=True), _request([steps] * 2, record_confidence=True)
    for row in logits:
        stepwise.extend(row)
    tokens = torch.stack([row.argmax(-1) for row in logits], 1)
    confidence = torch.stack([torch.softmax(row, -1).amax(-1) for row in logits], 1)
    chained.extend_tokens(tokens, confidence)
    a, b = stepwise.draft(), chained.draft()
    assert torch.equal(a.tokens, b.tokens) and a.lengths == b.lengths
    torch.testing.assert_close(a.confidence, b.confidence)
    assert torch.equal(stepwise.seq, chained.seq) and stepwise.positions.tolist() == chained.positions.tolist()
    assert [cost_events(c) for c in stepwise.costs] == [cost_events(c) for c in chained.costs]
    with pytest.raises(ConfigError):
        _request([steps] * 2).extend_tokens(tokens[:, :2])
    with pytest.raises(ConfigError):
        _request([steps] * 2, record_confidence=True).extend_tokens(tokens)


# ---- decoding with chain graphs -------------------------------------------------------------------------------------
JUMP = {
    "frozen": dict(target_layer=2, source_layer=4),
    "lora": dict(target_layer=2, source_layer=4, adaptation="lora", lora_rank=4),
    "full_readout": dict(target_layer=1, source_layer=3, exit_layer=4, adaptation="full"),
    "layer_attention": dict(target_layer=2, source_layer=5, extra_source_layers=(1, 3), mixer="layer_attention",
                            mixer_heads=4),
}
EAGLE3 = {"own": Eagle3Config(), "own_vocab": Eagle3Config(draft_vocab=96), "shared": Eagle3Config(head="shared")}


def _ids(size: int, seed: int = 5) -> torch.Tensor:
    return torch.randperm(VOCAB, generator=torch.Generator().manual_seed(seed))[:size]


def _build(kind: str, variant: str) -> tuple[economize.EfficientModel, Any]:
    base = make_tiny_model().to(DEVICE)
    if kind == "jump":
        model = economize.apply(base, JumpConfig(**JUMP[variant]))
        technique = model.techniques["jump"]
    elif kind == "eagle3":
        config = EAGLE3[variant]
        model = economize.apply(base, config)
        technique = model.techniques["eagle3"]
        if config.draft_vocab is not None:
            technique.set_draft_vocab(_ids(config.draft_vocab, seed=3))
    else:
        model = economize.apply(base, FRSpecConfig(drafter=JumpConfig(**JUMP[variant]), vocab_size=128,
                                                   token_ids=tuple(_ids(128).tolist())))
        technique = model.techniques["frspec"].inner
    perturb(technique, 0.05, seed=1)
    return model, technique


def _prompts(lengths: tuple[int, ...] = (13, 9, 11)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    return [torch.randint(3, VOCAB, (n,), generator=generator).to(DEVICE) for n in lengths]


def _chain_drafter(decoder: Any) -> ChainGraphs:
    drafter = decoder.active_drafter
    while not isinstance(drafter, ChainGraphs):
        drafter = drafter.inner
    return drafter


def _decode(model: Any, prompts: list[torch.Tensor], **config: Any) -> list[Any]:
    decoding = DecodeConfig(cycle_length=4, cuda_graphs=True, **config)
    for _ in range(3):  # eager warm-up, capture, replay
        results = model.decode_batch(prompts, MAX_NEW, decoding=decoding)
    return results


def _summary(results: list[Any]) -> list[tuple[Any, ...]]:
    return [(r.tokens.tolist(), r.acceptance, cost_events(r.cost)) for r in results]


CASES = [("jump", v) for v in JUMP] + [("eagle3", v) for v in EAGLE3] + [("frspec", "frozen"), ("frspec", "lora")]


@pytest.mark.parametrize(("kind", "variant"), CASES)
@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
@torch.no_grad()
def test_chain_graphs_are_lossless_and_draft_as_the_step_graphs(kind: str, variant: str) -> None:
    model, _ = _build(kind, variant)
    prompts = _prompts()
    greedy = [reference_greedy(model.base_model, p, MAX_NEW).tolist() for p in prompts]
    stepwise = _summary(_decode(model, prompts))
    for fused in (True, False):
        decoding = DecodeConfig(cycle_length=4, cuda_graphs=True, draft_loop_graph=True)
        drafter = _chain_drafter(model.decoder(decoding))
        drafter.fused_head = fused
        for _ in range(3):
            chained = _summary(model.decode_batch(prompts, MAX_NEW, decoding=decoding))
            assert [tokens for tokens, *_ in chained] == greedy
            assert chained == stepwise
        assert (3, fused) in drafter.__dict__.get("_chains", {})  # the chain ran (three drafts per cycle)


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
@torch.no_grad()
def test_chain_graphs_keep_the_confidence() -> None:
    model, _ = _build("jump", "frozen")
    prompts = _prompts()
    stepwise = _decode(model, prompts, record_confidence=True)
    chained = _decode(model, prompts, record_confidence=True, draft_loop_graph=True)
    assert _summary(chained) == _summary(stepwise)
    for a, b in zip(stepwise, chained):
        torch.testing.assert_close(torch.tensor(a.confidence), torch.tensor(b.confidence), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("config", [dict(min_confidence=0.3), dict(lossless=False)])
@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
@torch.no_grad()
def test_lossy_and_dynamic_decoding_agree_with_the_step_graphs(config: dict[str, Any]) -> None:
    model, _ = _build("jump", "lora")
    prompts = _prompts()
    stepwise = _summary(_decode(model, prompts, **config))
    chained = _summary(_decode(model, prompts, draft_loop_graph=True, **config))
    assert chained == stepwise
