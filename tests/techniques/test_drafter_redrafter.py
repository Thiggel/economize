"""ReDrafter: the beam tree, lossless chains and beam trees, sampling, costs, checkpoints and training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

import economize
from economize import ConfigError, DecodeConfig, TopPSampler
from economize.core.serialization import read_checkpoint
from economize.core.technique import TECHNIQUES
from economize.techniques.redrafter import LOG_0, RedrafterConfig, RedrafterTechnique, beam_tree
from economize.train import ObjectiveModule, default_objective, fit
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, perturb, reference_greedy


def _model(head: str = "own", beam_width: int = 1, beam_length: int = 5, std: float = 0.05,
           base: torch.nn.Module | None = None, **decoding: Any) -> Any:
    config = RedrafterConfig(num_layers=2, head=head, beam_width=beam_width, beam_length=beam_length, top_k=20)
    model = economize.apply(base if base is not None else make_tiny_model(), config,
                            decoding=DecodeConfig(**decoding) if decoding else None)
    if std:
        perturb(model.techniques["redrafter"], std, seed=1)
    return model


def _prompts(seed: int = 1, lengths: tuple[int, ...] = (PROMPT_LEN, 7, 20)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in lengths]


def _assert_greedy(model: Any, base: torch.nn.Module, new: int = 24, **kwargs: Any) -> list[Any]:
    prompts = _prompts()
    results = model.decode_batch(prompts, new, **kwargs)
    for prompt, result in zip(prompts, results):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, new))
    return results


# ---- configuration -------------------------------------------------------------------------------------------------
def test_registration_and_config() -> None:
    assert TECHNIQUES.get("redrafter") is RedrafterTechnique
    for config in (RedrafterConfig(), RedrafterConfig(num_layers=0, head="shared", beam_width=8, exit_dim=96),
                   RedrafterConfig(beam_width=45, beam_length=5, top_k=0, param_dtype="model")):
        assert RedrafterConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    for bad in (dict(num_layers=-1), dict(head="tied"), dict(beam_width=0), dict(beam_length=1),
                dict(top_k=4, beam_width=8), dict(exit_dim=0), dict(input_init="random"), dict(param_dtype="bfloat16")):
        with pytest.raises(ConfigError):
            RedrafterConfig(**bad)


def test_beam_tree_keeps_distinct_prefixes() -> None:
    beams = torch.tensor([[[5, 6, 7], [5, 6, 8], [5, 9, 7], [5, 6, 7]],
                          [[1, 2, 3], [4, 2, 3], [1, 2, 3], [1, 5, 3]]])
    tokens, lengths, parents = beam_tree(beams, [3, 2])
    # row 0: depth 1 {5}, depth 2 {56, 59}, depth 3 {567, 568, 597}
    assert lengths == [6, 5]
    assert tokens[0].tolist() == [5, 6, 9, 7, 8, 7]
    assert parents[0].tolist() == [-1, 0, 0, 1, 1, 2]
    # row 1 cut at depth 2: depth 1 {1, 4}, depth 2 {12, 42, 15}
    assert tokens[1].tolist() == [1, 4, 2, 2, 5, -1][: tokens.shape[1]]
    assert parents[1, :5].tolist() == [-1, -1, 0, 1, 0]
    assert tokens[1, 5:].eq(-1).all() and parents[1, 5:].eq(-1).all()


@torch.no_grad()
def test_untrained_heads_are_the_model_head() -> None:
    """Zero-initialized blocks and an LM-head (own) or identity (shared) output map read ``h_t`` alone."""
    base = make_tiny_model()
    ids = torch.randint(0, VOCAB, (2, 9), generator=torch.Generator().manual_seed(2))
    out = base(ids, output_hidden_states=True)
    for head in ("own", "shared"):
        technique = _model(head, std=0.0, base=base).techniques["redrafter"]
        context = technique.base(out.hidden_states[-1][:, 4])
        state = technique.recur(torch.zeros_like(context), ids[:, 5])
        logits = technique.head_logits(context, state)
        assert logits.shape == (2, VOCAB) and logits.dtype == torch.float32
        if head == "shared":
            torch.testing.assert_close(logits, out.logits[:, 4].float(), rtol=1e-5, atol=1e-5)
        else:  # own blocks keep PyTorch's default biases, as the reference
            assert torch.isfinite(logits).all()


@torch.no_grad()
def test_select_input_init_starts_at_the_narrow_head() -> None:
    """Under ``input_init="select"`` a widened untrained head equals the narrow one; its extra channels are not zero."""
    base = make_tiny_model()
    ids = torch.randint(0, VOCAB, (2, 9), generator=torch.Generator().manual_seed(2))
    out = base(ids, output_hidden_states=True)
    d = base.config.hidden_size
    for head in ("own", "shared"):
        torch.manual_seed(0)
        narrow = economize.apply(base, RedrafterConfig(head=head)).techniques["redrafter"]
        torch.manual_seed(0)
        wide = economize.apply(base, RedrafterConfig(head=head, exit_dim=3 * d, input_init="select")
                               ).techniques["redrafter"]
        assert wide.input_proj is not None and wide.input_proj.weight[2 * d:].abs().gt(0).any()
        context = wide.base(out.hidden_states[-1][:, 4])
        state = wide.recur(torch.zeros_like(context), ids[:, 5])
        if head == "shared":
            torch.testing.assert_close(wide.head_logits(context, state), out.logits[:, 4].float(),
                                       rtol=1e-5, atol=1e-5)
        else:  # own blocks keep PyTorch's default biases, which are drawn per width
            for block in (*narrow.blocks, *wide.blocks):
                block.linear.bias.zero_()
            torch.testing.assert_close(wide.head_logits(context, state), narrow.head_logits(context, state),
                                       rtol=1e-5, atol=1e-5)
    with pytest.raises(ConfigError):
        economize.apply(base, RedrafterConfig(exit_dim=d, input_init="select"))


def test_warp_and_beam_step() -> None:
    technique = _model("own").techniques["redrafter"]
    logits = torch.randn(3, VOCAB, generator=torch.Generator().manual_seed(0))
    warped = technique.warp(logits)
    assert (warped > LOG_0).sum(-1).tolist() == [20, 20, 20]
    assert torch.equal(warped.argmax(-1), logits.argmax(-1))
    d = technique.hidden_size
    context = torch.randn(2, d)
    log_p = torch.tensor([[0.0, LOG_0, LOG_0], [0.0, LOG_0, LOG_0]])
    tokens = torch.tensor([[3, 3, 3], [4, 4, 4]])
    values, new, beams, state = technique.beam_step(context, torch.zeros(2, 3, d), tokens, log_p)
    assert beams.eq(0).all()  # only the first beam is alive after the root
    step, _ = technique.draft_step(context, torch.zeros(2, d), tokens[:, 0])
    top = step.log_softmax(-1).topk(3, -1)
    assert torch.equal(new, top.indices)
    torch.testing.assert_close(values, top.values)
    assert state.shape == (2, 3, d)


# ---- lossless decoding ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("head", ["own", "shared"])
@pytest.mark.parametrize("cycle_length", [1, 3, 6])
def test_chains_are_greedy(head: str, cycle_length: int) -> None:
    base = make_tiny_model()
    model = _model(head, base=base, cycle_length=cycle_length)
    results = _assert_greedy(model, base)
    if cycle_length > 1:
        assert sum(k for r in results for k, _ in r.acceptance) > 0


@pytest.mark.parametrize("head", ["own", "shared"])
@pytest.mark.parametrize(("beam_width", "cycle_length"), [(2, 2), (2, 5), (4, 4), (8, 6)])
def test_beam_trees_are_greedy(head: str, beam_width: int, cycle_length: int) -> None:
    base = make_tiny_model()
    model = _model(head, beam_width=beam_width, base=base, cycle_length=cycle_length)
    results = _assert_greedy(model, base, new=30)
    assert sum(k for r in results for k, _ in r.acceptance) > 0


def test_untrained_shared_beams_accept_drafts() -> None:
    base = make_tiny_model()
    model = _model("shared", beam_width=4, std=0.0, base=base, cycle_length=4)
    results = _assert_greedy(model, base)
    assert sum(k for r in results for k, _ in r.acceptance) > 0


@pytest.mark.parametrize("beam_width", [1, 4])
def test_rejection_at_near_zero_temperature_is_greedy(beam_width: int) -> None:
    base = make_tiny_model()
    model = _model("own", beam_width=beam_width, base=base)
    config = DecodeConfig(cycle_length=4, verifier="rejection")
    _assert_greedy(model, base, decoding=config, selector=TopPSampler(1e-6, seed=0))


@pytest.mark.parametrize("beam_width", [1, 4])
def test_rejection_sampling_runs(beam_width: int) -> None:
    model = _model("shared", beam_width=beam_width)
    config = DecodeConfig(cycle_length=4, verifier="rejection")
    result = model.decode(_prompts()[0], 24, decoding=config, selector=TopPSampler(1.0, seed=0))
    assert result.tokens.shape == (24,) and int(result.tokens.max()) < VOCAB


def test_dynamic_draft_length_is_lossless() -> None:
    base = make_tiny_model()
    model = _model("own", base=base)
    _assert_greedy(model, base, decoding=DecodeConfig(cycle_length=5, min_confidence=0.6))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("beam_width", [1, 4])
@torch.no_grad()
def test_graphed_drafts_are_greedy_and_replayed(beam_width: int) -> None:
    base = make_tiny_model().cuda()
    model = _model("own", beam_width=beam_width, base=base)
    decoding = DecodeConfig(cycle_length=5, cuda_graphs=True)
    prompts = [p.cuda() for p in _prompts()]
    greedy = [reference_greedy(base, p, 24).tolist() for p in prompts]
    for _ in range(3):  # eager warm-up, capture, replay
        results = model.decode_batch(prompts, 24, decoding=decoding)
        assert [r.tokens.tolist() for r in results] == greedy
    graphs = model.decoder(decoding).graphs
    assert graphs is not None and graphs.num_graphs > 0


# ---- cost accounting -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("beam_width", [1, 4])
def test_cost_log(beam_width: int) -> None:
    model = _model("own", beam_width=beam_width, cycle_length=4)
    result = model.decode(_prompts()[0], 30)
    drafts = [e for e in result.cost.events if e.kind in ("draft", "draft_wasted")]
    assert all((e.tokens, e.first_layer, e.last_layer, e.module) == (beam_width, 1, 0, "redrafter") for e in drafts)
    if beam_width == 1:
        assert len(drafts) == sum(k for k, _ in result.acceptance)
    else:  # one event per beam step, three steps per cycle but in the last cycles
        assert 0 < len(drafts) <= 3 * len(result.acceptance)
    assert not [e for e in result.cost.events if e.kind == "draft_observe"]


# ---- save and load -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("beam_width", [1, 4])
def test_save_load_and_attach_round_trip(beam_width: int, tmp_path: Path) -> None:
    model = _model("shared", beam_width=beam_width, cycle_length=4)
    economize.save(model, tmp_path / "ckpt")
    loaded = economize.load(make_tiny_model(), tmp_path / "ckpt")
    assert loaded.drafting_technique.config == model.drafting_technique.config
    attached = economize.attach(make_tiny_model(), read_checkpoint(tmp_path / "ckpt"))
    prompt = _prompts()[1]
    first, second, third = model.decode(prompt, 20), loaded.decode(prompt, 20), attached.decode(prompt, 20)
    assert torch.equal(first.tokens, second.tokens) and first.acceptance == second.acceptance
    assert torch.equal(first.tokens, third.tokens)


# ---- training ------------------------------------------------------------------------------------------------------
def test_default_objective_is_step_distill() -> None:
    # perturbed: at the init the state's columns of proj are zero, so the recurrence has no gradient yet
    model = _model("own")
    objective = default_objective(model)
    assert type(objective).__name__ == "StepDistill"
    module = ObjectiveModule(model, objective)
    try:
        ids = torch.randint(0, VOCAB, (2, 14), generator=torch.Generator().manual_seed(4))
        with torch.enable_grad():
            out = module(ids)
            out["loss"].backward()
    finally:
        module.close()
    technique = model.techniques["redrafter"]
    for name in ("rnn_w", "rnn_u", "proj"):
        grad = getattr(technique, name).weight.grad
        assert grad is not None and grad.abs().sum() > 0, name
    assert not any(p.requires_grad for p in model.base_model.parameters())


@pytest.mark.parametrize("head", ["own", "shared"])
def test_fit_trains_the_head(head: str) -> None:
    rows = torch.randint(0, VOCAB, (16, 24), generator=torch.Generator().manual_seed(3))
    rows[:, 1::2] = (rows[:, 0::2] + 1) % VOCAB
    model = _model(head, std=0.0)
    with torch.enable_grad():
        result = fit(model, rows, steps=30, batch_size=8, lr=3e-3, warmup_steps=3, seed=0, log_every=10)
    assert result.steps == 30
    assert sum(result.losses[-5:]) < 0.95 * sum(result.losses[:5])
