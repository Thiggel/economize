"""Hydra heads (plain and Hydra++ with the prefix layer): lossless chains and trees, sampling, costs, checkpoints and
training."""

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
from economize.techniques.hydra import HydraConfig, HydraTechnique
from economize.train import ObjectiveModule, default_objective, fit
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, perturb, reference_greedy

TREE = ((0,), (1,), (0, 0), (0, 1), (1, 0), (0, 0, 0))


def _model(head: str = "own", prefix: bool = True, heads: int = 3, tree: Any = None, std: float = 0.05,
           base: torch.nn.Module | None = None, **decoding: Any) -> Any:
    config = HydraConfig(num_heads=heads, num_layers=2, head=head, prefix=prefix, tree=tree)
    model = economize.apply(base if base is not None else make_tiny_model(), config,
                            decoding=DecodeConfig(**decoding) if decoding else None)
    if std:
        perturb(model.techniques["hydra"], std, seed=1)
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
    assert TECHNIQUES.get("hydra") is HydraTechnique
    for config in (HydraConfig(), HydraConfig(num_heads=3, num_layers=1, prefix=False, head="shared", tree=TREE),
                   HydraConfig(num_heads=5, tree="mc_sim_7b_63", prefix_intermediate_size=96)):
        assert HydraConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    for bad in (dict(num_heads=0), dict(num_layers=0), dict(head="tied"), dict(tree="no_such_tree"),
                dict(num_heads=2, tree=[[0], [0, 0], [0, 0, 0]]), dict(prefix_intermediate_size=0),
                dict(param_dtype="bfloat16")):
        with pytest.raises(ConfigError):
            HydraConfig(**bad)


@pytest.mark.parametrize("prefix", [False, True])
@torch.no_grad()
def test_untrained_shared_heads_without_prefix_are_the_model_head(prefix: bool) -> None:
    """Zero-initialized shared blocks with identity input maps leave the base state: every head is the model's own
    next-token distribution at ``t`` (without the prefix layer)."""
    base = make_tiny_model()
    model = _model("shared", prefix=prefix, std=0.0, base=base)
    technique = model.techniques["hydra"]
    ids = torch.randint(0, VOCAB, (2, 9), generator=torch.Generator().manual_seed(2))
    out = base(ids, output_hidden_states=True)
    states = technique.prefix_states(out.hidden_states[-1])
    history = ids[:, :3].reshape(2, 3)
    logits = technique.head_logits(states[:, 4], history)
    assert logits.shape == (2, VOCAB) and logits.dtype == torch.float32
    if not prefix:
        torch.testing.assert_close(logits, out.logits[:, 4].float(), rtol=1e-5, atol=1e-5)


# ---- lossless decoding ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("head", ["own", "shared"])
@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("cycle_length", [1, 3, 6])
def test_chains_are_greedy(head: str, prefix: bool, cycle_length: int) -> None:
    base = make_tiny_model()
    model = _model(head, prefix=prefix, heads=5, base=base, cycle_length=cycle_length)
    results = _assert_greedy(model, base)
    if cycle_length > 1:
        assert sum(k for r in results for k, _ in r.acceptance) > 0


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize(("tree", "cycle_length"), [(TREE, 2), (TREE, 4), ("small", 4), ("mc_sim_7b_63", 8)])
def test_trees_are_greedy(prefix: bool, tree: Any, cycle_length: int) -> None:
    base = make_tiny_model()
    model = _model("own", prefix=prefix, heads=5, tree=tree, base=base, cycle_length=cycle_length)
    results = _assert_greedy(model, base, new=30)
    assert sum(k for r in results for k, _ in r.acceptance) > 0


def test_trained_like_heads_accept_drafts_in_trees() -> None:
    """A head equal to the model's own head at the next position accepts most drafts: copy of shared, no prefix."""
    base = make_tiny_model()
    model = _model("shared", prefix=False, heads=3, tree=TREE, std=0.0, base=base, cycle_length=4)
    results = _assert_greedy(model, base)
    assert all(len(r.tokens) == 24 for r in results)


@pytest.mark.parametrize("tree", [None, TREE])
def test_rejection_at_near_zero_temperature_is_greedy(tree: Any) -> None:
    base = make_tiny_model()
    model = _model("own", tree=tree, base=base)
    config = DecodeConfig(cycle_length=4, verifier="rejection")
    _assert_greedy(model, base, decoding=config, selector=TopPSampler(1e-6, seed=0))


@pytest.mark.parametrize("tree", [None, TREE])
def test_rejection_sampling_runs(tree: Any) -> None:
    model = _model("shared", tree=tree)
    config = DecodeConfig(cycle_length=4, verifier="rejection")
    result = model.decode(_prompts()[0], 24, decoding=config, selector=TopPSampler(1.0, seed=0))
    assert result.tokens.shape == (24,) and int(result.tokens.max()) < VOCAB


def test_dynamic_draft_length_is_lossless() -> None:
    base = make_tiny_model()
    model = _model("own", heads=5, base=base)
    _assert_greedy(model, base, decoding=DecodeConfig(cycle_length=6, min_confidence=0.6))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("tree", [None, "mc_sim_7b_63"])
@torch.no_grad()
def test_graphed_drafts_are_greedy_and_replayed(tree: Any) -> None:
    base = make_tiny_model().cuda()
    model = _model("own", heads=5, tree=tree, base=base)
    decoding = DecodeConfig(cycle_length=5, cuda_graphs=True)
    prompts = [p.cuda() for p in _prompts()]
    greedy = [reference_greedy(base, p, 24).tolist() for p in prompts]
    for _ in range(3):  # eager warm-up, capture, replay
        results = model.decode_batch(prompts, 24, decoding=decoding)
        assert [r.tokens.tolist() for r in results] == greedy
    graphs = model.decoder(decoding).graphs
    assert graphs is not None and graphs.num_graphs > 0


# ---- cost accounting -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("prefix", [False, True])
def test_cost_log(prefix: bool) -> None:
    model = _model("own", prefix=prefix, cycle_length=4)
    result = model.decode(_prompts()[0], 30)
    events = result.cost.events
    drafts = [e for e in events if e.kind in ("draft", "draft_wasted")]
    assert all((e.tokens, e.first_layer, e.last_layer, e.module) == (1, 1, 0, "hydra") for e in drafts)
    assert len(drafts) == sum(k for k, _ in result.acceptance)
    observe = [e for e in events if e.kind == "draft_observe"]
    assert bool(observe) == prefix
    assert all(e.module == "hydra" and e.last_layer - e.first_layer == 0 for e in observe)


# ---- save and load -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("tree", [None, "small"])
def test_save_load_and_attach_round_trip(tree: Any, tmp_path: Path) -> None:
    model = _model("shared", heads=5, tree=tree, cycle_length=4)
    economize.save(model, tmp_path / "ckpt")
    loaded = economize.load(make_tiny_model(), tmp_path / "ckpt")
    assert loaded.drafting_technique.config == model.drafting_technique.config
    checkpoint = read_checkpoint(tmp_path / "ckpt")
    attached = economize.attach(make_tiny_model(), checkpoint)
    prompt = _prompts()[1]
    first, second, third = model.decode(prompt, 20), loaded.decode(prompt, 20), attached.decode(prompt, 20)
    assert torch.equal(first.tokens, second.tokens) and first.acceptance == second.acceptance
    assert torch.equal(first.tokens, third.tokens)


# ---- training ------------------------------------------------------------------------------------------------------
def test_default_objective_is_step_distill() -> None:
    model = _model("own", std=0.0)
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
    grads = [p.grad for p in model.techniques["hydra"].parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    assert not any(p.requires_grad for p in model.base_model.parameters())


@pytest.mark.parametrize(("head", "prefix"), [("own", True), ("shared", False), ("shared", True)])
def test_fit_trains_the_heads(head: str, prefix: bool) -> None:
    rows = torch.randint(0, VOCAB, (16, 24), generator=torch.Generator().manual_seed(3))
    rows[:, 1::2] = (rows[:, 0::2] + 1) % VOCAB
    model = _model(head, prefix=prefix, std=0.0)
    with torch.enable_grad():
        result = fit(model, rows, steps=30, batch_size=8, lr=3e-3, warmup_steps=3, seed=0, log_every=10)
    assert result.steps == 30
    assert sum(result.losses[-5:]) < 0.95 * sum(result.losses[:5])
