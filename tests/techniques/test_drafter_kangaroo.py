"""Kangaroo: the shallow sub-network plus an attention adapter as a self-drafter (lossless decoding, the draft step
against the training rollout, training, costs, checkpoints and CUDA graphs)."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import economize
from economize import ConfigError, DecodeConfig
from economize.core.technique import TECHNIQUES
from economize.techniques.kangaroo import KangarooConfig, KangarooDrafter, KangarooTechnique
from economize.train import ObjectiveModule, StepDistill, fit
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, perturb, reference_greedy

LAYERS = 6


def _model(config: KangarooConfig | None = None, std: float = 0.05, base: torch.nn.Module | None = None,
           **decoding: Any) -> Any:
    model = economize.apply(base if base is not None else make_tiny_model(), config or KangarooConfig(),
                            decoding=DecodeConfig(**decoding) if decoding else None)
    if std:
        perturb(model.techniques["kangaroo"], std, seed=1)
    return model


def _prompts(seed: int = 1, lengths: tuple[int, ...] = (PROMPT_LEN, 7, 20)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in lengths]


def _assert_greedy(model: Any, base: torch.nn.Module, prompts: list[torch.Tensor] | None = None, new: int = 24,
                   **kwargs: Any) -> list[Any]:
    prompts = prompts if prompts is not None else _prompts()
    results = model.decode_batch(prompts, new, **kwargs)
    for prompt, result in zip(prompts, results):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, new))
    return results


@pytest.fixture(scope="module")
def trained() -> SimpleNamespace:
    """An adapter fitted for 60 steps (``step_distill``) to the tiny model's greedy continuations of four prompts."""
    generator = torch.Generator().manual_seed(1)
    prompts = [torch.randint(1, VOCAB, (PROMPT_LEN,), generator=generator) for _ in range(4)]
    base = make_tiny_model()
    with torch.no_grad():
        rows = torch.stack([torch.cat([p, reference_greedy(base, p, 28)]) for p in prompts])
    model = economize.apply(make_tiny_model(), KangarooConfig(exit_layer=3), decoding=DecodeConfig(cycle_length=4))
    with torch.enable_grad():
        result = fit(model, rows, StepDistill(steps=3, step_decay=1.0), steps=60, batch_size=4, lr=1e-2,
                     warmup_steps=0, seed=0, log_every=1)
    return SimpleNamespace(model=model, base=base, prompts=prompts, rows=rows, result=result)


# ---- configuration -------------------------------------------------------------------------------------------------
def test_registration_and_validation() -> None:
    assert TECHNIQUES.get("kangaroo") is KangarooTechnique
    for bad in (dict(exit_layer=0), dict(num_heads=0), dict(num_heads=4, num_kv_heads=3), dict(rms_norm_eps=0.0),
                dict(param_dtype="int8")):
        with pytest.raises((ConfigError, ValueError)):
            KangarooConfig(**bad)
    with pytest.raises(ConfigError, match="exit_layer"):
        economize.apply(make_tiny_model(), KangarooConfig(exit_layer=LAYERS))


def test_the_adapter_takes_the_models_geometry_and_only_it_is_trainable() -> None:
    model = _model(std=0.0)
    technique = model.techniques["kangaroo"]
    net = technique.adapter_net
    assert (net.num_heads, net.num_kv_heads, net.head_dim) == (4, 2, 16)
    assert technique.layer_span == (1, 2) and "l=2" in repr(technique)
    assert technique.dtype == torch.float32
    names = {name.split(".")[1] for name, _ in technique.named_parameters()}
    assert names == {"input_layernorm", "q_proj", "k_proj", "v_proj", "o_proj", "norm"}
    groups = technique.step_parameter_groups()
    assert len(groups) == 1 and sum(p.numel() for p in groups[0].params) == sum(
        p.numel() for p in technique.parameters())
    # the output norm starts as the model's final norm
    torch.testing.assert_close(net.norm.weight, model.base_model.model.norm.weight)
    custom = economize.apply(make_tiny_model(), KangarooConfig(num_heads=2, num_kv_heads=1)).techniques["kangaroo"]
    assert (custom.adapter_net.num_heads, custom.adapter_net.num_kv_heads) == (2, 1)


# ---- lossless decoding ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("exit_layer", [1, 2, 4])
@pytest.mark.parametrize("cycle_length", [1, 3, 6])
def test_greedy_chains_equal_plain_greedy(exit_layer: int, cycle_length: int) -> None:
    base = make_tiny_model()
    model = _model(KangarooConfig(exit_layer=exit_layer), base=base, cycle_length=cycle_length)
    results = _assert_greedy(model, base)
    assert all(k <= cycle_length - 1 for r in results for k, _ in r.acceptance)


@pytest.mark.parametrize("cycle_length", [3, 6])
def test_a_trained_drafter_is_accepted_and_lossless(trained: SimpleNamespace, cycle_length: int) -> None:
    results = _assert_greedy(trained.model, trained.base, trained.prompts,
                             decoding=DecodeConfig(cycle_length=cycle_length))
    assert sum(a for r in results for _, a in r.acceptance) > 0


def test_ragged_batch(trained: SimpleNamespace) -> None:
    prompts = [row[:n] for row, n in zip(trained.rows, (12, 17, 9, 22))]
    results = _assert_greedy(trained.model, trained.base, prompts, new=18)
    assert sum(a for r in results for _, a in r.acceptance) > 0


def test_the_second_early_exit_is_lossless(trained: SimpleNamespace) -> None:
    """Kangaroo's confidence stop (0.6, six speculative steps) is the dynamic draft length."""
    results = _assert_greedy(trained.model, trained.base, trained.prompts,
                             decoding=DecodeConfig(cycle_length=7, min_confidence=0.6))
    assert any(k < 6 for r in results for k, _ in r.acceptance)


def test_lossy_decoding_keeps_the_drafts(trained: SimpleNamespace) -> None:
    result = trained.model.decode(trained.prompts[0], 21, decoding=DecodeConfig(cycle_length=4, lossless=False))
    assert result.tokens.shape == (21,) and all(a == k for k, a in result.acceptance)


def test_cost_log(trained: SimpleNamespace) -> None:
    """A draft step is layers ``1..l`` of the model on one token, preceded by one adapter layer of the module
    ``kangaroo``; the adapter's writes after every exact pass are one-layer observe events of that module."""
    result = trained.model.decode(trained.prompts[1], 30)
    events = result.cost.events
    assert {e.kind for e in events} <= {"exact", "draft", "draft_wasted", "draft_observe", "kangaroo_adapter"}
    drafts = [e for e in events if e.kind in ("draft", "draft_wasted")]
    assert drafts and all((e.first_layer, e.last_layer, e.tokens) == (1, 3, 1) for e in drafts)
    observe = [e for e in events if e.kind == "draft_observe"]
    assert observe and all(e.module == "kangaroo" and e.layer_tokens == e.tokens for e in observe)
    assert all((e.first_layer, e.last_layer) == (1, LAYERS) for e in events if e.kind == "exact")
    assert len(drafts) == sum(k for k, _ in result.acceptance)
    adapter = [(i, e) for i, e in enumerate(events) if e.kind == "kangaroo_adapter"]
    assert len(adapter) == len(drafts)
    assert all(e.module == "kangaroo" and e.layer_tokens == 1 and events[i + 1].kind in ("draft", "draft_wasted")
               and events[i + 1].step == e.step for i, e in adapter)


# ---- the drafter and the training rollout agree ---------------------------------------------------------------------
@pytest.mark.parametrize("split", [(0, 13), (0, 9, 13)])
@torch.no_grad()
def test_drafter_steps_equal_the_training_steps(split: tuple[int, ...]) -> None:
    """After exact passes over positions ``0..12``, the draft step on the true token at ``12 + i`` gives the logits
    of training step ``i`` from start 12."""
    model = _model(KangarooConfig(exit_layer=3))
    technique, runner = model.techniques["kangaroo"], model.runner
    ids = torch.randint(0, VOCAB, (2, 17), generator=torch.Generator().manual_seed(5))
    states = model.base_model(ids, output_hidden_states=True).hidden_states
    teacher = SimpleNamespace(hidden={3: states[3]})
    expected = technique.step_states(model, None, teacher, torch.tensor([12]), 3)
    cache = runner.new_cache(64, 2)
    drafter = technique.drafter(runner)
    assert isinstance(drafter, KangarooDrafter) and drafter.source_layers == (3,)
    drafter.begin(runner, cache, 2)
    for start, end in pairwise(split):
        positions = torch.arange(start, end).expand(2, -1)
        exact = runner.forward(ids[:, start:end], runner.context(cache, positions, trusted=True), capture=(3,))
        drafter.observe(exact, torch.full((2,), end - start - 1))
    for i in range(1, 4):
        logits = drafter.step(ids[:, 12 + i], torch.full((2,), 12 + i), None)
        torch.testing.assert_close(logits, technique.step_logits(expected[i - 1][:, 0], i), rtol=1e-5, atol=1e-5)


def test_step_before_begin_is_an_error() -> None:
    model = _model()
    drafter = model.techniques["kangaroo"].drafter(model.runner)
    with pytest.raises(RuntimeError, match="begin"):
        drafter.step(torch.tensor([1]), torch.tensor([0]), None)


# ---- training ------------------------------------------------------------------------------------------------------
def test_training_decreases_the_loss_and_touches_the_adapter_only(trained: SimpleNamespace) -> None:
    result = trained.result
    assert result.steps == 60 and result.losses[-1] < result.losses[0]
    for a, b in zip(trained.model.base_model.parameters(), make_tiny_model().parameters()):
        assert torch.equal(a, b)


def test_default_objective_trains_the_adapter() -> None:
    model = _model()
    ids = torch.randint(0, VOCAB, (2, 14), generator=torch.Generator().manual_seed(4))
    module = ObjectiveModule(model, StepDistill(steps=2))
    try:
        with torch.enable_grad():
            out = module(ids)
            out["loss"].backward()
    finally:
        module.close()
    assert {"kl1", "kl2"} <= set(out["metrics"])
    grads = [p.grad for p in model.techniques["kangaroo"].parameters()]
    assert all(g is not None for g in grads) and any(g.abs().sum() > 0 for g in grads)


# ---- checkpoints ---------------------------------------------------------------------------------------------------
def test_checkpoint_round_trip_through_read_checkpoint_and_attach(trained: SimpleNamespace, tmp_path: Path) -> None:
    """bench31.py --ckpts loads a drafter as read_checkpoint + attach."""
    model, base, prompts = trained.model, trained.base, trained.prompts
    economize.save(model, tmp_path / "final")
    checkpoint = economize.read_checkpoint(tmp_path / "final")
    assert [c.technique for c in checkpoint.configs] == ["kangaroo"]
    loaded = economize.attach(make_tiny_model(), economize.Checkpoint(list(checkpoint.configs), checkpoint.states))
    for (name, a), (_, b) in zip(model.techniques["kangaroo"].state_dict().items(),
                                 loaded.techniques["kangaroo"].state_dict().items()):
        assert torch.equal(a, b), name
    config = DecodeConfig(cycle_length=4)
    first, second = model.decode(prompts[1], 20, decoding=config), loaded.decode(prompts[1], 20, decoding=config)
    assert torch.equal(first.tokens, second.tokens) and first.acceptance == second.acceptance
    assert torch.equal(second.tokens, reference_greedy(base, prompts[1], 20))


# ---- CUDA graphs ---------------------------------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@torch.no_grad()
def test_graphed_draft_steps_match_eager_and_are_replayed(trained: SimpleNamespace) -> None:
    """Graphed draft steps (persistent adapter cache, full-capacity reads) draft exactly what eager steps draft,
    also when rows leave the batch and when a longer generation outgrows the cache."""
    base = make_tiny_model().cuda()
    model = economize.apply(make_tiny_model().cuda(), KangarooConfig(exit_layer=3))
    model.techniques["kangaroo"].load_state_dict(trained.model.techniques["kangaroo"].state_dict())
    prompts = [p.cuda() for p in _prompts(lengths=(PROMPT_LEN, 7, 20, 5))]
    limits = [24, 3, 17, 9]
    greedy = [reference_greedy(base, p, n).tolist() for p, n in zip(prompts, limits)]
    eager = model.decode_batch(prompts, limits, decoding=DecodeConfig(cycle_length=5))
    graphed = DecodeConfig(cycle_length=5, cuda_graphs=True)
    for _ in range(3):  # eager warm-up, capture, replay
        results = model.decode_batch(prompts, limits, decoding=graphed)
        assert [r.tokens.tolist() for r in results] == greedy
        assert [r.acceptance for r in results] == [r.acceptance for r in eager]
    decoder = model.decoder(graphed)
    assert decoder.graphs is not None
    drafter = decoder.active_drafter
    assert isinstance(drafter, KangarooDrafter) and drafter._step_fn.enabled
    steps = drafter._step_fn._graph
    assert steps is not None and steps.num_graphs > 0
    for p, n in zip(prompts, limits):  # one row at a time (a trained drafter may reach exact passes of new widths)
        assert model.decode(p, n, decoding=graphed).tokens.tolist() == reference_greedy(base, p, n).tolist()
    recorded = decoder.graphs.num_graphs, steps.num_graphs
    for p, n in zip(prompts, limits):  # and again: every graph replayed
        assert model.decode(p, n, decoding=graphed).tokens.tolist() == reference_greedy(base, p, n).tolist()
    assert (decoder.graphs.num_graphs, steps.num_graphs) == recorded
    long = model.decode(prompts[0], 300, decoding=graphed)  # outgrows the cache: new buffers and graphs
    assert long.tokens.tolist() == reference_greedy(base, prompts[0], 300).tolist()
    assert long.acceptance == model.decode(prompts[0], 300, decoding=DecodeConfig(cycle_length=5)).acceptance
