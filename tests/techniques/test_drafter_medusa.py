"""Medusa heads: lossless chains and trees, sampling, costs, checkpoints (economize, FasterDecoding/Medusa) and
training."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
import torch

import economize
from economize import ConfigError, DecodeConfig, TopPSampler
from economize.core.serialization import FORMATS, read_checkpoint
from economize.core.technique import TECHNIQUES
from economize.errors import CheckpointFormatError, IncompatibleTechniquesError
from economize.techniques.medusa import MEDUSA_TREES, MedusaCE, MedusaConfig, MedusaDrafter, MedusaTechnique, TreePlan
from economize.train import DistillKL, ObjectiveModule, default_objective, fit
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, perturb, reference_greedy

LAYERS = 6
TREE = ((0,), (1,), (0, 0), (0, 1), (1, 0), (0, 0, 0))


def _model(head: str = "own", heads: int = 3, tree: Any = None, std: float = 0.05,
           base: torch.nn.Module | None = None, **decoding: Any) -> Any:
    config = MedusaConfig(num_heads=heads, head=head, tree=tree)
    model = economize.apply(base if base is not None else make_tiny_model(), config,
                         decoding=DecodeConfig(**decoding) if decoding else None)
    if std:
        perturb(model.techniques["medusa"].heads, std, seed=1)
    return model


def _prompts(seed: int = 1, lengths: tuple[int, ...] = (PROMPT_LEN, 7, 20)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in lengths]


def _loss(model: Any, objective: Any, ids: torch.Tensor) -> dict[str, Any]:
    module = ObjectiveModule(model, objective)
    try:
        return module(ids)
    finally:
        module.close()


def _assert_greedy(model: Any, base: torch.nn.Module, new: int = 24, **kwargs: Any) -> list[Any]:
    prompts = _prompts()
    results = model.decode_batch(prompts, new, **kwargs)
    for prompt, result in zip(prompts, results):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, new))
    return results


# ---- configuration -------------------------------------------------------------------------------------------------
def test_registration_and_config() -> None:
    assert TECHNIQUES.get("medusa") is MedusaTechnique
    assert {"medusa", "tskip_medusa"} <= set(FORMATS.names())
    for config in (MedusaConfig(), MedusaConfig(num_heads=4, num_layers=2, head="shared", tree=[[0], [1], [0, 0]]),
                   MedusaConfig(tree="mc_sim_7b_63")):
        assert MedusaConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    assert MedusaConfig(tree=[[1], [0]]).paths == ((1,), (0,))
    assert TreePlan(((1, 0), (1,), (0,))).paths == ((0,), (1,), (1, 0))  # Medusa's node order
    assert len(MedusaConfig(tree="mc_sim_7b_63").paths) == 63 == len(MEDUSA_TREES.get("mc_sim_7b_63"))
    for bad in (dict(num_heads=0), dict(num_layers=0), dict(head="tied"), dict(tree="no_such_tree"),
                dict(tree=[]), dict(tree=[[0, 0]]), dict(tree=[[0], [0]]), dict(tree=[[-1]]),
                dict(num_heads=2, tree=[[0], [0, 0], [0, 0, 0]])):
        with pytest.raises(ConfigError):
            MedusaConfig(**bad)


def test_tree_plan_layout() -> None:
    plan = TreePlan(TREE)
    assert (plan.top_k, plan.depth, plan.size(1), plan.size(2), plan.size(3)) == (2, 3, 2, 5, 6)
    heads, ranks, parents = plan.layout(3)
    assert heads == [0, 0, 1, 1, 1, 2] and ranks == [0, 1, 0, 1, 0, 0] and parents == [-1, -1, 0, 0, 1, 2]
    candidates = torch.tensor([[[10, 11], [20, 21], [30, 31]]])
    assert plan.tokens(candidates, 2).tolist() == [[10, 11, 20, 21, 20]]


def test_parameters_follow_the_head_kind() -> None:
    own = _model("own", std=0.0)
    technique = own.techniques["medusa"]
    lm_head = own.base_model.lm_head.weight
    names = {name for name, _ in technique.named_parameters()}
    assert names == {f"heads.{j}.{p}" for j in range(3) for p in ("blocks.0.linear.weight", "blocks.0.linear.bias",
                                                                   "proj.weight")}
    assert all(torch.equal(head.proj.weight, lm_head) and head.proj.weight.data_ptr() != lm_head.data_ptr()
               for head in technique.heads)
    assert all(p.dtype == torch.float32 for p in technique.parameters())
    shared = _model("shared", heads=2, std=0.0).techniques["medusa"]
    assert {name for name, _ in shared.named_parameters()} == {
        f"heads.{j}.blocks.0.linear.{p}" for j in range(2) for p in ("weight", "bias")}
    assert all(not p.any() for p in shared.parameters())  # zero init: every head starts as the model's own head
    drafter = shared.drafter(None)
    assert isinstance(drafter, MedusaDrafter) and drafter.layer_span == (1, 0) and drafter.observed_layers == ()


def _pre_norm(base: torch.nn.Module, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The last hidden state before and after the final norm, and the logits."""
    captured: dict[str, torch.Tensor] = {}
    handle = base.model.norm.register_forward_hook(lambda m, i, o: captured.update(h=i[0], x=o))
    try:
        logits = base(ids).logits
    finally:
        handle.remove()
    return captured["h"], captured["x"], logits


def test_untrained_heads_are_the_model_head() -> None:
    """Zero-initialized shared heads and lm_head-initialized own heads with a zero residual weight both start as the
    model's own head (own heads keep the default bias of Medusa's residual block, so they are perturbed here)."""
    ids = torch.randint(0, VOCAB, (2, 9), generator=torch.Generator().manual_seed(0))
    shared = _model("shared", std=0.0)
    own = _model("own", std=0.0)
    for head in own.techniques["medusa"].heads:
        head.blocks[0].linear.bias.zero_()
    hidden, _, logits = _pre_norm(shared.base_model, ids)
    for model in (shared, own):
        heads = model.techniques["medusa"].head_logits(hidden)
        assert heads.shape == (2, 9, 3, VOCAB)
        torch.testing.assert_close(heads, logits[:, :, None].expand(-1, -1, 3, -1), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(model.techniques["medusa"].logits_of(hidden, 2), heads[..., 1, :])


# ---- lossless decoding ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("head", ["own", "shared"])
@pytest.mark.parametrize("cycle_length", [1, 3, 6])
def test_greedy_chains_equal_plain_greedy(head: str, cycle_length: int) -> None:
    """Cycles longer than ``K + 1 = 4`` draft only the ``K`` heads."""
    base = make_tiny_model()
    model = _model(head, base=base, cycle_length=cycle_length)
    results = _assert_greedy(model, base)
    assert all(k <= 3 for r in results for k, _ in r.acceptance)
    if cycle_length > 1:
        assert sum(a for r in results for _, a in r.acceptance) > 0


@pytest.mark.parametrize("tree", [TREE, "small", "mc_sim_7b_63"])
@pytest.mark.parametrize("cycle_length", [2, 4, 8])
def test_greedy_trees_equal_plain_greedy(tree: Any, cycle_length: int) -> None:
    base = make_tiny_model()
    heads = 5 if tree == "mc_sim_7b_63" or tree == "small" else 3
    model = _model("own", heads=heads, tree=tree, base=base, cycle_length=cycle_length)
    results = _assert_greedy(model, base)
    assert sum(a for r in results for _, a in r.acceptance) > 0


@pytest.mark.parametrize("head", ["own", "shared"])
@torch.no_grad()
def test_param_dtype_model_runs_the_heads_in_the_model_dtype(head: str) -> None:
    """``param_dtype="model"`` stores the heads in the model's dtype (bfloat16 here) with float32 logits close to
    the float32 heads'; ``"float32"`` stays the default. (Greedy equality is not checked: in bfloat16 tree passes
    and single-token passes round differently, with float32 heads too.)"""
    assert MedusaConfig().param_dtype == "float32"
    with pytest.raises(ConfigError):
        MedusaConfig(param_dtype="bfloat16")
    base = make_tiny_model().to(torch.bfloat16)
    full = _model(head, base=base).techniques["medusa"]
    model = economize.apply(base, MedusaConfig(num_heads=3, head=head, tree=TREE, param_dtype="model"))
    half = model.techniques["medusa"]
    half.load_state_dict(full.state_dict())
    assert {p.dtype for p in full.parameters()} == {torch.float32}
    assert {p.dtype for p in half.parameters()} == {torch.bfloat16} and half.dtype == torch.bfloat16
    hidden = torch.randn(2, 5, base.config.hidden_size, generator=torch.Generator().manual_seed(3)).bfloat16()
    logits = half.head_logits(hidden)
    assert logits.dtype == torch.float32 and logits.shape[-2] == 3
    torch.testing.assert_close(logits, full.head_logits(hidden), rtol=0.05, atol=0.05)
    assert [len(model.decode(p, 12).tokens) for p in _prompts()] == [12] * 3


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@torch.no_grad()
def test_graphed_tree_passes_are_greedy_and_replayed() -> None:
    base = make_tiny_model().cuda()
    model = _model("own", heads=5, tree="mc_sim_7b_63", base=base)
    decoding = DecodeConfig(cycle_length=5, cuda_graphs=True)
    prompts = [p.cuda() for p in _prompts()]
    greedy = [reference_greedy(base, p, 24).tolist() for p in prompts]
    for _ in range(3):  # eager warm-up, capture, replay
        results = model.decode_batch(prompts, 24, decoding=decoding)
        assert [r.tokens.tolist() for r in results] == greedy
        assert sum(a for r in results for _, a in r.acceptance) > 0
    graphs = model.decoder(decoding).graphs
    assert graphs is not None and graphs.num_graphs > 0


def test_typical_acceptance_with_a_greedy_selector_is_greedy() -> None:
    base = make_tiny_model()
    model = _model("own", heads=5, tree="mc_sim_7b_63", base=base)
    _assert_greedy(model, base, decoding=DecodeConfig(cycle_length=6, verifier="typical"))


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
    assert sum(k for k, _ in result.acceptance) > 0


def test_dynamic_draft_length_is_lossless() -> None:
    base = make_tiny_model()
    model = _model("own", heads=5, base=base)
    results = _assert_greedy(model, base, decoding=DecodeConfig(cycle_length=6, min_confidence=0.6))
    assert any(k < 5 for r in results for k, _ in r.acceptance)


def test_lossy_decoding_keeps_the_head_drafts() -> None:
    model = _model("own", cycle_length=4, lossless=False)
    result = model.decode(_prompts()[0], 21)
    assert result.tokens.shape == (21,) and all(a == k for k, a in result.acceptance)


# ---- cost accounting -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("tree", [None, TREE])
def test_cost_log_has_one_zero_layer_event_per_head_run(tree: Any) -> None:
    model = _model("own", tree=tree, cycle_length=4)
    result = model.decode(_prompts()[0], 30)
    events = result.cost.events
    drafts = [e for e in events if e.kind in ("draft", "draft_wasted")]
    assert {e.kind for e in events} <= {"exact", "draft", "draft_wasted"}
    assert all((e.tokens, e.first_layer, e.last_layer, e.layer_tokens, e.module) == (1, 1, 0, 0, "medusa")
               for e in drafts)
    assert all((e.first_layer, e.last_layer) == (1, LAYERS) for e in events if e.kind == "exact")
    exact = [e for e in events if e.kind == "exact"]
    if tree is None:
        assert len(drafts) == sum(k for k, _ in result.acceptance)
    else:  # one event per tree level (head), the exact pass reads every node
        cycles = [e.tokens - 1 for e in exact[1:]]
        assert all(n in (0, 2, 5, 6) for n in cycles)
        assert len(drafts) == sum({0: 0, 2: 1, 5: 2, 6: 3}[n] for n in cycles)


# ---- save, load and combinations -----------------------------------------------------------------------------------
@pytest.mark.parametrize("tree", [None, "small"])
def test_save_and_load_round_trip(tree: Any, tmp_path: Path) -> None:
    model = _model("own", heads=5, tree=tree, cycle_length=4)
    economize.save(model, tmp_path / "ckpt")
    loaded = economize.load(make_tiny_model(), tmp_path / "ckpt")
    assert loaded.drafting_technique.config == model.drafting_technique.config
    prompt = _prompts()[1]
    first, second = model.decode(prompt, 20), loaded.decode(prompt, 20)
    assert torch.equal(first.tokens, second.tokens) and first.acceptance == second.acceptance


def test_medusa_on_a_pruned_model_equals_its_greedy(calib) -> None:
    from economize.techniques.prune import PruneConfig

    base = make_tiny_model()
    model = economize.apply(base, PruneConfig(method="wanda", sparsity="2:4"), MedusaConfig(num_heads=3, tree=TREE),
                         decoding=DecodeConfig(cycle_length=4), calibration=calib)
    assert list(model.techniques) == ["prune", "medusa"]
    _assert_greedy(model, base)


def _medusa_dir(path: Path, heads: int, layers: int, prefix: str, suffix: str, seed: int = 0) -> dict[str, Any]:
    """A FasterDecoding/Medusa head directory for the tiny model with random weights."""
    generator = torch.Generator().manual_seed(seed)
    hidden = make_tiny_model().config.hidden_size
    state = {}
    for i in range(heads):
        for j in range(layers):
            state[f"{prefix}{i}.{j}.linear.weight"] = torch.randn(hidden, hidden, generator=generator) * 0.05
            state[f"{prefix}{i}.{j}.linear.bias"] = torch.randn(hidden, generator=generator) * 0.05
        state[f"{prefix}{i}.{layers}.weight"] = torch.randn(VOCAB, hidden, generator=generator) * 0.1
    path.mkdir(parents=True)
    config = {"medusa_num_heads": heads, "medusa_num_layers": layers, "base_model_name_or_path": "tiny/base"}
    (path / "config.json").write_text(json.dumps(config))
    if suffix == ".safetensors":
        from safetensors.torch import save_file

        save_file({k: v.to(torch.bfloat16) for k, v in state.items()}, path / "medusa_lm_head.safetensors")
    else:
        torch.save(state, path / "medusa_lm_head.pt")
    return state


@pytest.mark.parametrize(("prefix", "suffix", "layers"), [("medusa_head.", ".safetensors", 1), ("", ".pt", 2)])
def test_fasterdecoding_medusa_heads_load(tmp_path: Path, prefix: str, suffix: str, layers: int) -> None:
    state = _medusa_dir(tmp_path / "heads", 4, layers, prefix, suffix)
    checkpoint = read_checkpoint(tmp_path / "heads")
    assert checkpoint.base_model == "tiny/base"
    assert checkpoint.configs == [MedusaConfig(num_heads=4, num_layers=layers, head="own")]
    base = make_tiny_model()
    model = economize.load(base, tmp_path / "heads", decoding=DecodeConfig(cycle_length=5))
    technique = model.techniques["medusa"]
    # Medusa's head: x = norm(h); x <- x + SiLU(W x + b) per block; logits = proj(x)
    ids = torch.randint(0, VOCAB, (1, 7), generator=torch.Generator().manual_seed(2))
    hidden, x, _ = _pre_norm(base, ids)
    tensors = {k.removeprefix(prefix): v.to(torch.bfloat16).float() if suffix == ".safetensors" else v
               for k, v in state.items()}
    for j in range(layers):
        x = x + torch.nn.functional.silu(x @ tensors[f"1.{j}.linear.weight"].T + tensors[f"1.{j}.linear.bias"])
    expected = x @ tensors[f"1.{layers}.weight"].T
    torch.testing.assert_close(technique.head_logits(hidden)[..., 1, :], expected, rtol=1e-5, atol=1e-5)
    _assert_greedy(model, base)


def test_tskip_medusa_jump_file_loads_by_name(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(0)
    hidden = make_tiny_model().config.hidden_size
    state = {"W": torch.randn(3, hidden, hidden, generator=generator) * 0.05,
             "b": torch.randn(3, hidden, generator=generator) * 0.05}
    run = tmp_path / "run"
    run.mkdir()
    torch.save({"kind": "medusa", "model": "tiny/base", "K": 3, "state": state}, run / "jump.pt")
    checkpoint = read_checkpoint(run, format="tskip_medusa")
    assert checkpoint.configs == [MedusaConfig(num_heads=3, num_layers=1, head="shared")]
    assert FORMATS.get("tskip_medusa").detect(run) and not FORMATS.get("medusa").detect(run)
    base = make_tiny_model()
    model = economize.load(base, run, format="tskip_medusa", decoding=DecodeConfig(cycle_length=4))
    blocks = model.techniques["medusa"].heads[2].blocks[0].linear
    assert torch.equal(blocks.weight, state["W"][2]) and torch.equal(blocks.bias, state["b"][2])
    _assert_greedy(model, base)


# ---- training ------------------------------------------------------------------------------------------------------
def test_default_objective_and_parameter_groups() -> None:
    model = _model("own", std=0.0)
    objective = default_objective(model)
    assert isinstance(objective, MedusaCE) and objective.defaults.lr == 1e-3 and objective.defaults.weight_decay == 0
    (group,) = objective.parameter_groups(model)
    assert group.kind == "head" and len(group.params) == 9
    with pytest.raises(IncompatibleTechniquesError):
        MedusaCE().parameter_groups(economize.apply(make_tiny_model()))
    with pytest.raises(ConfigError):
        MedusaCE(ce_weight=0.0, kl_weight=0.0)


def _reference_terms(model: Any, ids: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-head CE and KL of zero-initialized shared heads: head j at t is the model's own distribution at t."""
    logp = model.base_model(ids).logits.float().log_softmax(-1)
    length, ce, kl = ids.shape[1], [], []
    for j in range(1, 4):
        s = j + 1
        student = logp[:, : length - s]
        ce.append(-student.gather(-1, ids[:, s:, None])[..., 0].mean())
        teacher = logp[:, s - 1 : length - 1]
        kl.append((teacher.exp() * (teacher - student)).sum(-1).mean())
    return ce, kl


def test_medusa_ce_and_distillation_losses_at_initialization() -> None:
    ids = torch.randint(0, VOCAB, (2, 14), generator=torch.Generator().manual_seed(4))
    model = _model("shared", std=0.0)
    ce, kl = _reference_terms(model, ids)
    weights = [0.8 ** j for j in range(1, 4)]  # the paper's lambda_k = 0.8 ** k, summed over heads
    with torch.enable_grad():
        out = _loss(model, MedusaCE(), ids)
        expected = sum(w * c for w, c in zip(weights, ce))
        torch.testing.assert_close(out["loss"].detach(), expected, rtol=1e-5, atol=1e-6)
        mean = _loss(model, MedusaCE(normalize=True), ids)
        torch.testing.assert_close(mean["loss"].detach(), expected / sum(weights), rtol=1e-5, atol=1e-6)
        mixed = _loss(model, MedusaCE(ce_weight=0.5, kl_weight=2.0), ids)
        expected = sum(w * (0.5 * c + 2.0 * k) for w, c, k in zip(weights, ce, kl))
        torch.testing.assert_close(mixed["loss"].detach(), expected, rtol=1e-5, atol=1e-6)
        assert {"ce1", "kl1", "acc3"} <= set(mixed["metrics"])
        distill = _loss(model, DistillKL(), ids)  # the TrainableDrafter protocol: keys j + 1 for head j
        torch.testing.assert_close(distill["loss"].detach(), sum(kl) / 3, rtol=1e-4, atol=1e-6)
        mixed["loss"].backward()
    grads = [p.grad for p in model.techniques["medusa"].parameters()]
    assert all(g is not None for g in grads) and any(g.abs().sum() > 0 for g in grads)
    assert not any(p.requires_grad for p in model.base_model.parameters())


def test_distill_kl_trains_the_heads_through_the_trainable_drafter_protocol() -> None:
    rows = torch.randint(0, VOCAB, (8, 20), generator=torch.Generator().manual_seed(6))
    model = _model("shared", std=0.0)
    technique = model.techniques["medusa"]
    before = [p.detach().clone() for p in technique.parameters()]
    with torch.enable_grad():
        result = fit(model, rows, "distill_kl", steps=3, batch_size=8, lr=1e-2, warmup_steps=0, seed=0, log_every=1)
    assert result.steps == 3 and result.losses[-1] < result.losses[0]
    assert all(not torch.equal(a, b) for a, b in zip(before, technique.parameters()))


@pytest.mark.parametrize("head", ["own", "shared"])
def test_fit_trains_the_heads(head: str) -> None:
    rows = torch.randint(0, VOCAB, (16, 24), generator=torch.Generator().manual_seed(3))
    rows[:, 1::2] = (rows[:, 0::2] + 1) % VOCAB  # a learnable pattern: every odd token follows the one before
    model = _model(head, std=0.0)
    with torch.enable_grad():
        result = fit(model, rows, steps=30, batch_size=8, lr=3e-3, warmup_steps=3, seed=0, log_every=10)
    assert result.steps == 30
    assert sum(result.losses[-5:]) < 0.95 * sum(result.losses[:5])
    assert not any(p.requires_grad for p in model.base_model.parameters())


def test_the_head_count_comes_from_the_weights(tmp_path: Path) -> None:
    """FasterDecoding/medusa-vicuna-7b-v1.3 states 3 heads in config.json and ships 5: the weights decide."""
    _medusa_dir(tmp_path / "heads", 5, 1, "medusa_head.", ".pt")
    config = json.loads((tmp_path / "heads" / "config.json").read_text())
    (tmp_path / "heads" / "config.json").write_text(json.dumps({**config, "medusa_num_heads": 3}))
    with pytest.warns(UserWarning, match="using the weights"):
        checkpoint = read_checkpoint(tmp_path / "heads")
    assert checkpoint.configs == [MedusaConfig(num_heads=5, num_layers=1, head="own")]


def test_keep_heads_drops_the_trailing_heads(tmp_path: Path) -> None:
    from economize.techniques.medusa import keep_heads

    _medusa_dir(tmp_path / "heads", 5, 1, "", ".pt")
    full = read_checkpoint(tmp_path / "heads")
    kept = keep_heads(full, 4)
    assert kept.configs == [MedusaConfig(num_heads=4, num_layers=1, head="own")]
    assert not any(key.startswith("heads.4.") for key in kept.states["medusa"])
    base = make_tiny_model()
    model = economize.attach(base, dataclasses.replace(kept, configs=[dataclasses.replace(kept.configs[0],
                                                                                         tree="mc_sim_7b_63")]),
                          decoding=DecodeConfig(cycle_length=5))
    loaded = model.techniques["medusa"].heads[3].proj.weight
    assert torch.equal(loaded, full.states["medusa"]["heads.3.proj.weight"].to(loaded))
    _assert_greedy(model, base)
    with pytest.raises(CheckpointFormatError):
        keep_heads(full, 6)
