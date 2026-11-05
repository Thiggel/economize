"""EAGLE-3: lossless decoding, sampling, costs, the training-time test, and official and research checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F

import economize
from economize import ConfigError, DecodeConfig, TopPSampler
from economize.core.serialization import FORMATS, read_checkpoint
from economize.core.technique import TECHNIQUES
from economize.errors import CheckpointFormatError, IncompatibleTechniquesError
from economize.runtime import ExactPass
from economize.techniques.eagle3 import Eagle3Config, Eagle3Drafter, Eagle3Technique, Eagle3TTT, default_aux_layers
from economize.techniques.eagle3.formats import (
    base_model_of,
    config_from_eagle3,
    draft_layer_count,
    flat_config,
    state_from_eagle3,
)
from economize.train import DistillKL, ObjectiveModule, default_objective, fit
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, perturb, reference_greedy

LAYERS = 6
HIDDEN = 64
DRAFT = 96
ROPE = dict(rope_theta=1e4, head_dim=8, num_heads=2, num_kv_heads=1, intermediate_size=32)
VARIANTS = {
    "shared": Eagle3Config(head="shared"),
    "own": Eagle3Config(),
    "vocab": Eagle3Config(draft_vocab=DRAFT),
    "zero_source": Eagle3Config(zero_source_first=True),
    "norm_residual": Eagle3Config(norm_before_residual=True),
    "own_rope": Eagle3Config(**ROPE),
    "ends": Eagle3Config(aux_layers=(0, LAYERS)),
}


def _draft_ids(seed: int = 3) -> torch.Tensor:
    return torch.randperm(VOCAB, generator=torch.Generator().manual_seed(seed))[:DRAFT]


def _model(config: Eagle3Config | None = None, std: float = 0.05, base: torch.nn.Module | None = None,
           **decoding: Any) -> Any:
    config = config or Eagle3Config()
    model = economize.apply(base if base is not None else make_tiny_model(), config,
                         decoding=DecodeConfig(**decoding) if decoding else None)
    technique = model.techniques["eagle3"]
    if config.draft_vocab is not None:
        technique.set_draft_vocab(_draft_ids())
    if std:
        perturb(technique, std, seed=1)
    return model


def _prompts(seed: int = 1, lengths: tuple[int, ...] = (PROMPT_LEN, 7, 20)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in lengths]


def _hidden(base: torch.nn.Module, ids: torch.Tensor, layers: tuple[int, ...]) -> dict[int, torch.Tensor]:
    """The model's states at ``layers`` (``0``: embeddings, ``l < L``: block outputs) from HF."""
    states = base(ids, output_hidden_states=True).hidden_states
    return {layer: states[layer] for layer in layers}


def _loss(model: Any, objective: Any, ids: torch.Tensor) -> dict[str, Any]:
    module = ObjectiveModule(model, objective)
    try:
        return module(ids)
    finally:
        module.close()


def _assert_greedy(model: Any, base: torch.nn.Module, prompts: list[torch.Tensor] | None = None, new: int = 24,
                   **kwargs: Any) -> list[Any]:
    prompts = prompts if prompts is not None else _prompts()
    results = model.decode_batch(prompts, new, **kwargs)
    for prompt, result in zip(prompts, results):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, new))
    return results


@pytest.fixture(scope="module")
def trained() -> SimpleNamespace:
    """An own-head drafter fitted for 60 steps to the tiny model's greedy continuations of four prompts."""
    generator = torch.Generator().manual_seed(1)
    prompts = [torch.randint(1, VOCAB, (PROMPT_LEN,), generator=generator) for _ in range(4)]
    base = make_tiny_model()
    with torch.no_grad():
        rows = torch.stack([torch.cat([p, reference_greedy(base, p, 28)]) for p in prompts])
    model = economize.apply(make_tiny_model(), Eagle3Config(), decoding=DecodeConfig(cycle_length=4))
    with torch.enable_grad():
        fit(model, rows, steps=60, batch_size=4, lr=1e-2, warmup_steps=0, seed=0)
    return SimpleNamespace(model=model, base=base, prompts=prompts, rows=rows)


# ---- configuration -------------------------------------------------------------------------------------------------
def test_registration_and_config() -> None:
    assert TECHNIQUES.get("eagle3") is Eagle3Technique
    assert {"eagle3", "tskip_eagle3"} <= set(FORMATS.names())
    for config in (Eagle3Config(), Eagle3Config(aux_layers=(1, 3), head="shared", zero_source_first=True),
                   Eagle3Config(draft_vocab=32, rms_norm_eps=1e-5, attention_bias=True, **ROPE)):
        assert Eagle3Config.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    assert (default_aux_layers(6), default_aux_layers(8), default_aux_layers(32)) == ((2, 3), (2, 4, 5), (2, 16, 29))
    assert Eagle3Config(aux_layers=[3, 1]).aux_layers == (3, 1)
    for bad in (dict(aux_layers=()), dict(aux_layers=(-1,)), dict(head="tied"), dict(head="shared", draft_vocab=8),
                dict(draft_vocab=0), dict(num_heads=3, num_kv_heads=2), dict(head_dim=0), dict(rope_theta=0.0),
                dict(rms_norm_eps=0.0), dict(train_steps=0)):
        with pytest.raises(ConfigError):
            Eagle3Config(**bad)
    for bad in (dict(aux_layers=(LAYERS + 1,)), dict(draft_vocab=VOCAB + 72), dict(head_dim=8),
                dict(num_kv_heads=3)):
        with pytest.raises(ConfigError):
            economize.apply(make_tiny_model(), Eagle3Config(**bad))


def test_parameters_follow_the_head_kind() -> None:
    shared = _model(VARIANTS["shared"], std=0.0).techniques["eagle3"]
    names = {name for name, _ in shared.named_parameters()}
    assert "fc.weight" in names and not any(n.startswith(("norm.", "lm_head.")) for n in names)
    assert shared.aux_layers == (2, 3) and shared.fc.weight.shape == (HIDDEN, 2 * HIDDEN)
    assert shared.draft_ids is None and shared.draft_vocab_size == VOCAB
    with pytest.raises(ConfigError):
        shared.set_draft_vocab(range(DRAFT))
    own_model = _model(VARIANTS["own"], std=0.0)
    own = own_model.techniques["eagle3"]
    assert torch.equal(own.lm_head.weight, own_model.base_model.lm_head.weight)
    assert own.lm_head.weight.data_ptr() != own_model.base_model.lm_head.weight.data_ptr()
    assert all(p.dtype == torch.float32 for p in own.parameters())
    assert {name for name, _ in own.named_buffers()} >= {"d2t", "t2d"}
    states = torch.randn(3, 5, HIDDEN, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(own.head_logits(states), shared.head_logits(states), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(own.draft_logits(states), own.head_logits(states))
    drafter = own.drafter(None)
    assert isinstance(drafter, Eagle3Drafter) and drafter.layer_span == (1, 0) and drafter.observed_layers == (0, 2, 3)
    rope = _model(VARIANTS["own_rope"], std=0.0).techniques["eagle3"].layer
    assert (rope.num_heads, rope.num_kv_heads, rope.head_dim) == (2, 1, 8)


def test_draft_vocabulary() -> None:
    model = _model(Eagle3Config(draft_vocab=DRAFT), std=0.0)
    technique = model.techniques["eagle3"]
    ids = _draft_ids()
    order = ids.sort().values
    assert technique.draft_vocab_size == DRAFT and torch.equal(technique.draft_ids, order)
    assert int(technique.t2d.sum()) == DRAFT and bool(technique.t2d[ids].all())
    assert torch.equal(technique.lm_head.weight, model.base_model.lm_head.weight[order])
    states = torch.randn(2, HIDDEN, generator=torch.Generator().manual_seed(0))
    logits, head = technique.draft_logits(states), technique.head_logits(states)
    assert logits.shape == (2, VOCAB) and head.shape == (2, DRAFT)
    assert torch.isneginf(logits[:, ~technique.t2d]).all() and torch.equal(logits[:, order], head)
    for bad in (ids[:-1], torch.cat([ids[:-1], ids[:1]]), torch.cat([ids[:-1], torch.tensor([VOCAB])])):
        with pytest.raises(ConfigError):
            technique.set_draft_vocab(bad)


# ---- lossless decoding ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("cycle_length", [1, 3, 6])
def test_greedy_chains_equal_plain_greedy(variant: str, cycle_length: int) -> None:
    base = make_tiny_model()
    model = _model(VARIANTS[variant], base=base, cycle_length=cycle_length)
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


def test_rejection_at_near_zero_temperature_is_greedy(trained: SimpleNamespace) -> None:
    config = DecodeConfig(cycle_length=4, verifier="rejection")
    _assert_greedy(trained.model, trained.base, trained.prompts, decoding=config, selector=TopPSampler(1e-6, seed=0))
    base = make_tiny_model()
    _assert_greedy(_model(VARIANTS["vocab"], base=base), base, decoding=config, selector=TopPSampler(1e-6, seed=0))


@pytest.mark.parametrize("variant", ["trained", "vocab"])
def test_rejection_sampling_runs(trained: SimpleNamespace, variant: str) -> None:
    model = trained.model if variant == "trained" else _model(VARIANTS["vocab"])
    config = DecodeConfig(cycle_length=4, verifier="rejection")
    result = model.decode(trained.prompts[0], 24, decoding=config, selector=TopPSampler(1.0, seed=0))
    assert result.tokens.shape == (24,) and int(result.tokens.max()) < VOCAB
    assert sum(k for k, _ in result.acceptance) > 0


def test_dynamic_draft_length_is_lossless(trained: SimpleNamespace) -> None:
    results = _assert_greedy(trained.model, trained.base, trained.prompts,
                             decoding=DecodeConfig(cycle_length=6, min_confidence=0.6))
    assert any(k < 5 for r in results for k, _ in r.acceptance)


def test_lossy_decoding_keeps_the_drafts(trained: SimpleNamespace) -> None:
    result = trained.model.decode(trained.prompts[0], 21, decoding=DecodeConfig(cycle_length=4, lossless=False))
    assert result.tokens.shape == (21,) and all(a == k for k, a in result.acceptance)


def test_cost_log_has_one_draft_layer_event_per_draft_step(trained: SimpleNamespace) -> None:
    result = trained.model.decode(trained.prompts[1], 30)
    events = result.cost.events
    drafts = [e for e in events if e.kind in ("draft", "draft_wasted")]
    assert {e.kind for e in events} <= {"exact", "draft", "draft_wasted", "draft_observe"}
    # one step runs the draft layer (a separate network, so its own module) on one token
    assert all((e.tokens, e.layer_tokens, e.module) == (1, 1, "eagle3") for e in drafts)
    # every exact pass followed by a proposal also runs the draft layer over the pass's tokens
    observe = [e for e in events if e.kind == "draft_observe"]
    assert observe and all(e.module == "eagle3" and e.layer_tokens == e.tokens for e in observe)
    assert all((e.first_layer, e.last_layer) == (1, LAYERS) for e in events if e.kind == "exact")
    assert len(drafts) == sum(k for k, _ in result.acceptance)


# ---- the drafter and the training-time test agree -------------------------------------------------------------------
@pytest.mark.parametrize("variant", ["shared", "zero_source", "own_rope", "vocab"])
def test_drafter_steps_equal_the_training_rollout(variant: str) -> None:
    """After exact passes over positions ``0..12``, draft step ``i`` (position ``12 + i``) equals rollout step ``i``
    at position 12."""
    model = _model(VARIANTS[variant])
    technique = model.techniques["eagle3"]
    ids = torch.randint(0, VOCAB, (2, 16), generator=torch.Generator().manual_seed(5))
    hidden = _hidden(model.base_model, ids, (0, 2, 3))
    expected = technique.rollout(hidden, 3)
    drafter = technique.drafter(None)
    drafter.begin(None, SimpleNamespace(capacity=64, num_slots=65), 2)
    for start, end in ((0, 10), (10, 13)):
        context = SimpleNamespace(positions=torch.arange(start, end).expand(2, -1), valid=None)
        exact = ExactPass(hidden[3][:, start:end], {k: v[:, start:end] for k, v in hidden.items()}, context)
        drafter.observe(exact, torch.full((2,), end - start - 1))
    for i in range(1, 4):
        logits = drafter.step(ids[:, 12 + i], torch.full((2,), 12 + i), None)
        torch.testing.assert_close(logits, technique.draft_logits(expected[i - 1][:, 12]), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("variant", ["own", "zero_source", "own_rope"])
def test_checkpointed_steps_give_the_same_rollout_and_gradients(variant: str) -> None:
    """``checkpoint_steps`` recomputes the training-time test's steps in the backward pass: same outputs, same
    gradients, and no effect without gradients."""
    import dataclasses

    plain = _model(VARIANTS[variant])
    checkpointed = _model(dataclasses.replace(VARIANTS[variant], checkpoint_steps=True))
    checkpointed.techniques["eagle3"].load_state_dict(plain.techniques["eagle3"].state_dict())
    ids = torch.randint(0, VOCAB, (2, 16), generator=torch.Generator().manual_seed(5))
    hidden = _hidden(plain.base_model, ids, (0, 2, 3))
    weights = torch.randn(4, generator=torch.Generator().manual_seed(2))
    grads = []
    for model in (plain, checkpointed):
        technique = model.techniques["eagle3"]
        with torch.enable_grad():
            outputs = technique.rollout(hidden, 4)
            loss = sum(w * technique.draft_logits(o).logsumexp(-1).mean() for w, o in zip(weights, outputs))
            loss.backward()
        grads.append({n: p.grad.clone() for n, p in technique.named_parameters() if p.grad is not None})
        with torch.no_grad():
            grads[-1]["outputs"] = torch.stack(technique.rollout(hidden, 4))
    assert grads[0].keys() == grads[1].keys() and len(grads[0]) > 1
    for name in grads[0]:
        torch.testing.assert_close(grads[1][name], grads[0][name], rtol=1e-5, atol=1e-6, msg=name)
    with pytest.raises(ConfigError):
        Eagle3Config(train_steps=0, checkpoint_steps=True)


@pytest.mark.parametrize("variant", ["own", "shared", "own_rope"])
def test_param_dtype_model_stores_the_drafter_in_the_model_dtype(variant: str) -> None:
    """``param_dtype="model"`` keeps the drafter in the model's dtype (bfloat16 here) with float32 logits and
    gradients in the parameters' dtype, close to the float32 drafter; ``"float32"`` stays the default."""
    import dataclasses

    assert Eagle3Config().param_dtype == "float32"
    with pytest.raises(ConfigError):
        Eagle3Config(param_dtype="bfloat16")
    base = make_tiny_model().to(torch.bfloat16)
    full = _model(VARIANTS[variant], base=base).techniques["eagle3"]
    half = _model(dataclasses.replace(VARIANTS[variant], param_dtype="model"), base=base).techniques["eagle3"]
    assert {p.dtype for p in full.parameters()} == {torch.float32}
    assert {p.dtype for p in half.parameters()} == {torch.bfloat16} and half.dtype == torch.bfloat16
    with torch.no_grad():
        half.load_state_dict(full.state_dict())
    ids = torch.randint(0, VOCAB, (2, 16), generator=torch.Generator().manual_seed(5))
    hidden = _hidden(base, ids, (0, 2, 3))
    with torch.enable_grad():
        outputs = half.rollout(hidden, 3)
        logits = half.draft_logits(outputs[-1])
        logits.logsumexp(-1).mean().backward()
    assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
    grads = [p.grad for p in half.parameters() if p.grad is not None]
    assert grads and all(g.dtype == torch.bfloat16 and torch.isfinite(g).all() for g in grads)
    with torch.no_grad():
        reference = full.draft_logits(full.rollout(hidden, 3)[-1])
    torch.testing.assert_close(logits.detach(), reference, rtol=0.05, atol=0.05)


@pytest.mark.parametrize("variant", ["own", "own_rope"])
def test_param_dtype_model_decodes(variant: str) -> None:
    """Regression: with ``param_dtype="model"`` on a bfloat16 model the drafter's own K/V cache and fused source
    were allocated in float32, so the first ``observe`` failed (``scatter(): Expected self.dtype to be equal to
    src.dtype``). The cache now has the drafter's dtype and decoding matches the float32 drafter with equal weights."""
    import dataclasses

    base = make_tiny_model().to(torch.bfloat16)
    prompts = _prompts()
    tokens = {}
    full = _model(VARIANTS[variant], base=base, cycle_length=4)
    state = full.techniques["eagle3"].state_dict()
    tokens["float32"] = [r.tokens.tolist() for r in full.decode_batch(prompts, 12)]
    full.detach()
    half = _model(dataclasses.replace(VARIANTS[variant], param_dtype="model"), std=0, base=base, cycle_length=4)
    with torch.no_grad():
        half.techniques["eagle3"].load_state_dict(state)
    tokens["model"] = [r.tokens.tolist() for r in half.decode_batch(prompts, 12)]
    assert tokens["model"] == tokens["float32"]
    assert tokens["model"] == [half.decode(p, 12).tokens.tolist() for p in prompts]


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("param_dtype", ["float32", "model"])
@pytest.mark.parametrize("cycle_length", [3, 6])
def test_bfloat16_decoding(variant: str, param_dtype: str, cycle_length: int) -> None:
    """Every variant decodes a bfloat16 model, with a float32 drafter or one in the model's dtype: a ragged batch
    gives the tokens of decoding each row alone, and at most ``cycle_length - 1`` drafts are kept per cycle.

    Equality with :func:`reference_greedy` is not asserted: in bfloat16 a batched verification pass rounds
    differently from single-token steps, so greedy chains drift apart after a while (the fp32 tests are lossless)."""
    import dataclasses

    base = make_tiny_model().to(torch.bfloat16)
    config = dataclasses.replace(VARIANTS[variant], param_dtype=param_dtype)
    model = _model(config, base=base, cycle_length=cycle_length)
    technique = model.techniques["eagle3"]
    expected = torch.float32 if param_dtype == "float32" else torch.bfloat16
    assert {p.dtype for p in technique.parameters()} == {expected}
    prompts = _prompts()
    results = model.decode_batch(prompts, 16)
    for prompt, result in zip(prompts, results):
        assert result.tokens.shape == (16,) and int(result.tokens.min()) >= 0 and int(result.tokens.max()) < VOCAB
        assert all(k <= cycle_length - 1 for k, _ in result.acceptance)
        alone = model.decode(prompt, 16)
        assert torch.equal(result.tokens, alone.tokens) and result.acceptance == alone.acceptance
        # The prefill's token is the base model's greedy token, up to ties within bf16 rounding (two logits can be
        # equal in bf16, and which one the batched prefill picks depends on the CPU's kernels).
        logits = base(prompt.view(1, -1)).logits[0, -1].float()
        assert logits.max() - logits[result.tokens[0]] <= 2**-6 * logits.abs().max()


# ---- save, load and combinations -----------------------------------------------------------------------------------
@pytest.mark.parametrize("variant", ["trained", "vocab"])
def test_save_and_load_round_trip(trained: SimpleNamespace, variant: str, tmp_path: Path) -> None:
    model = trained.model if variant == "trained" else _model(VARIANTS["vocab"], cycle_length=4)
    economize.save(model, tmp_path / "ckpt")
    loaded = economize.load(make_tiny_model(), tmp_path / "ckpt")
    assert loaded.drafting_technique.config == model.drafting_technique.config
    for (name, a), (_, b) in zip(model.techniques["eagle3"].state_dict().items(),
                                 loaded.techniques["eagle3"].state_dict().items()):
        assert torch.equal(a, b), name
    first, second = model.decode(trained.prompts[2], 20), loaded.decode(trained.prompts[2], 20)
    assert torch.equal(first.tokens, second.tokens) and first.acceptance == second.acceptance


def test_eagle3_on_a_pruned_model_equals_its_greedy(trained: SimpleNamespace, calib: torch.Tensor) -> None:
    from economize.techniques.prune import PruneConfig

    base = make_tiny_model()
    model = economize.apply(base, PruneConfig(method="wanda", sparsity="2:4"), Eagle3Config(),
                         decoding=DecodeConfig(cycle_length=4), calibration=calib)
    assert list(model.techniques) == ["prune", "eagle3"]
    model.techniques["eagle3"].load_state_dict(trained.model.techniques["eagle3"].state_dict())
    _assert_greedy(model, base, trained.prompts)


# ---- official EAGLE-3 heads ----------------------------------------------------------------------------------------
YUHUILI = dict(  # SafeAILab's layout: midlayer.*, a copy of the embedding, a draft vocabulary, no rope_theta
    architectures=["LlamaForCausalLMEagle3"], hidden_size=HIDDEN, intermediate_size=96, num_attention_heads=4,
    num_key_value_heads=2, num_hidden_layers=1, rms_norm_eps=1e-5, vocab_size=VOCAB, draft_vocab_size=DRAFT,
    hidden_act="silu", max_position_embeddings=2048, tie_word_embeddings=False, base_model_name_or_path="tiny/base",
)
ANGELSLIM = dict(  # AngelSlim / SpecForge layout: layers.0.*, own head width, rope_theta, explicit aux layers
    architectures=["Eagle3LlamaForCausalLM"], hidden_size=HIDDEN, intermediate_size=128, num_attention_heads=8,
    num_key_value_heads=4, head_dim=8, num_hidden_layers=1, rms_norm_eps=1e-6, rope_theta=1e6, vocab_size=VOCAB,
    draft_vocab_size=VOCAB, attention_bias=False,
    eagle_config={"eagle_aux_hidden_state_layer_ids": [1, 3, 4], "use_aux_hidden_state": True},
)
SPECULATORS = dict(  # vLLM speculators (RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3): nested layer config
    architectures=["Eagle3Speculator"], speculators_model_type="eagle3", draft_vocab_size=DRAFT,
    norm_before_residual=False, target_hidden_size=None,
    speculators_config={"algorithm": "eagle3", "verifier": {"name_or_path": "tiny/base"}},
    transformer_layer_config=dict(
        hidden_size=HIDDEN, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_hidden_layers=1, rms_norm_eps=1e-5, rope_theta=5e5, rope_scaling=None, vocab_size=VOCAB,
        max_position_embeddings=2048, attention_bias=False, model_type="llama",
    ),
)


def _official_state(config: dict[str, Any], aux: int, prefix: str, draft_ids: torch.Tensor,
                    embedding: bool, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    d, width = config["hidden_size"], config["intermediate_size"]
    heads, kv = config["num_attention_heads"], config["num_key_value_heads"]
    dim = config.get("head_dim", d // heads)

    def rand(*shape: int, std: float = 0.05) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * std

    state = {"fc.weight": rand(d, aux * d), "norm.weight": 1 + rand(d, std=0.1),
             "lm_head.weight": rand(len(draft_ids), d, std=0.2)}
    layer = {
        "input_layernorm.weight": 1 + rand(d, std=0.1), "hidden_norm.weight": 1 + rand(d, std=0.1),
        "post_attention_layernorm.weight": 1 + rand(d, std=0.1),
        "self_attn.q_proj.weight": rand(heads * dim, 2 * d, std=0.1),
        "self_attn.k_proj.weight": rand(kv * dim, 2 * d, std=0.1),
        "self_attn.v_proj.weight": rand(kv * dim, 2 * d), "self_attn.o_proj.weight": rand(d, heads * dim),
        "mlp.gate_proj.weight": rand(width, d), "mlp.up_proj.weight": rand(width, d),
        "mlp.down_proj.weight": rand(d, width),
    }
    state.update({prefix + key: value for key, value in layer.items()})
    state["d2t"] = draft_ids - torch.arange(len(draft_ids))
    t2d = torch.zeros(VOCAB, dtype=torch.bool)
    t2d[draft_ids] = True
    state["t2d"] = t2d
    if embedding:
        state["embed_tokens.weight"] = rand(VOCAB, d)
    return state


def _write_head(path: Path, config: dict[str, Any], state: dict[str, torch.Tensor], safetensors: bool) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config))
    if safetensors:
        from safetensors.torch import save_file

        save_file({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in state.items()},
                  path / "model.safetensors")
    else:
        torch.save(state, path / "pytorch_model.bin")


def _reference_step(tensors: dict[str, torch.Tensor], config: dict[str, Any], prefix: str, aux: tuple[int, ...],
                    hidden: dict[int, torch.Tensor]) -> torch.Tensor:
    """EAGLE-3's first draft step at positions ``1..T-1`` (SafeAILab ``cnets.py``), written out from the raw tensors:
    logits ``[B, T - 1, V]`` over the model's vocabulary (``-inf`` outside the draft vocabulary)."""
    eps, theta = config["rms_norm_eps"], config.get("rope_theta", 10000.0)
    heads, kv = config["num_attention_heads"], config["num_key_value_heads"]
    dim = config.get("head_dim", config["hidden_size"] // heads)

    def w(name: str) -> torch.Tensor:
        return tensors[prefix + name]

    def rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return x * cos + torch.cat([-x[..., dim // 2:], x[..., : dim // 2]], -1) * sin

    fused = torch.cat([hidden[layer] for layer in aux], -1) @ tensors["fc.weight"].T
    source, emb = fused[:, :-1], hidden[0][:, 1:]
    batch, length, _ = source.shape
    x = torch.cat([rms(emb, w("input_layernorm.weight")), rms(source, w("hidden_norm.weight"))], -1)
    q = (x @ w("self_attn.q_proj.weight").T).view(batch, length, heads, dim).transpose(1, 2)
    k = (x @ w("self_attn.k_proj.weight").T).view(batch, length, kv, dim).transpose(1, 2)
    v = (x @ w("self_attn.v_proj.weight").T).view(batch, length, kv, dim).transpose(1, 2)
    angles = torch.arange(1, length + 1).float()[:, None] / theta ** (torch.arange(0, dim, 2).float() / dim)
    cos, sin = torch.cat([angles, angles], -1).cos(), torch.cat([angles, angles], -1).sin()
    q, k = rotate(q, cos, sin), rotate(k, cos, sin)
    k, v = k.repeat_interleave(heads // kv, 1), v.repeat_interleave(heads // kv, 1)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(dim)
    scores = scores.masked_fill(~torch.ones(length, length, dtype=torch.bool).tril(), float("-inf"))
    attended = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, heads * dim)
    a = source + attended @ w("self_attn.o_proj.weight").T
    h = rms(a, w("post_attention_layernorm.weight"))
    y = a + (F.silu(h @ w("mlp.gate_proj.weight").T) * (h @ w("mlp.up_proj.weight").T)) @ w("mlp.down_proj.weight").T
    logits = rms(y, tensors["norm.weight"]) @ tensors["lm_head.weight"].T
    ids = torch.arange(logits.shape[-1]) + tensors["d2t"]
    return logits.new_full((batch, length, VOCAB), float("-inf")).index_copy(-1, ids, logits)


@pytest.mark.parametrize("style", ["yuhuili", "vicuna13b", "speculators", "angelslim"])
def test_official_eagle3_heads_load_and_match_a_reference(style: str, tmp_path: Path) -> None:
    """``vicuna13b``: ``yuhuili/EAGLE3-Vicuna1.3-13B`` states the base model's 40 layers for its one draft layer."""
    if style in ("yuhuili", "vicuna13b"):
        layers, config, prefix, aux, safetensors = 8, YUHUILI, "midlayer.", (2, 4, 5), False
        config = {**YUHUILI, "num_hidden_layers": 40} if style == "vicuna13b" else YUHUILI
        draft_ids = _draft_ids().sort().values
        expected = Eagle3Config(head="own", draft_vocab=DRAFT, intermediate_size=96, num_heads=4, num_kv_heads=2,
                                head_dim=16, attention_bias=False, rope_theta=10000.0, rms_norm_eps=1e-5)
    elif style == "speculators":
        layers, config, prefix, aux, safetensors = 8, SPECULATORS, "layers.0.", (2, 4, 5), True
        draft_ids = _draft_ids().sort().values
        expected = Eagle3Config(head="own", draft_vocab=DRAFT, intermediate_size=96, num_heads=4, num_kv_heads=2,
                                head_dim=16, attention_bias=False, rope_theta=5e5, rms_norm_eps=1e-5)
    else:
        layers, config, prefix, aux, safetensors = 6, ANGELSLIM, "layers.0.", (1, 3, 4), True
        draft_ids = torch.arange(VOCAB)
        expected = Eagle3Config(aux_layers=aux, head="own", intermediate_size=128, num_heads=8, num_kv_heads=4,
                                head_dim=8, attention_bias=False, rope_theta=1e6, rms_norm_eps=1e-6)
    head = tmp_path / "head"
    state = _official_state(flat_config(config), len(aux), prefix, draft_ids, embedding=not safetensors)
    _write_head(head, config, state, safetensors)
    assert FORMATS.get("eagle3").detect(head) and not FORMATS.get("tskip_eagle3").detect(head)
    checkpoint = read_checkpoint(head)
    assert checkpoint.configs == [expected] and checkpoint.base_model == base_model_of(config)
    config = flat_config(config)
    base = make_tiny_model(num_layers=layers)
    model = economize.load(base, head, decoding=DecodeConfig(cycle_length=4))
    technique = model.techniques["eagle3"]
    assert technique.aux_layers == aux and torch.equal(technique.draft_ids, draft_ids)
    tensors = {k: v.to(torch.bfloat16).float() if safetensors and v.is_floating_point() else v
               for k, v in state.items()}
    ids = torch.randint(0, VOCAB, (2, 11), generator=torch.Generator().manual_seed(2))
    hidden = _hidden(base, ids, (0, *aux))
    reference = _reference_step(tensors, config, prefix, aux, hidden)
    torch.testing.assert_close(technique.draft_logits(technique.rollout(hidden, 1)[0])[:, :-1], reference,
                               rtol=1e-4, atol=1e-4)
    _assert_greedy(model, base)


def test_official_rope_scaling_is_passed_to_the_draft_layer() -> None:
    """A ``llama3`` ``rope_scaling`` (Llama-3.1 heads) reaches the draft layer's own table, which equals the one
    transformers builds for that configuration."""
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    scaling = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
               "original_max_position_embeddings": 8192}
    raw = {**YUHUILI, "rope_theta": 5e5, "rope_scaling": scaling, "max_position_embeddings": 131072}
    config = config_from_eagle3(raw)
    assert config.rope_theta == 5e5 and config.rope_scaling == scaling
    model = economize.apply(make_tiny_model(), config)
    positions = torch.arange(0, 40000, 997)[None]
    cos, sin = model.techniques["eagle3"].rotary(positions)
    hf = LlamaRotaryEmbedding(LlamaConfig(hidden_size=config.num_heads * config.head_dim,
                                          num_attention_heads=config.num_heads, head_dim=config.head_dim,
                                          max_position_embeddings=131072,
                                          rope_parameters={"rope_theta": 5e5, **scaling}))
    expected_cos, expected_sin = hf(torch.zeros(1, dtype=torch.float32), positions)
    torch.testing.assert_close(cos, expected_cos, rtol=0, atol=1e-6)
    torch.testing.assert_close(sin, expected_sin, rtol=0, atol=1e-6)
    plain = config_from_eagle3({**YUHUILI, "rope_scaling": {"rope_type": "default"}})
    assert plain.rope_scaling is None and plain.rope_theta == 1e4


def test_unsupported_official_variants_are_refused(tmp_path: Path) -> None:
    for change in (dict(num_hidden_layers=2), dict(norm_before_fc=True), dict(eagle_config={"fc_norm": True}),
                   dict(eagle_config={"use_aux_hidden_state": False}), dict(target_hidden_size=2 * HIDDEN)):
        with pytest.raises(CheckpointFormatError):
            config_from_eagle3({**YUHUILI, **change})
    with pytest.raises(ConfigError, match="rope"):
        config_from_eagle3({**YUHUILI, "rope_scaling": {"rope_type": "dynamic", "factor": 2.0}})
    state = _official_state(YUHUILI, 3, "midlayer.", torch.arange(DRAFT), embedding=False)
    assert set(state_from_eagle3(state)) >= {"fc.weight", "layer.self_attn.q_proj.weight", "d2t", "t2d"}
    bad_states = (
        {**state, "midlayer.self_attn.q_norm.weight": torch.ones(16)},
        {**state, "layers.1.mlp.up_proj.weight": torch.ones(1)},
        {**state, "extra.weight": torch.ones(1)},
        {k: v for k, v in state.items() if k != "fc.weight"},
        {**state, "d2t": torch.zeros(DRAFT - 1, dtype=torch.long)},
        {**state, "d2t": torch.zeros(DRAFT, dtype=torch.long) - torch.arange(DRAFT) % 2},
    )
    for bad in bad_states:
        with pytest.raises(CheckpointFormatError):
            state_from_eagle3(bad)
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    torch.save({}, plain / "pytorch_model.bin")
    assert not FORMATS.get("eagle3").detect(plain)



def test_other_speculators_drafters_are_not_read_as_eagle3(tmp_path: Path) -> None:
    """A ``speculators`` directory of another algorithm (RedHatAI's DFlash drafters also set ``draft_vocab_size``) is
    not detected or read as an EAGLE-3 head, so auto-detection stays unambiguous."""
    head = tmp_path / "dflash"
    head.mkdir()
    (head / "config.json").write_text(json.dumps({**SPECULATORS, "architectures": ["DFlashDraftModel"],
                                                  "speculators_model_type": "dflash"}))
    torch.save({}, head / "pytorch_model.bin")
    assert not FORMATS.get("eagle3").detect(head)
    with pytest.raises(CheckpointFormatError, match="does not describe an EAGLE-3 head"):
        FORMATS.get("eagle3").read(head)

# ---- research checkpoints (tskip --pred eagle3) ----------------------------------------------------------------------
def _tskip_run(path: Path, model: Any, cfg: dict[str, Any]) -> Path:
    """A ``jump.pt`` of ``tskip.train --pred eagle3`` holding the technique's weights under tskip's names."""
    state = {}
    for key, value in model.techniques["eagle3"].state_dict().items():
        if key == "fc.weight":
            state["mix.proj.weight"] = value
        elif key == "layer.hidden_norm.weight":
            state["pred.hidden_norm.weight"] = value
        else:
            state["pred." + key] = value
    path.mkdir()
    torch.save({"cfg": cfg, "model": "tiny/base", "state": state}, path / "jump.pt")
    return path


def test_tskip_eagle3_jump_file_loads_by_name(tmp_path: Path) -> None:
    config = Eagle3Config(aux_layers=(2, 3), head="shared", attention_bias=True, zero_source_first=True)
    source = _model(config)
    cfg = dict(m=LAYERS, u=LAYERS, r=LAYERS, mode="frozen", pred="eagle3", src="3,2", mix="fc", feed="fused")
    run = _tskip_run(tmp_path / "run", source, cfg)
    checkpoint = read_checkpoint(run, format="tskip_eagle3")
    assert checkpoint.configs == [config] and checkpoint.base_model == "tiny/base"
    assert FORMATS.get("tskip_eagle3").detect(run) and not FORMATS.get("eagle3").detect(run)
    base = make_tiny_model()
    model = economize.load(base, run, format="tskip_eagle3", decoding=DecodeConfig(cycle_length=4))
    for (name, a), (_, b) in zip(source.techniques["eagle3"].state_dict().items(),
                                 model.techniques["eagle3"].state_dict().items()):
        assert torch.equal(a, b), name
    _assert_greedy(model, base)
    for change in (dict(mode="lora"), dict(feed="state"), dict(m=2), dict(pred="mlp")):
        bad = _tskip_run(tmp_path / f"bad_{len(list(tmp_path.iterdir()))}", source, {**cfg, **change})
        with pytest.raises(CheckpointFormatError):
            read_checkpoint(bad, format="tskip_eagle3")


# ---- training ------------------------------------------------------------------------------------------------------
def test_default_objective_and_parameter_groups() -> None:
    model = _model(std=0.0)
    objective = default_objective(model)
    assert isinstance(objective, Eagle3TTT) and objective.step_decay == 0.8
    assert (objective.defaults.lr, objective.defaults.weight_decay, objective.defaults.grad_clip) == (1e-4, 0.0, 0.5)
    (group,) = objective.parameter_groups(model)
    assert group.kind == "head" and len(group.params) == len(list(model.techniques["eagle3"].parameters()))
    with pytest.raises(IncompatibleTechniquesError):
        Eagle3TTT().parameter_groups(economize.apply(make_tiny_model()))
    for bad in (dict(steps=0), dict(step_decay=0.0)):
        with pytest.raises(ValueError):
            Eagle3TTT(**bad)
    with pytest.raises(KeyError):
        Eagle3TTT(divergence="no_such_divergence")
    request = model.techniques["eagle3"].train_request(model)
    assert request.hidden_layers == (0, 2, 3)


@pytest.mark.parametrize("head", ["own", "shared"])
def test_training_time_test_equals_distill_kl_on_the_full_vocabulary(head: str) -> None:
    model = _model(Eagle3Config(head=head, train_steps=3))
    ids = torch.randint(0, VOCAB, (2, 14), generator=torch.Generator().manual_seed(4))
    with torch.enable_grad():
        ttt = _loss(model, Eagle3TTT(), ids)
        distill = _loss(model, DistillKL(step_decay=0.8), ids)
    torch.testing.assert_close(ttt["loss"].detach(), distill["loss"].detach(), rtol=1e-5, atol=1e-6)
    assert {"kl1", "kl3", "acc3"} <= set(ttt["metrics"]) and "kl4" in distill["metrics"]


def test_training_time_test_on_a_draft_vocabulary() -> None:
    """The teacher is restricted to the draft vocabulary and renormalized; positions whose teacher argmax lies outside
    it do not count."""
    model = _model(Eagle3Config(draft_vocab=DRAFT))
    technique = model.techniques["eagle3"]
    ids = torch.randint(0, VOCAB, (2, 14), generator=torch.Generator().manual_seed(4))
    length = ids.shape[1]
    with torch.enable_grad():
        out = _loss(model, Eagle3TTT(steps=1), ids)
    hidden = _hidden(model.base_model, ids, (0, 2, 3))
    student = technique.head_logits(technique.rollout(hidden, 1)[0][:, : length - 2]).log_softmax(-1)
    teacher = model.base_model(ids).logits.float().log_softmax(-1)[:, 1 : length - 1]
    inside = technique.t2d[teacher.argmax(-1)]
    assert 0 < int(inside.sum()) < inside.numel()
    restricted = teacher[..., technique.draft_ids].log_softmax(-1)
    expected = (restricted.exp() * (restricted - student)).sum(-1)[inside].mean()
    torch.testing.assert_close(out["loss"].detach(), expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(ConfigError):
        technique.train_logits(model, SimpleNamespace(input_ids=ids), SimpleNamespace(hidden={}))


@pytest.mark.parametrize("head", ["own", "shared"])
def test_distill_kl_trains_the_drafter_through_the_trainable_drafter_protocol(head: str) -> None:
    rows = torch.randint(0, VOCAB, (8, 20), generator=torch.Generator().manual_seed(6))
    model = _model(Eagle3Config(head=head, train_steps=3), std=0.0)
    technique = model.techniques["eagle3"]
    before = [p.detach().clone() for p in technique.parameters()]
    with torch.enable_grad():
        result = fit(model, rows, "distill_kl", steps=3, batch_size=8, lr=1e-2, warmup_steps=0, seed=0, log_every=1)
        assert result.steps == 3 and result.losses[-1] < result.losses[0]
        assert all(not torch.equal(a, b) for a, b in zip(before, technique.parameters()))
        technique.zero_grad(set_to_none=True)
        _loss(model, DistillKL(), rows[:2])["loss"].backward()
    for name, parameter in technique.named_parameters():
        assert parameter.grad is not None and bool(parameter.grad.abs().sum() > 0), name
    assert not any(p.requires_grad for p in model.base_model.parameters())


def test_eagle3_ttt_trains_a_draft_vocabulary() -> None:
    rows = torch.randint(0, VOCAB, (8, 20), generator=torch.Generator().manual_seed(6))
    model = _model(Eagle3Config(draft_vocab=DRAFT, train_steps=3), std=0.0)
    with torch.enable_grad():
        result = fit(model, rows, steps=3, batch_size=8, lr=1e-2, warmup_steps=0, seed=0, log_every=1)
    assert result.steps == 3 and result.losses[-1] < result.losses[0]


def test_the_layout_helpers_read_both_config_layouts() -> None:
    flat = flat_config(SPECULATORS)
    assert flat["hidden_size"] == HIDDEN and flat["draft_vocab_size"] == DRAFT and flat["rope_theta"] == 5e5
    assert flat_config(YUHUILI) == YUHUILI
    assert base_model_of(SPECULATORS) == "tiny/base"
    assert base_model_of({}) is None
    assert draft_layer_count({"midlayer.mlp.up_proj.weight": torch.zeros(1)}) == 1
    assert draft_layer_count({"layers.0.mlp.up_proj.weight": torch.zeros(1), "layers.1.x": torch.zeros(1)}) == 2
    assert draft_layer_count({"fc.weight": torch.zeros(1)}) == 0
    with pytest.raises(CheckpointFormatError):
        config_from_eagle3({**YUHUILI, "num_hidden_layers": 40})
    assert config_from_eagle3({**YUHUILI, "num_hidden_layers": 40}, draft_layers=1) == config_from_eagle3(YUHUILI)


# ---- CUDA graphs ---------------------------------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("variant", ["own", "vocab", "shared"])
@torch.no_grad()
def test_graphed_draft_steps_match_eager_and_are_replayed(variant: str) -> None:
    """Graphed draft steps (persistent cache and states, full-capacity reads) draft exactly what eager steps draft,
    also when rows leave the batch and when a longer generation outgrows the cache."""
    base = make_tiny_model().cuda()
    model = _model(VARIANTS[variant], base=base)
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
    assert isinstance(drafter, Eagle3Drafter) and drafter._step_fn.enabled
    steps = drafter._step_fn._graph
    assert steps is not None and steps.num_graphs > 0  # the draft steps themselves are graphs
    recorded = decoder.graphs.num_graphs
    for p, n in zip(prompts, limits):  # one row at a time: the batch-size-1 graphs of the shrunken batch, replayed
        assert model.decode(p, n, decoding=graphed).tokens.tolist() == reference_greedy(base, p, n).tolist()
    assert decoder.graphs.num_graphs == recorded
    long = model.decode(prompts[0], 300, decoding=graphed)  # outgrows the cache: new buffers and graphs
    assert long.tokens.tolist() == reference_greedy(base, prompts[0], 300).tolist()
    assert long.acceptance == model.decode(prompts[0], 300, decoding=DecodeConfig(cycle_length=5)).acceptance
