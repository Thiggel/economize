"""EAGLE-1/2: the official checkpoint format, a reference forward from its raw tensors, lossless chains and dynamic
trees in float32, the drafter against the training rollout, refusals, saving and training."""

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
from economize import DecodeConfig, TopPSampler
from economize.core.serialization import FORMATS, read_checkpoint
from economize.core.technique import TECHNIQUES
from economize.errors import CheckpointFormatError
from economize.runtime import ExactPass
from economize.techniques.eagle import EagleConfig, EagleDrafter, EagleTechnique
from economize.techniques.eagle.formats import config_from_eagle, state_from_eagle
from economize.techniques.hass import HassDistill
from economize.train import default_objective, fit
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, perturb, reference_greedy

LAYERS = 6
HIDDEN = 64
#: ``config.json`` of the official heads (the base model's, with one layer), reduced to the tiny model's width.
VICUNA = {"architectures": ["LlamaForCausalLM"], "hidden_size": HIDDEN, "intermediate_size": 96,
          "num_attention_heads": 4, "num_hidden_layers": 1, "rms_norm_eps": 1e-6, "vocab_size": VOCAB,
          "max_position_embeddings": 2048, "model_type": "llama", "torch_dtype": "float16"}
LLAMA3 = {**VICUNA, "num_key_value_heads": 2, "rope_theta": 5e5, "bias": False, "rms_norm_eps": 1e-5}
QWEN2 = {**VICUNA, "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "num_attention_heads": 8,
         "num_key_value_heads": 8, "rope_theta": 1e6, "qkv_bias": True}
STYLES = {  # config, fc bias, q/k/v bias, safetensors
    "vicuna": (VICUNA, True, False, False),
    "llama3": (LLAMA3, False, False, True),
    "qwen2": (QWEN2, True, True, False),
}
EXPECTED = {
    "vicuna": EagleConfig(intermediate_size=96, num_heads=4, num_kv_heads=4, head_dim=16, attention_bias=False,
                          rope_theta=1e4, rms_norm_eps=1e-6),
    "llama3": EagleConfig(fc_bias=False, intermediate_size=96, num_heads=4, num_kv_heads=2, head_dim=16,
                          attention_bias=False, rope_theta=5e5, rms_norm_eps=1e-5),
    "qwen2": EagleConfig(intermediate_size=96, num_heads=8, num_kv_heads=8, head_dim=8, attention_bias=True,
                         rope_theta=1e6, rms_norm_eps=1e-6),
}


def _official_state(config: dict[str, Any], fc_bias: bool, qkv_bias: bool, seed: int = 0) -> dict[str, torch.Tensor]:
    """A random state dict in the layout of ``yuhuili/EAGLE-*`` (``cnets1.py``): ``fc``, the layer of index 0 (no
    input norm), the embedding and a stale rotary buffer."""
    generator = torch.Generator().manual_seed(seed)
    d, inner = config["hidden_size"], config["intermediate_size"]
    heads = config["num_attention_heads"]
    kv = config.get("num_key_value_heads", heads)
    dim = d // heads

    def rand(*shape: int, std: float = 0.1) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * std

    state = {"embed_tokens.weight": rand(VOCAB, d), "fc.weight": rand(d, 2 * d)}
    if fc_bias:
        state["fc.bias"] = rand(d)
    for name, rows in (("q", heads * dim), ("k", kv * dim), ("v", kv * dim)):
        state[f"layers.0.self_attn.{name}_proj.weight"] = rand(rows, d)
        if qkv_bias:
            state[f"layers.0.self_attn.{name}_proj.bias"] = rand(rows)
    state["layers.0.self_attn.o_proj.weight"] = rand(d, heads * dim)
    state["layers.0.self_attn.rotary_emb.inv_freq"] = torch.ones(dim // 2)
    state["layers.0.mlp.gate_proj.weight"] = rand(inner, d)
    state["layers.0.mlp.up_proj.weight"] = rand(inner, d)
    state["layers.0.mlp.down_proj.weight"] = rand(d, inner)
    state["layers.0.post_attention_layernorm.weight"] = 1 + rand(d)
    return state


def _write_head(path: Path, config: dict[str, Any], state: dict[str, torch.Tensor], safetensors: bool) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config))
    if safetensors:
        from safetensors.torch import save_file

        save_file({k: v.to(torch.bfloat16) for k, v in state.items()}, path / "model.safetensors")
    else:
        torch.save(state, path / "pytorch_model.bin")
    return path


def _head(tmp_path: Path, style: str) -> tuple[Path, dict[str, torch.Tensor]]:
    """An official head of ``style`` on disk and its tensors as read back (bfloat16-rounded for safetensors)."""
    config, fc_bias, qkv_bias, safetensors = STYLES[style]
    state = _official_state(config, fc_bias, qkv_bias)
    head = _write_head(tmp_path / style, config, state, safetensors)
    return head, {k: v.to(torch.bfloat16).float() if safetensors else v for k, v in state.items()}


def _states(base: torch.nn.Module, ids: torch.Tensor) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """economize's teacher states ``{0: embeddings, L: last block's output}`` (before the final norm) and HF's last
    hidden state (after it), the feature of the official implementation."""
    captured: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, _args: Any, output: Any) -> None:
        captured.append(output[0] if isinstance(output, tuple) else output)

    handle = base.model.layers[-1].register_forward_hook(hook)
    try:
        out = base(ids, output_hidden_states=True)
    finally:
        handle.remove()
    return {0: out.hidden_states[0], LAYERS: captured[0]}, out.hidden_states[-1]


def _reference_step(tensors: dict[str, torch.Tensor], config: dict[str, Any], feature: torch.Tensor,
                    embedding: torch.Tensor, lm_head: torch.Tensor) -> torch.Tensor:
    """EAGLE-1's first draft step at positions ``1..T-1`` (SafeAILab ``cnets1.py``, ``Model.forward`` and the base
    model's LM head), written out from the raw tensors: logits ``[B, T - 1, V]``."""
    eps, theta = config["rms_norm_eps"], config.get("rope_theta", 10000.0)
    heads = config["num_attention_heads"]
    kv = config.get("num_key_value_heads", heads)
    dim = config["hidden_size"] // heads

    def linear(x: torch.Tensor, name: str) -> torch.Tensor:
        bias = tensors.get(name + ".bias")
        return x @ tensors[name + ".weight"].T + (0 if bias is None else bias)

    def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return x * cos + torch.cat([-x[..., dim // 2:], x[..., : dim // 2]], -1) * sin

    x = linear(torch.cat([embedding[:, 1:], feature[:, :-1]], -1), "fc")
    batch, length, _ = x.shape
    q = linear(x, "layers.0.self_attn.q_proj").view(batch, length, heads, dim).transpose(1, 2)
    k = linear(x, "layers.0.self_attn.k_proj").view(batch, length, kv, dim).transpose(1, 2)
    v = linear(x, "layers.0.self_attn.v_proj").view(batch, length, kv, dim).transpose(1, 2)
    angles = torch.arange(1, length + 1).float()[:, None] / theta ** (torch.arange(0, dim, 2).float() / dim)
    cos, sin = torch.cat([angles, angles], -1).cos(), torch.cat([angles, angles], -1).sin()
    q, k = rotate(q, cos, sin), rotate(k, cos, sin)
    k, v = k.repeat_interleave(heads // kv, 1), v.repeat_interleave(heads // kv, 1)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(dim)
    scores = scores.masked_fill(~torch.ones(length, length, dtype=torch.bool).tril(), float("-inf"))
    attended = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, heads * dim)
    a = x + linear(attended, "layers.0.self_attn.o_proj")
    norm = tensors["layers.0.post_attention_layernorm.weight"]
    h = norm * a * torch.rsqrt(a.pow(2).mean(-1, keepdim=True) + eps)
    y = a + linear(F.silu(linear(h, "layers.0.mlp.gate_proj")) * linear(h, "layers.0.mlp.up_proj"),
                   "layers.0.mlp.down_proj")
    return y @ lm_head.T


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


def _tree(cycle_length: int = 5, topk: int = 3, total_tokens: int = 12) -> DecodeConfig:
    return DecodeConfig(cycle_length=cycle_length, draft_policy="dynamic_tree",
                        draft_policy_options={"depth": cycle_length - 1, "topk": topk, "total_tokens": total_tokens})


@pytest.fixture(scope="module")
def trained() -> SimpleNamespace:
    """A new head fitted with its default objective to the tiny model's greedy continuations of four prompts."""
    generator = torch.Generator().manual_seed(1)
    prompts = [torch.randint(1, VOCAB, (PROMPT_LEN,), generator=generator) for _ in range(4)]
    base = make_tiny_model()
    with torch.no_grad():
        rows = torch.stack([torch.cat([p, reference_greedy(base, p, 28)]) for p in prompts])
    model = economize.apply(make_tiny_model(), EagleConfig(), decoding=DecodeConfig(cycle_length=4))
    with torch.enable_grad():
        fit(model, rows, steps=60, batch_size=4, lr=1e-2, warmup_steps=0, seed=0)
    return SimpleNamespace(model=model, base=base, prompts=prompts, rows=rows)


# ---- configuration -------------------------------------------------------------------------------------------------
def test_registration_and_parameters() -> None:
    assert TECHNIQUES.get("eagle") is EagleTechnique and "eagle" in FORMATS.names()
    config = EagleConfig(fc_bias=False, attention_bias=True, num_heads=8, num_kv_heads=2)
    assert EagleConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    technique = economize.apply(make_tiny_model(), config).techniques["eagle"]
    names = {name for name, _ in technique.named_parameters()}
    assert "layer.fc.weight" in names and "layer.fc.bias" not in names and "layer.self_attn.q_proj.bias" in names
    assert not any("input_layernorm" in name or name.startswith(("norm.", "lm_head.")) for name in names)
    assert technique.layer.fc.weight.shape == (HIDDEN, 2 * HIDDEN) and technique.source_layers == (LAYERS,)
    assert all(p.dtype == torch.float32 for p in technique.parameters())
    drafter = technique.drafter(None)
    assert isinstance(drafter, EagleDrafter) and drafter.observed_layers == (0, LAYERS)


# ---- official checkpoints ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("style", list(STYLES))
def test_official_heads_load_and_match_a_reference(style: str, tmp_path: Path) -> None:
    head, tensors = _head(tmp_path, style)
    assert [name for name in FORMATS.names() if FORMATS.get(name).detect(head)] == ["eagle"]
    checkpoint = read_checkpoint(head)
    assert checkpoint.configs == [EXPECTED[style]] and checkpoint.base_model is None
    base = make_tiny_model()
    model = economize.load(base, head, decoding=DecodeConfig(cycle_length=4))
    technique = model.techniques["eagle"]
    ids = torch.randint(0, VOCAB, (2, 11), generator=torch.Generator().manual_seed(2))
    hidden, feature = _states(base, ids)
    reference = _reference_step(tensors, STYLES[style][0], feature, hidden[0], base.lm_head.weight)
    torch.testing.assert_close(technique.head_logits(technique.rollout(hidden, 1)[0])[:, :-1], reference,
                               rtol=1e-4, atol=1e-4)
    _assert_greedy(model, base)
    _assert_greedy(model, base, decoding=_tree())


@pytest.mark.parametrize("cycle_length", [1, 3, 6])
def test_greedy_chains_and_trees_equal_plain_greedy(cycle_length: int, tmp_path: Path) -> None:
    head, _ = _head(tmp_path, "qwen2")
    base = make_tiny_model()
    model = economize.load(base, head, decoding=DecodeConfig(cycle_length=cycle_length))
    results = _assert_greedy(model, base)
    assert all(k <= cycle_length - 1 for r in results for k, _ in r.acceptance)
    if cycle_length > 1:
        _assert_greedy(model, base, decoding=_tree(cycle_length, topk=2, total_tokens=6))


def test_drafter_steps_equal_the_training_rollout() -> None:
    """After exact passes over positions ``0..12``, draft step ``i`` (position ``12 + i``) equals rollout step ``i``
    at position 12."""
    model = economize.apply(make_tiny_model(), EagleConfig())
    technique = model.techniques["eagle"]
    perturb(technique, 0.05, seed=1)
    ids = torch.randint(0, VOCAB, (2, 16), generator=torch.Generator().manual_seed(5))
    hidden, _ = _states(model.base_model, ids)
    expected = technique.rollout(hidden, 3)
    drafter = technique.drafter(None)
    drafter.begin(None, SimpleNamespace(capacity=64, num_slots=65), 2)
    for start, end in ((0, 10), (10, 13)):
        context = SimpleNamespace(positions=torch.arange(start, end).expand(2, -1), valid=None)
        exact = ExactPass(hidden[LAYERS][:, start:end], {k: v[:, start:end] for k, v in hidden.items()}, context)
        drafter.observe(exact, torch.full((2,), end - start - 1))
    for i in range(1, 4):
        logits = drafter.step(ids[:, 12 + i], torch.full((2,), 12 + i), None)
        torch.testing.assert_close(logits, technique.head_logits(expected[i - 1][:, 12]), rtol=1e-5, atol=1e-5)


def test_other_heads_are_not_read_as_eagle(tmp_path: Path) -> None:
    """Only EAGLE-1/2 layouts are detected: a full base model, an EAGLE-3 head, a DFlash or Medusa directory and a
    head of two layers are not."""
    config, fc_bias, qkv_bias, _ = STYLES["vicuna"]
    state = _official_state(config, fc_bias, qkv_bias)
    eagle3 = {**{k: v for k, v in state.items() if not k.startswith("fc.")}, "fc.weight": torch.zeros(HIDDEN, 192)}
    full = {**{k: v for k, v in state.items() if not k.startswith("fc.")}, "norm.weight": torch.ones(HIDDEN),
            "lm_head.weight": torch.zeros(VOCAB, HIDDEN)}
    cases = {
        "eagle3_weights": (config, eagle3),
        "eagle3_config": ({**config, "draft_vocab_size": 96}, state),
        "dflash": ({**config, "dflash_config": {}}, state),
        "medusa": ({**config, "medusa_num_heads": 3}, state),
        "two_layers": ({**config, "num_hidden_layers": 2}, state),
        "base_model": ({**config, "num_hidden_layers": 6}, full),
        "own_head": (config, {**state, "lm_head.weight": torch.zeros(VOCAB, HIDDEN)}),
    }
    for name, (raw, weights) in cases.items():
        head = _write_head(tmp_path / name, raw, weights, safetensors=False)
        assert not FORMATS.get("eagle").detect(head), name
    with pytest.raises(CheckpointFormatError, match="does not describe an EAGLE-1/2 head"):
        FORMATS.get("eagle").read(tmp_path / "medusa")
    garbage = tmp_path / "garbage"
    garbage.mkdir()
    (garbage / "config.json").write_text(json.dumps(config))
    (garbage / "pytorch_model.bin").write_bytes(b"not a checkpoint")
    assert not FORMATS.get("eagle").detect(garbage)


def test_unsupported_variants_are_refused() -> None:
    config, fc_bias, qkv_bias, _ = STYLES["vicuna"]
    state = _official_state(config, fc_bias, qkv_bias)
    renamed = state_from_eagle(state)
    assert set(renamed) == {f"layer.{k.split('.', 2)[2] if k.startswith('layers.') else k}" for k in state
                            if not k.startswith("embed_tokens.") and "rotary_emb" not in k}
    assert config_from_eagle(config, renamed) == EXPECTED["vicuna"]
    zero_bias = {**state, "layers.0.self_attn.o_proj.bias": torch.zeros(HIDDEN)}
    assert set(state_from_eagle(zero_bias)) == set(renamed)
    for bad in (
        {**state, "layers.0.input_layernorm.weight": torch.ones(HIDDEN)},
        {**state, "layers.1.mlp.up_proj.weight": torch.ones(1)},
        {**state, "layers.0.self_attn.o_proj.bias": torch.ones(HIDDEN)},
        {**state, "extra.weight": torch.ones(1)},
        {k: v for k, v in state.items() if k != "fc.weight"},
    ):
        with pytest.raises(CheckpointFormatError):
            state_from_eagle(bad)


# ---- training and saving -------------------------------------------------------------------------------------------
def test_the_default_objective_is_hass() -> None:
    model = economize.apply(make_tiny_model(), EagleConfig())
    objective = default_objective(model)
    assert isinstance(objective, HassDistill) and objective.technique == "eagle"
    (group,) = objective.parameter_groups(model)
    assert len(group.params) == len(list(model.techniques["eagle"].parameters()))


@pytest.mark.parametrize("decoding", ["chain", "tree"])
def test_a_trained_head_is_accepted_and_lossless(trained: SimpleNamespace, decoding: str) -> None:
    config = DecodeConfig(cycle_length=4) if decoding == "chain" else _tree(4)
    results = _assert_greedy(trained.model, trained.base, trained.prompts, decoding=config)
    assert sum(a for r in results for _, a in r.acceptance) > 0


def test_rejection_sampling_is_greedy_at_near_zero_temperature_and_runs(trained: SimpleNamespace) -> None:
    config = DecodeConfig(cycle_length=4, verifier="rejection")
    _assert_greedy(trained.model, trained.base, trained.prompts, decoding=config, selector=TopPSampler(1e-6, seed=0))
    result = trained.model.decode(trained.prompts[0], 24, decoding=config, selector=TopPSampler(1.0, seed=0))
    assert result.tokens.shape == (24,) and int(result.tokens.max()) < VOCAB
    assert sum(k for k, _ in result.acceptance) > 0


def test_cost_log_has_one_draft_layer_event_per_draft_step(trained: SimpleNamespace) -> None:
    result = trained.model.decode(trained.prompts[1], 30)
    events = result.cost.events
    drafts = [e for e in events if e.kind in ("draft", "draft_wasted")]
    assert {e.kind for e in events} <= {"exact", "draft", "draft_wasted", "draft_observe"}
    assert all((e.tokens, e.layer_tokens, e.module) == (1, 1, "eagle") for e in drafts)
    observe = [e for e in events if e.kind == "draft_observe"]
    assert observe and all(e.module == "eagle" and e.layer_tokens == e.tokens for e in observe)
    assert all((e.first_layer, e.last_layer) == (1, LAYERS) for e in events if e.kind == "exact")
    assert len(drafts) == sum(k for k, _ in result.acceptance)


def test_eagle_on_a_pruned_model_equals_its_greedy(trained: SimpleNamespace, calib: torch.Tensor) -> None:
    from economize.techniques.prune import PruneConfig

    base = make_tiny_model()
    model = economize.apply(base, PruneConfig(method="wanda", sparsity="2:4"), EagleConfig(),
                            decoding=DecodeConfig(cycle_length=4), calibration=calib)
    assert list(model.techniques) == ["prune", "eagle"]
    model.techniques["eagle"].load_state_dict(trained.model.techniques["eagle"].state_dict())
    _assert_greedy(model, base, trained.prompts)


def test_save_and_load_round_trip(trained: SimpleNamespace, tmp_path: Path) -> None:
    economize.save(trained.model, tmp_path / "ckpt")
    loaded = economize.load(make_tiny_model(), tmp_path / "ckpt")
    assert loaded.drafting_technique.config == trained.model.drafting_technique.config
    for (name, a), (_, b) in zip(trained.model.techniques["eagle"].state_dict().items(),
                                 loaded.techniques["eagle"].state_dict().items()):
        assert torch.equal(a, b), name
    first, second = trained.model.decode(trained.prompts[2], 20), loaded.decode(trained.prompts[2], 20)
    assert torch.equal(first.tokens, second.tokens) and first.acceptance == second.acceptance
