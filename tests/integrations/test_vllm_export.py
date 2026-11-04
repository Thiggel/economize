"""The exported jump drafter and the vLLM draft model class, checked on CPU against the economize technique.

None of these tests needs vLLM: the draft model class does not import it, and its arithmetic (predicted state
after layer ``m``, the jump path, readout, final norm, LM head) must equal the technique's to float32 precision.
The jump path runs in :class:`vllm_fakes.VllmStyleLayer`, a decoder layer laid out as vLLM lays it out (fused
projections with shard-wise weight loaders, ``(hidden, residual)`` interface).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import load_file
from vllm_fakes import reference_jump_step, vllm_target

from economize.backends.vllm import DRAFTER_ARCHITECTURE, drafter_config, drafter_weights, export_drafter
from economize.backends.vllm.drafters import SYNC_PREFIX
from economize.backends.vllm.jump.model import EconomizeJumpEagle3ForCausalLM
from economize.errors import ConfigError
from economize.techniques.jump.export import CONFIG_KEY, ExportedJumpDrafter, JumpDrafterSpec
from helpers import PROMPT_LEN, VOCAB, make_tiny_model

#: exit below the last layer (readout adapter) and three source layers (mixer)
RICH = {"target_layer": 2, "source_layer": 4, "exit_layer": 5, "extra_source_layers": (0, 1)}


@pytest.fixture
def exported(applied_jump: Callable[..., Any], tmp_path: Path) -> tuple[Any, Any, Path]:
    """(base model, EfficientModel, export folder) of a frozen drafter with a readout and a mixer."""
    base = make_tiny_model(seed=0)
    model = applied_jump(base=base, **RICH)
    return base, model, export_drafter(model, tmp_path / "drafter")


def _inputs(technique: Any, seed: int = 0) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    d = technique.adapter.hidden_size
    states = [torch.randn(PROMPT_LEN, d, generator=generator) for _ in technique.source_layers]
    ids = torch.randint(0, VOCAB, (PROMPT_LEN,), generator=generator)
    return states, ids, torch.randn(PROMPT_LEN, d, generator=generator)


def _vllm_config(folder: Path) -> SimpleNamespace:
    """The parts of a ``VllmConfig`` the draft model reads (the hf_config is the exported config.json)."""
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(folder)
    draft = SimpleNamespace(hf_config=hf_config, architectures=list(hf_config.architectures))
    return SimpleNamespace(speculative_config=SimpleNamespace(draft_model_config=draft))


def _target(base: Any) -> SimpleNamespace:
    return SimpleNamespace(model=base.model, lm_head=base.lm_head, logits_processor=lambda head, h: head(h))


# ---- export --------------------------------------------------------------------------------------------------------
def test_export_writes_only_drafter_tensors(exported: tuple[Any, Any, Path]) -> None:
    _, model, folder = exported
    technique = model.techniques["jump"]
    tensors = load_file(str(folder / "model.safetensors"))
    assert set(tensors) == {"jump." + name for name in technique.state_dict()}
    assert not any(name.startswith(("jump.model.", "jump.lm_head", "model.", "lm_head")) for name in tensors)
    assert all(t.dtype == torch.float32 for t in tensors.values())

    config = json.loads((folder / "config.json").read_text())
    assert config["architectures"] == [DRAFTER_ARCHITECTURE] and DRAFTER_ARCHITECTURE.endswith("Eagle3")
    assert config["eagle_aux_hidden_state_layer_ids"] == [0, 1, 4]
    block = config[CONFIG_KEY]
    assert (block["target_layer"], block["exit_layer"], block["num_layers"]) == (2, 5, 6)
    assert block["shared_with_target"] == ["model.layers.2", "model.layers.3", "model.layers.4", "model.norm",
                                           "model.embed_tokens", "lm_head"]
    assert "auto_map" not in config and config["hidden_size"] == technique.adapter.hidden_size


def test_speculators_layout(applied_jump: Callable[..., Any], tmp_path: Path) -> None:
    model = applied_jump()
    config = drafter_config(model, format="speculators", num_speculative_tokens=3, verifier="tiny/qwen2")
    assert config["speculators_model_type"] == "eagle3"
    methods = config["speculators_config"]["proposal_methods"]
    assert methods[0]["speculative_tokens"] == 3
    assert config["speculators_config"]["verifier"]["name_or_path"] == "tiny/qwen2"
    assert config["transformer_layer_config"][CONFIG_KEY] == config[CONFIG_KEY]
    assert JumpDrafterSpec.from_hf_config(config) == JumpDrafterSpec.of(model.techniques["jump"])
    with pytest.raises(ConfigError, match="unknown export format"):
        drafter_config(model, format="gguf")  # type: ignore[arg-type]


def test_spec_round_trip_and_version_guard(applied_jump: Callable[..., Any]) -> None:
    spec = JumpDrafterSpec.of(applied_jump(**RICH).techniques["jump"])
    assert JumpDrafterSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    assert list(spec.jump_layers) == [3, 4, 5]
    with pytest.raises(ConfigError, match="newer"):
        JumpDrafterSpec.from_dict(spec.to_dict() | {"format_version": 99})
    with pytest.raises(ConfigError, match="no 'economize_jump' block"):
        JumpDrafterSpec.from_hf_config({"architectures": ["Qwen2ForCausalLM"]})
    assert JumpDrafterSpec.from_hf_config({"effkit_jump": spec.to_dict()}) == spec  # exported before the rename
    assert JumpDrafterSpec.from_hf_config({"transformer_layer_config": {CONFIG_KEY: spec.to_dict()}}) == spec


# ---- the backend-neutral drafter -----------------------------------------------------------------------------------
def test_exported_drafter_matches_technique(exported: tuple[Any, Any, Path]) -> None:
    _, model, folder = exported
    technique = model.techniques["jump"]
    drafter = ExportedJumpDrafter.from_pretrained(folder)
    states, _, emb = _inputs(technique)
    with torch.no_grad():
        expected = technique.predictor(technique.source(states, emb), emb)
        torch.testing.assert_close(drafter.predict(states, emb), expected, rtol=0, atol=0)
        torch.testing.assert_close(drafter.readout_hidden(emb), technique.readout(emb), rtol=0, atol=0)
    with pytest.raises(ConfigError, match="expected 3 source states"):
        drafter.split_sources(emb)


@pytest.mark.parametrize("name", ["model.layers.3.mlp.up_proj.weight", "lm_head.weight", "jump.model.norm.weight"])
def test_drafter_refuses_target_weights(exported: tuple[Any, Any, Path], name: str) -> None:
    drafter = ExportedJumpDrafter.from_pretrained(exported[2])
    with pytest.raises(ConfigError, match="not a drafter tensor"):
        drafter.load_weights([(name, torch.zeros(1))])


@pytest.mark.parametrize("adaptation,head", [("lora", "lora."), ("full", "blocks.")])
def test_adapted_paths_export_their_tensors(applied_jump: Callable[..., Any], tmp_path: Path, adaptation: str,
                                            head: str) -> None:
    folder = export_drafter(applied_jump(adaptation=adaptation), tmp_path / adaptation)
    names = set(load_file(str(folder / "model.safetensors")))
    assert any(name.startswith("jump." + head) for name in names)
    drafter = ExportedJumpDrafter.from_pretrained(folder)
    assert drafter.adapted_tensors and all(name.startswith(head) for name in drafter.adapted_tensors)


# ---- the vLLM draft model class ------------------------------------------------------------------------------------
def test_vllm_model_loads_like_vllm(exported: tuple[Any, Any, Path]) -> None:
    """vLLM's DefaultModelLoader requires load_weights to return every parameter name of the module."""
    base, _, folder = exported
    draft = EconomizeJumpEagle3ForCausalLM(vllm_config=_vllm_config(folder))
    loaded = draft.load_weights(load_file(str(folder / "model.safetensors")).items())
    assert loaded == {name for name, _ in draft.named_parameters()}
    draft.tie_to_target(_target(base))
    assert {name for name, _ in draft.named_parameters()} == loaded  # the target's modules are not registered
    assert all(name.startswith("jump.") for name in draft.state_dict())


def test_vllm_model_matches_technique(exported: tuple[Any, Any, Path]) -> None:
    base, model, folder = exported
    technique = model.techniques["jump"]
    draft = EconomizeJumpEagle3ForCausalLM(vllm_config=_vllm_config(folder))
    draft.load_weights(load_file(str(folder / "model.safetensors")).items())
    with pytest.raises(ConfigError, match="not tied"):
        draft.compute_logits(torch.zeros(1, technique.adapter.hidden_size))
    tied = draft.tie_to_target(_target(base))
    assert [id(layer) for layer in tied.layers] == [id(base.model.layers[i]) for i in (2, 3, 4)]

    states, ids, hidden = _inputs(technique, seed=1)
    with torch.no_grad():
        predicted = draft.predict_target_state(ids, torch.cat(states, dim=-1))
        torch.testing.assert_close(predicted, technique.predict(states, technique.adapter.embed(ids)), rtol=0, atol=0)
        torch.testing.assert_close(draft.compute_logits(hidden), technique.logits(hidden), rtol=1e-6, atol=1e-6)
    with pytest.raises(NotImplementedError):
        draft(ids, torch.arange(len(ids)))


def test_weight_sync_reaches_the_drafter_without_copy(exported: tuple[Any, Any, Path]) -> None:
    """An RL weight sync copies new policy weights into the target in place; the tied drafter sees them."""
    base, _, folder = exported
    draft = EconomizeJumpEagle3ForCausalLM(vllm_config=_vllm_config(folder))
    draft.load_weights(load_file(str(folder / "model.safetensors")).items())
    draft.tie_to_target(_target(base))
    hidden = torch.randn(3, base.config.hidden_size, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        before = draft.compute_logits(hidden)
        base.model.norm.weight.mul_(1.5)  # what vLLM's load_weights does to the target during a sync
        after = draft.compute_logits(hidden)
        expected = base.lm_head(base.model.norm(draft.jump.readout_hidden(hidden)))
    assert not torch.allclose(before, after)
    torch.testing.assert_close(after, expected, rtol=1e-6, atol=1e-6)


def test_tie_checks_the_layer_count(exported: tuple[Any, Any, Path]) -> None:
    draft = EconomizeJumpEagle3ForCausalLM(vllm_config=_vllm_config(exported[2]))
    with pytest.raises(ConfigError, match="exported for 6 layers"):
        draft.tie_to_target(_target(make_tiny_model(num_layers=4)))


# ---- the jump path in vLLM-style layers ----------------------------------------------------------------------------
def _tied(folder: Path, target: SimpleNamespace) -> EconomizeJumpEagle3ForCausalLM:
    draft = EconomizeJumpEagle3ForCausalLM(vllm_config=_vllm_config(folder))
    draft.load_weights(load_file(str(folder / "model.safetensors")).items())
    draft.tie_to_target(target)
    return draft


def test_layer_buffers_load_like_parameters() -> None:
    """Gemma-4's vLLM layer keeps ``layer_scalar`` as a buffer; a full jump path must find, copy and load it."""
    from economize.backends.vllm.jump.model import _copy_layer, _layer_tensors, _load, _resolve

    target = vllm_target(make_tiny_model(seed=0))
    layer = target.model.layers[0]
    layer.register_buffer("layer_scalar", torch.ones(1))
    copy = _copy_layer(layer)
    assert copy.layer_scalar is not layer.layer_scalar
    tensors = _layer_tensors(copy)
    name, shard = _resolve(tensors, "layer_scalar")
    assert name == "layer_scalar" and shard is None
    _load(tensors[name], torch.full((1,), 0.5), shard)
    assert copy.layer_scalar.item() == 0.5 and layer.layer_scalar.item() == 1.0
    name, shard = _resolve(tensors, "self_attn.q_proj.weight")  # fused parameters still resolve
    assert name.endswith("qkv_proj.weight") and shard == "q"


def _draft_step(draft: EconomizeJumpEagle3ForCausalLM, ids: torch.Tensor,
                states: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        hidden, sources = draft.draft_forward(ids, torch.arange(len(ids)), torch.cat(states, dim=-1))
        return draft.compute_logits(hidden), sources


@pytest.mark.parametrize("adaptation", ["frozen", "lora", "full"])
def test_vllm_draft_step_matches_technique(applied_jump: Callable[..., Any], tmp_path: Path, adaptation: str) -> None:
    base = make_tiny_model(seed=0)
    model = applied_jump(base=base, adaptation=adaptation, **RICH)
    technique = model.techniques["jump"]
    target = vllm_target(base)
    weights = {name: p.clone() for name, p in target.model.named_parameters()}
    draft = _tied(export_drafter(model, tmp_path / adaptation), target)

    path, layers = draft.jump_path, draft.target.layers
    if adaptation == "frozen":
        assert all(a is b for a, b in zip(path, layers))
    else:  # own weights, the target's attention (hence its KV cache) and rotary tables
        assert all(a is not b and a.self_attn.attn is b.self_attn.attn for a, b in zip(path, layers))
        assert all(a.self_attn.rotary_emb is b.self_attn.rotary_emb for a, b in zip(path, layers))
        own = {id(p) for layer in path for p in layer.parameters()}
        assert own.isdisjoint(id(p) for p in target.model.parameters())
        assert hasattr(path[0].self_attn.qkv_proj.weight, "weight_loader")

    states, ids, _ = _inputs(technique, seed=3)
    logits, sources = _draft_step(draft, ids, states)
    expected_logits, expected_sources = reference_jump_step(technique, ids, states)
    torch.testing.assert_close(logits, expected_logits, rtol=1e-4, atol=1e-4)  # merged vs hooked LoRA
    torch.testing.assert_close(sources, torch.cat(expected_sources, dim=-1), rtol=1e-4, atol=1e-4)
    # the draft K/V of every jump-path layer went to the target's attention; the target's weights are untouched
    assert [len(layer.self_attn.attn.kv_cache) for layer in layers] == [1, 1, 1]
    assert all(torch.equal(p, weights[name]) for name, p in target.model.named_parameters())


@pytest.mark.parametrize("adaptation", ["frozen", "lora", "full"])
def test_vllm_jump_path_after_a_weight_sync(applied_jump: Callable[..., Any], tmp_path: Path, adaptation: str) -> None:
    """``frozen`` follows the policy by reference, ``lora`` after :meth:`refresh_jump_path`, ``full`` keeps its copy."""
    base = make_tiny_model(seed=0)
    model = applied_jump(base=base, adaptation=adaptation, **RICH)
    technique = model.techniques["jump"]
    target = vllm_target(base)
    draft = _tied(export_drafter(model, tmp_path / adaptation), target)
    states, ids, _ = _inputs(technique, seed=4)
    before, _ = _draft_step(draft, ids, states)

    with torch.no_grad():  # a policy update, synced into the vLLM target in place
        for layer in base.model.layers:
            for linear in (layer.self_attn.q_proj, layer.self_attn.v_proj, layer.mlp.up_proj, layer.mlp.down_proj):
                linear.weight.mul_(1.25)
        for fake, layer in zip(target.model.layers, base.model.layers):
            fake.load_from(layer)
    draft.refresh_jump_path()
    after, _ = _draft_step(draft, ids, states)
    if adaptation == "full":
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        return
    expected, _ = reference_jump_step(technique, ids, states)
    assert not torch.allclose(after, before, atol=1e-3)
    torch.testing.assert_close(after, expected, rtol=1e-4, atol=1e-4)


def test_layer_copy_keeps_vllm_parameter_attributes() -> None:
    from economize.backends.vllm.jump.model import _copy_layer

    target = vllm_target(make_tiny_model(seed=0))
    layer = target.model.layers[0]
    copy = _copy_layer(layer)
    for (name, original), (_, cloned) in zip(layer.named_parameters(), copy.named_parameters()):
        assert cloned is not original and torch.equal(cloned, original) and not cloned.requires_grad, name
        assert cloned.__dict__.keys() == original.__dict__.keys(), name
    assert copy.self_attn.attn is layer.self_attn.attn


# ---- RL weight syncs through the target's load_weights -------------------------------------------------------------
def _policy_update(base: Any, scale: float = 1.25) -> list[tuple[str, torch.Tensor]]:
    """New policy weights of every layer, in Hugging Face names (what TRL and verl push into vLLM)."""
    return [(name, param.detach() * scale) for name, param in base.named_parameters()
            if name.startswith("model.layers.") and name.endswith(("q_proj.weight", "up_proj.weight"))]


@pytest.mark.parametrize("adaptation", ["frozen", "lora", "full"])
def test_policy_sync_marks_the_jump_path(applied_jump: Callable[..., Any], tmp_path: Path, adaptation: str) -> None:
    """A sync of the target reaches ``frozen`` by reference and makes ``lora`` stale; ``full`` keeps its weights."""
    base = make_tiny_model(seed=0)
    model = applied_jump(base=base, adaptation=adaptation, **RICH)
    technique = model.techniques["jump"]
    target = vllm_target(base)
    draft = _tied(export_drafter(model, tmp_path / adaptation), target)
    states, ids, _ = _inputs(technique, seed=5)
    before, _ = _draft_step(draft, ids, states)

    loaded = target.load_weights(_policy_update(base))
    assert all(name.startswith("model.layers.") for name in loaded)
    assert draft.stale is (adaptation == "lora")
    assert draft.refresh_if_stale() is (adaptation == "lora")
    assert not draft.stale and not draft.refresh_if_stale()
    assert draft.refreshes == (adaptation == "lora")
    after, _ = _draft_step(draft, ids, states)
    if adaptation == "full":
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        return
    expected, _ = reference_jump_step(technique, ids, states)
    assert not torch.allclose(after, before, atol=1e-3)
    torch.testing.assert_close(after, expected, rtol=1e-4, atol=1e-4)


def test_sync_outside_the_jump_path_keeps_lora_fresh(applied_jump: Callable[..., Any], tmp_path: Path) -> None:
    base = make_tiny_model(seed=0)
    model = applied_jump(base=base, adaptation="lora", **RICH)
    target = vllm_target(base)
    draft = _tied(export_drafter(model, tmp_path / "lora"), target)
    first = draft.spec.jump_layers[0] - 1  # Hugging Face index of block m+1
    outside = [(name, tensor) for name, tensor in _policy_update(base)
               if int(name.split(".")[2]) < first]
    assert outside
    target.load_weights(outside)
    assert not draft.stale


@pytest.mark.parametrize("adaptation", ["frozen", "lora", "full"])
def test_redistilled_drafter_rides_the_policy_sync(applied_jump: Callable[..., Any], tmp_path: Path,
                                                   adaptation: str) -> None:
    """``drafter_weights`` tensors go to the drafter (never to the target) and rebuild an adapted jump path."""
    base = make_tiny_model(seed=0)
    model = applied_jump(base=base, adaptation=adaptation, **RICH)
    technique = model.techniques["jump"]
    target = vllm_target(base)
    draft = _tied(export_drafter(model, tmp_path / adaptation), target)
    states, ids, _ = _inputs(technique, seed=6)
    before, _ = _draft_step(draft, ids, states)

    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():  # a re-distillation step of the drafter
        for param in technique.parameters():
            if param.requires_grad:
                param.add_(torch.randn(param.shape, generator=generator) * 0.05)
    pushed = drafter_weights(model)
    assert pushed and all(name.startswith(SYNC_PREFIX + "jump.") for name, _ in pushed)
    loaded = target.load_weights(_policy_update(base, 1.0) + pushed)  # KeyError if a drafter tensor hit the target
    assert {name for name, _ in pushed} <= loaded
    assert draft.stale is (adaptation != "frozen")
    draft.refresh_if_stale()
    after, _ = _draft_step(draft, ids, states)
    expected, _ = reference_jump_step(technique, ids, states)
    assert not torch.allclose(after, before, atol=1e-3)
    torch.testing.assert_close(after, expected, rtol=1e-4, atol=1e-4)


def test_weight_sync_hook_is_installed_once(exported: tuple[Any, Any, Path]) -> None:
    base, _, folder = exported
    target = vllm_target(base)
    draft = _tied(folder, target)
    hook = target.load_weights
    draft.tie_to_target(target)
    assert target.load_weights is hook and hook.economize_drafter is draft
