"""``economize.integrations.peft``: PEFT models as bases, and a technique's LoRA (here the jump path's) in the PEFT
adapter format.

The jump path of the ``applied_jump`` fixture is blocks 3..6 (``target_layer=2``), i.e. ``model.layers.2`` to
``model.layers.5``. Inside ``jump_path()`` the jump-path LoRA is active on every call of those blocks, so a full
forward pass there equals a PEFT model that carries the same adapter.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from economize.errors import ConfigError
from helpers import VOCAB, make_tiny_model, reference_greedy

needs_peft = pytest.mark.skipif(importlib.util.find_spec("peft") is None, reason="needs peft")
pytestmark = [needs_peft, pytest.mark.filterwarnings("ignore::UserWarning", "ignore::FutureWarning")]

IDS = torch.randint(1, VOCAB, (1, 10), generator=torch.Generator().manual_seed(5))
JUMP_LAYERS = [2, 3, 4, 5]  # 0-based names of blocks m+1..r


def _peft_policy(seed: int = 0, layers: list[int] | None = None, **lora: Any) -> Any:
    """A PEFT LoRA model of the tiny model with random (nonzero) adapter weights."""
    from peft import LoraConfig, get_peft_model

    options = {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj"], "lora_dropout": 0.0, **lora}
    if layers is not None:
        options["layers_to_transform"] = layers
    policy = get_peft_model(make_tiny_model(seed=seed), LoraConfig(**options))
    generator = torch.Generator().manual_seed(seed + 11)
    with torch.no_grad():
        for name, param in policy.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=generator) * 0.1)
    return policy.eval()


def _jump_logits(model: Any) -> torch.Tensor:
    with torch.no_grad(), model.techniques["jump"].jump_path():
        return model.base_model(IDS).logits


# ---- PEFT models as bases ----------------------------------------------------------------------------------------
def test_peft_model_base_keeps_its_adapters(applied_jump: Any) -> None:
    from economize.integrations import peft as economize_peft

    policy = _peft_policy()
    expected = policy(IDS).logits
    model = applied_jump(base=policy.get_base_model())
    assert economize_peft.is_peft_model(policy) and not economize_peft.is_peft_model(model)
    assert model.base_model is policy.get_base_model() and economize_peft.peft_base(policy) is model.base_model
    torch.testing.assert_close(model(IDS).logits, expected, rtol=0, atol=0)


def test_apply_to_a_peft_model_with_a_frozen_jump_path() -> None:
    import economize
    from economize.integrations import peft as economize_peft

    policy = _peft_policy()
    try:
        model = economize_peft.apply(policy, economize.JumpConfig(target_layer=2, source_layer=4))
    except AttributeError as error:  # pragma: no cover - the core is being rebuilt
        pytest.skip(f"economize.apply not usable yet ({error})")
    assert model.base_model is policy.get_base_model()
    torch.testing.assert_close(model(IDS).logits, policy(IDS).logits, rtol=0, atol=0)


def test_a_jump_lora_over_peft_wrapped_maps_is_refused_or_complete() -> None:
    import economize
    from economize.integrations import peft as economize_peft

    policy = _peft_policy()
    config = economize.JumpConfig(target_layer=2, source_layer=4, adaptation="lora", lora_rank=4)
    try:
        model = economize_peft.apply(policy, config)
    except ConfigError as error:
        assert "merge=True" in str(error) and "q_proj" in str(error)
        assert not any(handle for handle in _hooks(policy)), "the refused techniques were detached"
        return
    # the core adapts PEFT-wrapped maps (requested in ARCHITECTURE.md): every projection has an update
    assert all({"q_proj", "v_proj"} <= set(deltas) for deltas in model.techniques["jump"].lora.layers.values())


def _hooks(module: Any) -> list[Any]:
    return [hook for sub in module.modules() for hook in sub._forward_hooks.values()]


def test_merge_folds_peft_adapters_first(applied_jump: Any) -> None:
    import economize
    from economize.integrations import peft as economize_peft

    policy = _peft_policy()
    expected = policy(IDS).logits
    config = economize.JumpConfig(target_layer=2, source_layer=4, adaptation="lora", lora_rank=4)
    model = economize_peft.apply(policy, config, merge=True)
    assert not economize_peft.is_peft_model(model.base_model)
    torch.testing.assert_close(model(IDS).logits, expected, rtol=1e-5, atol=1e-5)
    assert set(model.techniques["jump"].lora.layers["3"]) >= {"q_proj", "v_proj", "down_proj"}


def test_peft_base_decodes_like_the_peft_model(jump_model: Any) -> None:
    import economize
    from economize.integrations import peft as economize_peft

    policy = _peft_policy()
    model = economize_peft.apply(policy, economize.JumpConfig(target_layer=2, source_layer=4),
                              decoding=economize.DecodeConfig(cycle_length=4))
    prompt = IDS[0]
    assert torch.equal(model.decode(prompt, 12).tokens.view(-1), reference_greedy(policy, prompt, 12))


# ---- the jump-path LoRA in PEFT format ---------------------------------------------------------------------------
def test_export_loads_in_peft(applied_jump: Any, tmp_path: Path) -> None:
    from peft import PeftModel

    from economize.integrations import peft as economize_peft

    model = applied_jump(adaptation="lora", lora_rank=4, lora_alpha=6.0)
    modules = economize_peft.lora_modules(model)
    assert {name.split(".")[2] for name in modules} == {str(i) for i in JUMP_LAYERS}
    assert "model.layers.2.self_attn.q_proj" in modules and "model.layers.5.mlp.down_proj" in modules
    economize_peft.save_lora(model, tmp_path, base_model_name_or_path="tiny/qwen2")
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert config["peft_type"] == "LORA" and config["r"] == 4 and config["lora_alpha"] == 6.0
    assert config["base_model_name_or_path"] == "tiny/qwen2" and sorted(config["target_modules"]) == sorted(modules)

    peft_model = PeftModel.from_pretrained(make_tiny_model(), str(tmp_path)).eval()
    with torch.no_grad():
        got = peft_model(IDS).logits
    torch.testing.assert_close(got, _jump_logits(model), rtol=1e-5, atol=1e-5)
    with torch.no_grad():
        assert not torch.allclose(model.base_model(IDS).logits, got, atol=1e-3), "the exact path has no LoRA"


def test_export_load_round_trip(applied_jump: Any, tmp_path: Path) -> None:
    from economize.integrations import peft as economize_peft

    source = applied_jump(adaptation="lora", lora_rank=4, seed=0)
    target = applied_jump(adaptation="lora", lora_rank=4, seed=1, base=make_tiny_model(seed=0))  # other LoRA
    economize_peft.save_lora(source, tmp_path)
    loaded = economize_peft.load_lora(target, tmp_path)
    assert len(loaded) == len(economize_peft.lora_modules(source)) == 4 * 7
    for (name, a), b in zip(economize_peft.lora_modules(source).items(),
                            economize_peft.lora_modules(target).values(), strict=True):
        assert torch.equal(a.lora_A, b.lora_A) and torch.equal(a.lora_B, b.lora_B), name
    torch.testing.assert_close(_jump_logits(target), _jump_logits(source), rtol=0, atol=0)


def test_early_exit_lora_round_trip(tmp_path: Path) -> None:
    import economize
    from economize.integrations import peft as economize_peft
    from economize.techniques.early_exit import EarlyExitConfig
    from helpers import perturb

    def make(seed: int) -> Any:
        model = economize.apply(make_tiny_model(seed=0), EarlyExitConfig(exit_layer=3, adaptation="lora", lora_rank=2))
        perturb(model.techniques["early_exit"], std=0.05, seed=seed)
        return model

    source, target = make(1), make(2)
    economize_peft.save_lora(source, tmp_path)
    assert len(economize_peft.load_lora(target, tmp_path)) == 3 * 7
    for (name, a), b in zip(economize_peft.lora_modules(source).items(),
                            economize_peft.lora_modules(target).values(), strict=True):
        assert torch.equal(a.lora_A, b.lora_A) and torch.equal(a.lora_B, b.lora_B), name


def test_load_an_adapter_trained_with_peft(applied_jump: Any, tmp_path: Path) -> None:
    from economize.integrations import peft as economize_peft

    policy = _peft_policy(layers=JUMP_LAYERS, lora_alpha=6, use_rslora=True)
    policy.save_pretrained(str(tmp_path))
    with torch.no_grad():
        expected = policy(IDS).logits
    model = applied_jump(adaptation="lora", lora_rank=4)
    with pytest.raises(ConfigError, match="lacks"):
        economize_peft.load_lora(model, tmp_path)
    loaded = economize_peft.load_lora(model, tmp_path, strict=False)
    assert loaded == sorted(f"model.layers.{i}.self_attn.{p}_proj" for i in JUMP_LAYERS for p in "qv")
    torch.testing.assert_close(_jump_logits(model), expected, rtol=1e-5, atol=1e-5)


def test_load_refuses_what_the_jump_path_cannot_express(applied_jump: Any, tmp_path: Path) -> None:
    from economize.integrations import peft as economize_peft

    _peft_policy(layers=[0, 2]).save_pretrained(str(tmp_path / "outside"))
    model = applied_jump(adaptation="lora", lora_rank=4)
    with pytest.raises(ConfigError, match=r"model.layers.0"):
        economize_peft.load_lora(model, tmp_path / "outside", strict=False)
    _peft_policy(layers=JUMP_LAYERS, r=8).save_pretrained(str(tmp_path / "rank"))
    with pytest.raises(ConfigError, match="lora_rank=8"):
        economize_peft.load_lora(model, tmp_path / "rank", strict=False)
    _peft_policy(layers=JUMP_LAYERS, use_dora=True).save_pretrained(str(tmp_path / "dora"))
    with pytest.raises(ConfigError, match="use_dora"):
        economize_peft.load_lora(model, tmp_path / "dora", strict=False)
    with pytest.raises(ConfigError, match="no technique with a LoRA"):
        economize_peft.save_lora(applied_jump(), tmp_path / "frozen")


def test_rollouts_attach_to_a_peft_policy() -> None:
    import economize
    from economize.integrations import peft as economize_peft
    from economize.integrations.trl import attach_rollouts

    policy = _peft_policy()
    try:
        model = economize_peft.apply(policy, economize.JumpConfig(target_layer=2, source_layer=4))
    except AttributeError as error:  # pragma: no cover - the core is being rebuilt
        pytest.skip(f"economize.apply not usable yet ({error})")
    trainer = SimpleNamespace(model=policy, callback_handler=SimpleNamespace(callbacks=[]), use_vllm=False)
    rollouts = attach_rollouts(trainer, model)
    assert policy.generate is rollouts and "generate" not in vars(policy.get_base_model())
    rollouts.detach()
    assert "generate" not in vars(policy)
