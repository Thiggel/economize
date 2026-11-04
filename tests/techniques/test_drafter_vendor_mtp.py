"""Vendor MTP: chained modules (DeepSeek-V3's sequential MTP), the family registry, vendor checkpoint names and the
dequantization of FP8 module weights."""

from __future__ import annotations

import pytest
import torch

import economize
from economize import DecodeConfig
from economize.errors import CheckpointFormatError, ConfigError
from economize.runtime import ExactPass
from economize.techniques.vendor_mtp import (
    DEFAULT_FAMILY,
    MTP_FAMILIES,
    MTPFamily,
    MTPLayout,
    VendorMTPConfig,
    dequantize_fp8,
    family_for,
)
from economize.techniques.vendor_mtp.formats import state_from_vendor, state_to_vendor
from economize.train import Batch, run_teacher
from helpers import VOCAB, make_tiny_model, perturb, reference_greedy

LAYERS = 6


def _model(modules: int = 2, steps: int = 2):
    model = economize.apply(make_tiny_model(), VendorMTPConfig(num_modules=modules, train_steps=steps),
                         decoding=DecodeConfig(cycle_length=4))
    technique = model.techniques["vendor_mtp"]
    perturb(technique, std=0.05, seed=1)
    return model, technique


# ---- chaining ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("modules", [2, 3])
def test_draft_steps_equal_the_chained_parallel_depths(modules: int) -> None:
    """After exact passes over positions ``0..12``, draft step ``k <= num_modules`` equals depth ``k`` of
    ``train_logits`` at position 12 (module ``k - 1`` on module ``k - 2``'s chained output)."""
    model, technique = _model(modules=modules, steps=modules)
    ids = torch.randint(0, VOCAB, (2, 16), generator=torch.Generator().manual_seed(5))
    batch = Batch(ids)
    with torch.no_grad():
        teacher = run_teacher(model, batch, technique.train_request(model))
        expected = technique.train_logits(model, batch, teacher)
        final, embedding = teacher.hidden[LAYERS], model.adapter.embed(ids)
        runner = model.runner
        drafter = technique.drafter(runner)
        drafter.begin(runner, runner.new_cache(64, 2), 2)
        scratch = runner.new_cache(64, 2)
        for start, end in ((0, 10), (10, 13)):
            context = runner.context(scratch, torch.arange(start, end), trusted=True)
            exact = ExactPass(final[:, start:end], {0: embedding[:, start:end]}, context)
            drafter.observe(exact, torch.full((2,), end - start - 1))
        for k in range(1, modules + 1):
            logits = drafter.step(ids[:, 12 + k], torch.full((2,), 12 + k), None)
            torch.testing.assert_close(logits, expected[k + 1][:, 12], rtol=1e-4, atol=1e-4)


def test_depths_differ_from_reusing_the_first_module() -> None:
    """The second module reads the first one's output: swapping in module 0 for depth 2 changes the drafts."""
    model, technique = _model(modules=2, steps=2)
    batch = Batch(torch.randint(0, VOCAB, (1, 12), generator=torch.Generator().manual_seed(2)))
    with torch.no_grad():
        teacher = run_teacher(model, batch, technique.train_request(model))
        chained = technique.train_logits(model, batch, teacher)[3]
        technique.layers[1].load_state_dict(technique.layers[0].state_dict())
        shared = technique.train_logits(model, batch, teacher)[3]
    assert not torch.allclose(chained, shared)


@pytest.mark.parametrize("modules", [1, 2, 3])
def test_chained_modules_decode_losslessly(modules: int) -> None:
    model, _ = _model(modules=modules)
    prompt = torch.randint(1, VOCAB, (9,), generator=torch.Generator().manual_seed(4))
    result = model.decode(prompt, 20)
    assert torch.equal(result.tokens, reference_greedy(model.base_model, prompt, 20))
    events = result.cost.events
    assert all(e.module == "mtp" for e in events if e.kind in ("draft", "draft_observe", "draft_wasted"))
    assert all(e.layer_tokens == e.tokens for e in events if e.kind in ("draft", "draft_wasted"))
    if modules > 1:  # the later modules' entries of drafted tokens are written during the draft steps
        observe = [e for e in events if e.kind == "draft_observe"]
        assert any(e.tokens == 1 and e.layer_tokens == 1 for e in observe)


# ---- families -------------------------------------------------------------------------------------------------
def test_family_registry_and_lookup() -> None:
    assert {"deepseek_v3", "glm4_moe", "mimo", "qwen3_next", "qwen3_5"} <= set(MTP_FAMILIES.names())
    assert family_for("kimi_k2") == "deepseek_v3" and family_for("qwen3_5_moe_text") == "qwen3_5"
    with pytest.raises(ConfigError):
        family_for("qwen2")
    assert VendorMTPConfig().resolve_family("qwen2") == DEFAULT_FAMILY
    assert VendorMTPConfig(family="mimo").resolve_family("qwen2") == "mimo"


def test_a_registered_family_is_used_without_editing_the_package() -> None:
    family = MTPFamily(MTPLayout("hidden_emb", False, "normed"), ("my_mtp_model",), module_prefix="mtp.{index}.",
                       numbered_after_model=False)
    MTP_FAMILIES.register("my_family", family)
    try:
        assert family_for("my_mtp_model") == "my_family"
        assert VendorMTPConfig().layout("my_mtp_model") == family.layout
        state = {"layers.0.enorm.weight": torch.ones(4), "layers.0.block.mlp.up_proj.weight": torch.ones(2, 4)}
        vendor = state_to_vendor(state, "my_family", 6)
        assert set(vendor) == {"mtp.0.enorm.weight", "mtp.0.mlp.up_proj.weight"}
        assert state_from_vendor(vendor, "my_family", 6).keys() == state.keys()
    finally:
        MTP_FAMILIES.unregister("my_family")


def _module_state(modules: int, shared: bool) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    state: dict[str, torch.Tensor] = {}
    norms = {name: torch.randn(8, generator=generator) for name in ("enorm", "hnorm", "norm")}
    proj = torch.randn(8, 16, generator=generator)
    for i in range(modules):
        for name, value in norms.items():
            state[f"layers.{i}.{name}.weight"] = value if shared else value + i
        state[f"layers.{i}.eh_proj.weight"] = proj if shared else proj + i
        state[f"layers.{i}.block.self_attn.q_proj.weight"] = torch.randn(8, 8, generator=generator)
        state[f"layers.{i}.block.input_layernorm.weight"] = torch.randn(8, generator=generator)
    return state


@pytest.mark.parametrize("family", ["deepseek_v3", "glm4_moe", "mimo", "qwen3_next", "qwen3_5"])
def test_vendor_names_round_trip(family: str) -> None:
    shared = MTP_FAMILIES.get(family).shared_prefix is not None
    state = _module_state(2, shared)
    vendor = state_to_vendor(state, family, LAYERS)
    distractors = {"model.layers.3.self_attn.q_proj.weight": torch.zeros(8, 8), "lm_head.weight": torch.zeros(4, 8)}
    back = state_from_vendor({**vendor, **distractors}, family, LAYERS)
    assert back.keys() == state.keys()
    assert all(torch.equal(back[key], state[key]) for key in state)


def test_shared_qwen_parameters_must_agree() -> None:
    with pytest.raises(CheckpointFormatError):
        state_to_vendor(_module_state(2, shared=False), "qwen3_next", LAYERS)


def test_deepseek_layers_are_numbered_after_the_model() -> None:
    vendor = state_to_vendor(_module_state(1, False), "deepseek_v3", LAYERS)
    assert f"model.layers.{LAYERS}.shared_head.norm.weight" in vendor
    assert f"model.layers.{LAYERS}.eh_proj.weight" in vendor
    assert f"model.layers.{LAYERS}.self_attn.q_proj.weight" in vendor


# ---- FP8 ------------------------------------------------------------------------------------------------------
def test_fp8_block_dequantization() -> None:
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(5, 7, generator=generator)
    scale = torch.rand(3, 4, generator=generator) + 0.5  # blocks of 2 x 2, the last ones partial
    fp8 = weight.to(torch.float8_e4m3fn)
    out = dequantize_fp8({"a.weight": fp8, "a.weight_scale_inv": scale, "b.weight": weight}, block=(2, 2))
    expected = torch.empty(5, 7)
    for r in range(5):
        for c in range(7):
            expected[r, c] = fp8[r, c].float() * scale[r // 2, c // 2]
    assert set(out) == {"a.weight", "b.weight"} and out["a.weight"].dtype == torch.bfloat16
    torch.testing.assert_close(out["a.weight"], expected.to(torch.bfloat16))
    assert out["b.weight"] is weight


@pytest.mark.parametrize("tensors", [
    {"a.weight": torch.zeros(4, 4), "a.weight_scale_inv": torch.ones(3, 3)},  # wrong scale shape
    {"a.weight_scale_inv": torch.ones(1, 1)},  # scale without weight
    {"a.weight": torch.zeros(4, 4), "a.weight_scale": torch.ones(())},  # per-tensor scheme
])
def test_fp8_refuses_unsupported_tensors(tensors: dict[str, torch.Tensor]) -> None:
    with pytest.raises(CheckpointFormatError):
        dequantize_fp8(tensors, block=(2, 2))


def test_fp8_module_weights_load_through_the_family() -> None:
    state = {"layers.0.eh_proj.weight": torch.ones(4, 8)}
    vendor = state_to_vendor(state, "deepseek_v3", LAYERS)
    key = f"model.layers.{LAYERS}.eh_proj.weight"
    vendor[key] = vendor[key].to(torch.float8_e4m3fn)
    vendor[f"model.layers.{LAYERS}.eh_proj.weight_scale_inv"] = torch.full((1, 1), 2.0)
    back = state_from_vendor(vendor, "deepseek_v3", LAYERS, fp8_block=(128, 128))
    torch.testing.assert_close(back["layers.0.eh_proj.weight"], torch.full((4, 8), 2.0, dtype=torch.bfloat16))

