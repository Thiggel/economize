"""``slicegpt``: rotate the residual stream onto its principal components and delete the trailing ones.

The reference is the identity the paper's method rests on (Ashkboos et al., 2024, Sections 3.2-3.3): the sliced
model computes the dense model whose residual stream is projected onto the kept principal subspace ``Q_k Q_k^T`` at
every residual point. The tests rebuild that projected dense model from the recorded bases and compare logits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import economize
from economize.backends import UnsupportedBackendError
from economize.errors import ConfigError, IncompatibleTechniquesError, UnsupportedModelError
from economize.techniques.jump import JumpConfig
from economize.techniques.slicegpt import (
    SliceGPTConfig,
    SliceGPTTechnique,
    install_shortcuts,
    principal_basis,
    set_shortcuts,
    shortcut_widths,
)
from economize.techniques.slicegpt.config import ROTATIONS
from helpers import VOCAB, make_family_model, make_tiny_model, reference_greedy

TOKENS = torch.randint(0, VOCAB, (2, 16), generator=torch.Generator().manual_seed(3))


def _attach(model: nn.Module, config: SliceGPTConfig, adapter_of, attach_context, calib) -> SliceGPTTechnique:
    adapter = adapter_of(model)
    technique = SliceGPTTechnique(config, adapter)
    technique.attach(model, adapter, attach_context(calib))
    return technique


def _projected_reference(model: nn.Module, technique: SliceGPTTechnique) -> list[Any]:
    """Turn the dense ``model`` into the dense model with every residual point projected onto the recorded bases."""
    layers = list(model.model.layers)
    rotations = technique.rotations.double()
    per_block = technique.config.rotation == "per_block"

    def projector(point: int) -> torch.Tensor:
        if per_block and point == 2 * len(layers) and not technique.config.slices_head:
            return torch.eye(rotations.shape[1], dtype=torch.float64)
        basis = rotations[point if per_block else 0]
        return basis @ basis.T

    def project(matrix: torch.Tensor):
        def hook(_module: nn.Module, _args: Any, output: torch.Tensor) -> torch.Tensor:
            return output @ matrix.to(output)

        return hook

    handles = [model.model.embed_tokens.register_forward_hook(project(projector(0)))]
    for index, layer in enumerate(layers, start=1):
        mid, out = projector(2 * index - 1), projector(2 * index)
        install_shortcuts(layer, technique_layout(model))
        set_shortcuts(layer, mid, out)
        handles.append(layer.self_attn.o_proj.register_forward_hook(project(mid)))
        handles.append(layer.mlp.down_proj.register_forward_hook(project(out)))
    return handles


def technique_layout(model: nn.Module) -> Any:
    from economize.arch import adapter_for
    from economize.techniques._structured import layout_for

    return layout_for(adapter_for(model), "slicegpt")


def _confine(model: nn.Module, rank: int, seed: int = 0) -> None:
    """Make every write to the residual stream land in one random ``rank``-dimensional subspace."""
    hidden = model.config.hidden_size
    generator = torch.Generator().manual_seed(seed)
    basis, _ = torch.linalg.qr(torch.randn(hidden, rank, generator=generator, dtype=torch.float64))
    projector = (basis @ basis.T).float()
    model.config.tie_word_embeddings = False
    model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone(), requires_grad=False)
    embedding = model.model.embed_tokens
    embedding.weight = nn.Parameter(embedding.weight.detach() @ projector, requires_grad=False)
    for layer in model.model.layers:
        for linear in (layer.self_attn.o_proj, layer.mlp.down_proj):
            linear.weight = nn.Parameter(projector @ linear.weight.detach(), requires_grad=False)


# ---- the rotation itself -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_zero_sparsity_is_the_dense_model(rotation: str, adapter_of, attach_context, calib) -> None:
    dense, model = make_tiny_model(), make_tiny_model()
    _attach(model, SliceGPTConfig(sparsity=0.0, rotation=rotation), adapter_of, attach_context, calib)
    torch.testing.assert_close(model(TOKENS).logits, dense(TOKENS).logits, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(("rotation", "slice_head"), [("per_block", False), ("per_block", True), ("global", None)])
def test_the_sliced_model_is_the_projected_dense_model(rotation: str, slice_head: bool | None, adapter_of,
                                                       attach_context, calib) -> None:
    model = make_tiny_model()
    config = SliceGPTConfig(sparsity=0.25, rotation=rotation, slice_head=slice_head)
    technique = _attach(model, config, adapter_of, attach_context, calib)
    reference = make_tiny_model()
    handles = _projected_reference(reference, technique)
    try:
        torch.testing.assert_close(model(TOKENS).logits, reference(TOKENS).logits, atol=2e-4, rtol=2e-4)
    finally:
        for handle in handles:
            handle.remove()


def test_the_first_basis_is_the_principal_basis_of_the_embeddings(tiny2, adapter_of, attach_context, calib) -> None:
    embeddings = tiny2.model.embed_tokens(calib).double()
    flat = embeddings.reshape(-1, embeddings.shape[-1])
    expected = principal_basis(flat.T @ flat)[:, :48].float()
    technique = _attach(tiny2, SliceGPTConfig(sparsity=0.25), adapter_of, attach_context, calib)
    cosines = (technique.rotations[0] * expected).sum(0).abs()  # each column agrees up to its sign
    torch.testing.assert_close(cosines, torch.ones(48), atol=1e-4, rtol=0)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_a_stream_of_lower_rank_is_sliced_without_loss(rotation: str, adapter_of, attach_context, calib) -> None:
    dense, model = make_tiny_model(), make_tiny_model()
    _confine(dense, 40)
    _confine(model, 40)
    technique = _attach(model, SliceGPTConfig(sparsity=0.25, rotation=rotation), adapter_of, attach_context, calib)
    torch.testing.assert_close(model(TOKENS).logits, dense(TOKENS).logits, atol=2e-4, rtol=2e-4)

    random = make_tiny_model()
    _confine(random, 40)
    adapter = adapter_of(random)
    replay = SliceGPTTechnique(technique.config, adapter)
    generator = torch.Generator().manual_seed(1)
    for point in range(replay.rotations.shape[0]):
        replay.rotations[point] = torch.linalg.qr(torch.randn(64, 48, generator=generator))[0]
    replay.head_rotation.copy_(technique.head_rotation)
    replay.fitted.fill_(True)
    replay.attach(random, adapter, attach_context(None))
    assert (random(TOKENS).logits - dense(TOKENS).logits).abs().max() > 1e-1  # a random basis loses the signal


def test_per_block_widths_and_shortcuts(adapter_of, attach_context, calib) -> None:
    model = make_tiny_model()
    technique = _attach(model, SliceGPTConfig(sparsity=0.25), adapter_of, attach_context, calib)
    layers = model.model.layers
    assert model.model.embed_tokens.weight.shape == (VOCAB, 48)
    assert layers[0].self_attn.q_proj.in_features == 48 and layers[0].self_attn.o_proj.out_features == 48
    assert layers[0].mlp.down_proj.out_features == 48 and layers[-1].mlp.down_proj.out_features == 64
    assert shortcut_widths(layers[0]) == (torch.Size([48, 48]), torch.Size([48, 48]))
    assert shortcut_widths(layers[-1]) == (torch.Size([48, 48]), torch.Size([48, 64]))
    assert model.lm_head.in_features == 64 and model.lm_head.weight is not model.model.embed_tokens.weight
    assert not model.config.tie_word_embeddings
    assert technique.summary["width"] == 48 and technique.summary["removed_parameters"] > 0


def test_global_rotation_rewrites_the_config(adapter_of, attach_context, calib) -> None:
    model = make_tiny_model()
    _attach(model, SliceGPTConfig(sparsity=0.25, rotation="global"), adapter_of, attach_context, calib)
    assert model.config.hidden_size == 48 and model.config.head_dim == 16
    assert model.lm_head.in_features == 48 and shortcut_widths(model.model.layers[0]) == (None, None)


def test_configuration_errors(tiny2, adapter_of, attach_context) -> None:
    with pytest.raises(ConfigError, match="calibration"):
        _attach(tiny2, SliceGPTConfig(), adapter_of, attach_context, None)
    with pytest.raises(ConfigError):
        SliceGPTConfig(sparsity=1.0).validate(adapter_of(tiny2))
    with pytest.raises(ConfigError):
        SliceGPTConfig(rotation="layerwise")
    assert SliceGPTConfig(sparsity=0.3).width(64) == 40  # 44.8 rounded down to a multiple of 8


def test_gemma_sandwich_norms_are_refused(adapter_of) -> None:
    model = make_family_model("gemma3_text")
    with pytest.raises(UnsupportedModelError, match="slicegpt"):
        SliceGPTTechnique(SliceGPTConfig(), adapter_of(model))


# ---- composition, checkpoints ------------------------------------------------------------------------------------


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_a_saved_slicing_replays_without_calibration(rotation: str, tmp_path: Path, calib) -> None:
    model = economize.apply(make_tiny_model(), SliceGPTConfig(rotation=rotation), calibration=[calib])
    expected = economize.unwrap.base_model_of(model)(TOKENS).logits
    economize.save(model, tmp_path / "sliced")
    again = economize.load(make_tiny_model(), tmp_path / "sliced")
    assert torch.equal(economize.unwrap.base_model_of(again)(TOKENS).logits, expected)


def test_per_block_refuses_drafters_and_global_takes_them(calib) -> None:
    jump = JumpConfig(target_layer=2, source_layer=4)
    with pytest.raises(IncompatibleTechniquesError, match="per_block"):
        economize.apply(make_tiny_model(), SliceGPTConfig(), jump, calibration=[calib])
    model = economize.apply(make_tiny_model(), SliceGPTConfig(rotation="global"), jump, calibration=[calib])
    prompt = TOKENS[:1, :8]
    out = model.generate(prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=6, do_sample=False)
    assert torch.equal(out[0, 8:], reference_greedy(economize.unwrap.base_model_of(model), prompt, 6))


# ---- serving -----------------------------------------------------------------------------------------------------


def _saved(name: str, tmp_path: Path) -> Any:
    from transformers import AutoModelForCausalLM

    folder = tmp_path / name
    (make_family_model(name) if name != "qwen2_tiny" else make_tiny_model()).save_pretrained(folder)
    return AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32).eval()


def test_global_slicing_serves_as_a_plain_checkpoint_on_vllm(tmp_path: Path, calib) -> None:
    from transformers import AutoModelForCausalLM

    from economize.backends.vllm.serving import to_vllm

    model = economize.apply(_saved("llama", tmp_path), SliceGPTConfig(rotation="global"), calibration=[calib])
    plan = to_vllm(model, tmp_path / "out")
    assert plan.techniques == {"slicegpt": "native"}
    config = json.loads((Path(plan.model) / "config.json").read_text())
    assert config["hidden_size"] == 48 and config["head_dim"] == 16
    served = AutoModelForCausalLM.from_pretrained(plan.model, dtype=torch.float32).eval()
    torch.testing.assert_close(served(TOKENS).logits, economize.unwrap.base_model_of(model)(TOKENS).logits,
                               atol=1e-5, rtol=1e-5)


def test_vllm_refuses_per_block_and_qwen2(tmp_path: Path, calib) -> None:
    from economize.backends.vllm.serving import check_servable

    per_block = economize.apply(_saved("llama", tmp_path), SliceGPTConfig(), calibration=[calib])
    with pytest.raises(UnsupportedBackendError, match="per_block"):
        check_servable(per_block)
    qwen2 = economize.apply(_saved("qwen2_tiny", tmp_path), SliceGPTConfig(rotation="global"), calibration=[calib])
    with pytest.raises(UnsupportedBackendError, match="head size"):
        check_servable(qwen2)


def test_sglang_serves_global_slicing_of_qwen2(tmp_path: Path, calib) -> None:
    from economize.backends.sglang.serving import check_servable, to_sglang

    qwen2 = economize.apply(_saved("qwen2_tiny", tmp_path), SliceGPTConfig(rotation="global"), calibration=[calib])
    plan = to_sglang(qwen2, tmp_path / "out")
    assert plan.techniques == {"slicegpt": "native"}
    per_block = economize.apply(_saved("qwen2_tiny", tmp_path), SliceGPTConfig(), calibration=[calib])
    with pytest.raises(UnsupportedBackendError, match="per_block"):
        check_servable(per_block)


def test_backend_support_is_real_where_served() -> None:
    from economize.core.compatibility import SupportStatus

    rows = economize.backend_support(SliceGPTConfig(rotation="global"))
    served = ("transformers", "vllm", "sglang")
    assert {name: rows[name].status for name in served} == {
        "transformers": SupportStatus.REAL, "vllm": SupportStatus.REAL, "sglang": SupportStatus.REAL,
    }
    assert economize.backend_support(SliceGPTConfig())["vllm"].status is SupportStatus.UNSUPPORTED
