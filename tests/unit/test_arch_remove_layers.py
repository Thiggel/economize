"""Layer removal keeps every layer-indexed config field consistent, so a shallower model saves and reloads exactly.

:meth:`ArchAdapter.remove_layers` deletes decoder layers in place. Besides ``num_hidden_layers`` and the per-layer
lists, some configs address layers by index (DeepSeek-V3's ``first_k_dense_replace``, the Qwen MoE
``mlp_only_layers`` / ``decoder_sparse_step``, Qwen's ``max_window_layers``) or pack one slice per layer into a
weight (Gemma-4's per-layer embeddings). A field left with the old numbering makes ``from_pretrained`` build a
different module in some position than the one that was saved.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import transformers
from torch import nn

from economize.arch import adapter_for
from economize.errors import UnsupportedModelError
from helpers import FAMILY_MODELS, VOCAB, family_skip_reason, make_family_model

IDS = torch.randint(3, VOCAB, (2, 9), generator=torch.Generator().manual_seed(0))


def _build(family: str, **overrides: object) -> nn.Module:
    """``FAMILY_MODELS[family]`` (a causal LM, not a multimodal wrapper) with config overrides."""
    class_name, text, attn = FAMILY_MODELS[family]
    cls = getattr(transformers, class_name)
    torch.manual_seed(0)
    model = cls._from_config(cls.config_class(**dict(text, **overrides)), attn_implementation=attn)
    return model.float().eval().requires_grad_(False)


@torch.no_grad()
def _check_reload(model: nn.Module, tmp_path: Path) -> None:
    """Save ``model``, reload it with its class and assert identical weights and logits."""
    model.save_pretrained(tmp_path)
    attn = model.config._attn_implementation
    reloaded, info = type(model).from_pretrained(tmp_path, output_loading_info=True, attn_implementation=attn)
    reloaded = reloaded.float().eval()
    problems = {k: v for k, v in info.items() if v and k != "error_msgs"} | (
        {"error_msgs": info["error_msgs"]} if info.get("error_msgs") else {}
    )
    assert not problems, f"reload after layer removal: {problems}"
    torch.testing.assert_close(reloaded(input_ids=IDS).logits, model(input_ids=IDS).logits, atol=1e-5, rtol=0)


def test_deepseek_v3_first_dense_layer_removed(tmp_path: Path) -> None:
    """Removing DeepSeek-V3's only dense layer leaves every remaining layer MoE: ``first_k_dense_replace`` -> 0."""
    model = make_family_model("deepseek_v3")
    adapter_for(model).remove_layers([1])
    assert model.config.first_k_dense_replace == 0
    _check_reload(model, tmp_path)


def test_deepseek_v3_moe_layer_removed(tmp_path: Path) -> None:
    model = _build("deepseek_v3", first_k_dense_replace=2)
    adapter_for(model).remove_layers([2, 3])
    assert model.config.first_k_dense_replace == 1
    _check_reload(model, tmp_path)


@pytest.mark.parametrize("family", ["qwen2_moe", "qwen3_moe", "qwen3_next"])
def test_qwen_moe_sparse_layout_is_renumbered(family: str, tmp_path: Path) -> None:
    """``mlp_only_layers`` follows its layers, and a ``decoder_sparse_step`` pattern becomes an explicit list."""
    model = _build(family, decoder_sparse_step=2, mlp_only_layers=[3])  # sparse: layer index 1 only
    sparse_before = [type(layer.mlp).__name__ for layer in adapter_for(model).layers]
    adapter_for(model).remove_layers([1])
    assert [type(layer.mlp).__name__ for layer in adapter_for(model).layers] == sparse_before[1:]
    assert model.config.decoder_sparse_step == 1
    _check_reload(model, tmp_path)


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "qwen2_moe"])
def test_max_window_layers_counts_kept_layers(family: str) -> None:
    model = _build(family, max_window_layers=2)
    adapter_for(model).remove_layers([1, 4])
    assert model.config.max_window_layers == 1


def test_gemma4_per_layer_embeddings_follow_their_layers(tmp_path: Path) -> None:
    """Gemma-4 packs one per-layer-embedding slice per layer; the slices of removed layers go with them."""
    model = _build("gemma4_text", num_kv_shared_layers=0)
    adapter = adapter_for(model)
    assert adapter.uses_per_layer_inputs
    adapter.remove_layers([2])
    assert model.model.embed_tokens_per_layer.weight.shape[1] == 5 * model.config.hidden_size_per_layer_input
    _check_reload(model, tmp_path)


@pytest.mark.parametrize("family", sorted(FAMILY_MODELS))
def test_removed_layers_save_and_reload(family: str, tmp_path: Path) -> None:
    """Every family: remove a layer, ``save_pretrained``, ``from_pretrained``; the weights and logits survive.

    Families that share K/V between layers refuse the removal instead."""
    reason = family_skip_reason(family)
    if reason is not None:
        pytest.skip(reason)
    model = make_family_model(family)
    adapter = adapter_for(model)
    if any(adapter.kv_source(i) != i for i in range(1, adapter.num_layers + 1)):
        with pytest.raises(UnsupportedModelError, match="shares K/V between layers"):
            adapter.remove_layers([1])
        return
    adapter.remove_layers([1])
    _check_reload(model, tmp_path)
