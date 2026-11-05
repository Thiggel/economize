"""Structured width pruning of Gemma 4 through the ``"gemma4"`` decoder layout.

The tiny dense model mirrors gemma-4-31B-it in miniature: sliding layers with narrow heads and four key/value heads,
full-attention layers with wider heads, two key/value heads and no value map (``attention_k_eq_v``), sandwich norms,
per-layer embeddings and no KV sharing. Its query heads match across layer types at an attention sparsity of 0.5
(as the 31B's do at multiples of 0.25), so that pruning leaves a plain checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from economize.errors import UnsupportedModelError
from economize.techniques._structured import Gemma4Layout, layout_for
from economize.techniques.slicegpt import SliceGPTConfig, SliceGPTTechnique
from economize.techniques.width_prune import WidthPruneConfig, WidthPruneTechnique, layer_shape
from helpers import _VISION, MAX_POSITIONS, VOCAB, WINDOW, make_family_model

TOKENS = torch.randint(0, VOCAB, (2, 16), generator=torch.Generator().manual_seed(3))
LAYERS = 6
DENSE = dict(
    vocab_size=VOCAB, hidden_size=64, intermediate_size=96, num_hidden_layers=LAYERS, num_attention_heads=4,
    num_key_value_heads=4, head_dim=16, max_position_embeddings=MAX_POSITIONS, initializer_range=0.1,
    tie_word_embeddings=False, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2, sliding_window=WINDOW,
    global_head_dim=32, num_global_key_value_heads=2, attention_k_eq_v=True, num_kv_shared_layers=0,
    hidden_size_per_layer_input=8, vocab_size_per_layer_input=VOCAB, final_logit_softcapping=3.0,
)
FULL = (2, 5)  # 0-indexed full-attention layers
#: Every layer pruned (the criteria's defaults protect the first and last layers).
ALL_LAYERS = dict(keep_first=0, keep_last=0)


def make_gemma4(name: str = "dense", seed: int = 0) -> nn.Module:
    """The tiny dense Gemma 4 above, its multimodal wrapper (``"wrapper"``) or the helpers' MoE variant."""
    import transformers

    if name == "moe":
        return make_family_model("gemma4_text_moe", seed)
    if name == "wrapper":
        config: Any = transformers.Gemma4Config(text_config=dict(DENSE), vision_config=dict(_VISION["gemma4"]),
                                                audio_config=None)
        cls: Any = transformers.Gemma4ForConditionalGeneration
    else:
        config, cls = transformers.Gemma4TextConfig(**DENSE), transformers.Gemma4ForCausalLM
    torch.manual_seed(seed)
    model = cls._from_config(config, attn_implementation="sdpa").float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _attach(model: nn.Module, config: WidthPruneConfig, adapter_of, attach_context, calib) -> WidthPruneTechnique:
    adapter = adapter_of(model)
    technique = WidthPruneTechnique(config, adapter)
    technique.attach(model, adapter, attach_context(calib))
    return technique


def _logits(model: nn.Module) -> torch.Tensor:
    return model(input_ids=TOKENS).logits


def _mask_removed(model: nn.Module, technique: WidthPruneTechnique, adapter_of) -> None:
    """Zero (or, with compensation, set to the calibration mean) the inputs of the removed heads and FFN channels of
    the dense ``model``, using each layer's own head width and group size."""
    adapter = adapter_of(model)
    layout = layout_for(adapter, "width_prune")
    compensated = technique.config.compensates
    for i in range(adapter.num_layers):
        layer = adapter.layer(i + 1)
        shape = layer_shape(layout, layer)
        width = shape.heads * shape.head_dim
        heads = technique.groups_kept[i, : shape.kv_heads].repeat_interleave(shape.group_width)
        channels = technique.channels_kept[i, : shape.intermediate]
        means = ((technique.attn_mean[i, :width], technique.mlp_mean[i, : shape.intermediate]) if compensated
                 else (None, None))
        targets = ((layout.linear(layer, layout.attn_output), heads, means[0]),
                   (layout.linear(layer, layout.mlp_output), channels, means[1]))
        for linear, kept, mean in targets:
            def pre(_m: nn.Module, args: Any, kept: torch.Tensor = kept, mean: Any = mean) -> Any:
                fill = torch.zeros_like(args[0]) if mean is None else mean.to(args[0]).expand_as(args[0])
                return (torch.where(kept.to(args[0].device), args[0], fill),)

            linear.register_forward_pre_hook(pre)


def _kv_heads(config: Any) -> list[int]:
    return [config.per_layer_config[i].num_key_value_heads for i in range(config.num_hidden_layers)]


# ---- layout ------------------------------------------------------------------------------------------------------


def test_gemma4_adapters_resolve_the_gemma4_layout(adapter_of) -> None:
    for name in ("dense", "wrapper"):
        layout = layout_for(adapter_of(make_gemma4(name)), "width_prune")
        assert isinstance(layout, Gemma4Layout) and layout.mlp_norm == "pre_feedforward_layernorm"


def test_the_layout_skips_the_value_map_of_layers_whose_values_are_the_keys(adapter_of) -> None:
    adapter = adapter_of(make_gemma4())
    layout = layout_for(adapter, "width_prune")
    for i in range(LAYERS):
        names = list(layout.maps(adapter.layer(i + 1), layout.attn_inputs))
        assert names == (["self_attn.q_proj", "self_attn.k_proj"] if i in FULL else list(layout.attn_inputs))
    shapes = [layer_shape(layout, adapter.layer(i + 1)) for i in range(LAYERS)]
    assert [(s.heads, s.kv_heads, s.head_dim) for s in shapes] == [(4, 4, 16), (4, 4, 16), (4, 2, 32)] * 2


# ---- numerics ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["flap", "llm_pruner", "minitron"])
def test_zero_sparsity_is_the_original_model(method: str, adapter_of, attach_context, calib) -> None:
    pruned = make_gemma4()
    config = WidthPruneConfig(method=method, structure="uniform", sparsity=0.0, **ALL_LAYERS)
    _attach(pruned, config, adapter_of, attach_context, calib)
    assert torch.equal(_logits(pruned), _logits(make_gemma4()))


@pytest.mark.parametrize(("model", "method", "options"), [
    ("dense", "flap", dict(structure="uniform", compensate=False)),
    ("dense", "flap", dict(structure="uniform", compensate=True)),
    ("dense", "flap", dict(structure="adaptive", compensate=True)),
    ("dense", "llm_pruner", dict(structure="uniform")),
    ("dense", "minitron", dict(structure="uniform")),
    ("moe", "flap", dict(structure="uniform", compensate=True)),
    ("moe", "minitron", dict(structure="uniform")),
])
def test_pruning_is_the_original_model_with_the_removed_units_masked(model: str, method: str, options: dict,
                                                                     adapter_of, attach_context, calib) -> None:
    pruned = make_gemma4(model)
    config = WidthPruneConfig(method=method, sparsity=0.5, **ALL_LAYERS, **options)
    technique = _attach(pruned, config, adapter_of, attach_context, calib)
    dense = make_gemma4(model)
    assert sum(technique.summary["kv_heads"]) < sum(_kv_heads(dense.config))
    _mask_removed(dense, technique, adapter_of)
    # The compacted and the masked matrices sum their products in different orders (1e-5 is too tight on AVX2).
    torch.testing.assert_close(_logits(pruned), _logits(dense), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("model", ["dense", "wrapper"])
def test_uniform_pruning_saves_and_reloads_as_a_plain_checkpoint(model: str, tmp_path: Path, adapter_of,
                                                                  attach_context, calib) -> None:
    import transformers

    pruned = make_gemma4(model)
    config = WidthPruneConfig(method="flap", structure="uniform", sparsity=0.5, compensate=False, **ALL_LAYERS)
    technique = _attach(pruned, config, adapter_of, attach_context, calib)
    summary = technique.summary
    assert summary["uniform"] and summary["plain_checkpoint"]
    assert summary["heads"] == [2] * LAYERS and summary["kv_heads"] == [2, 2, 1] * 2
    text = adapter_of(pruned).config
    assert text.num_attention_heads == 2 and text.intermediate_size == 48 and _kv_heads(text) == [2, 2, 1] * 2
    assert [text.per_layer_config[i].head_dim for i in range(LAYERS)] == [16, 16, 32] * 2
    pruned.save_pretrained(tmp_path / "plain")
    auto = transformers.AutoModelForCausalLM if model == "dense" else transformers.AutoModelForImageTextToText
    again = auto.from_pretrained(tmp_path / "plain", dtype=torch.float32).eval()
    assert _kv_heads(adapter_of(again).config) == [2, 2, 1] * 2
    assert torch.equal(_logits(again), _logits(pruned))


def test_attention_compensation_is_a_bias_and_mlp_compensation_needs_economize(adapter_of, attach_context,
                                                                               calib) -> None:
    pruned = make_gemma4()
    config = WidthPruneConfig(method="flap", structure="uniform", sparsity=0.5, **ALL_LAYERS)
    technique = _attach(pruned, config, adapter_of, attach_context, calib)
    assert technique.summary["compensated"] == {"attention": True, "mlp": True}
    assert technique.summary["uniform"] and not technique.summary["plain_checkpoint"]
    attention = adapter_of(pruned).layer(FULL[0] + 1).self_attn
    assert pruned.config.attention_bias and attention.o_proj.bias is not None and attention.v_proj is None


def test_layer_types_with_different_query_heads_leave_the_config_as_it_is(adapter_of, attach_context, calib) -> None:
    pruned = make_gemma4()
    config = WidthPruneConfig(method="minitron", structure="uniform", sparsity=0.25, **ALL_LAYERS)
    technique = _attach(pruned, config, adapter_of, attach_context, calib)
    assert technique.summary["heads"] == [3, 3, 4] * 2  # 3 of 4 groups of one head, 2 of 2 groups of two heads
    assert not technique.summary["uniform"] and not technique.summary["plain_checkpoint"]
    assert pruned.config.num_attention_heads == 4 and pruned.config.intermediate_size == 96
    assert _kv_heads(pruned.config) == [4, 4, 2] * 2
    assert torch.isfinite(_logits(pruned)).all()


def test_chunked_llm_pruner_gradients_give_the_same_scores(adapter_of, calib) -> None:
    from economize.techniques.width_prune import LLMPruner

    scores = []
    for chunk in (0, 2):
        config = WidthPruneConfig(method="llm_pruner", gradient_chunk=chunk, **ALL_LAYERS)
        adapter = adapter_of(make_gemma4())
        scores.append(LLMPruner().score(adapter, layout_for(adapter, "width_prune"), [calib],
                                        config.pruned_layers(LAYERS), config))
    for i in range(1, LAYERS + 1):
        torch.testing.assert_close(scores[1].groups[i], scores[0].groups[i])
        torch.testing.assert_close(scores[1].channels[i], scores[0].channels[i])
    assert all(len(scores[0].groups[i]) == (2 if i - 1 in FULL else 4) for i in range(1, LAYERS + 1))


def test_the_recovery_lora_has_no_value_update_where_values_are_the_keys(adapter_of, attach_context, calib) -> None:
    pruned = make_gemma4()
    config = WidthPruneConfig(method="llm_pruner", sparsity=0.5, lora_rank=4, **ALL_LAYERS)
    technique = _attach(pruned, config, adapter_of, attach_context, calib)
    assert technique.lora is not None
    for i in range(LAYERS):
        names = set(technique.lora.layers[str(i + 1)].keys())
        assert ("v_proj" in names) == (i not in FULL) and {"q_proj", "k_proj", "o_proj", "down_proj"} <= names


@pytest.mark.parametrize("model", ["dense", "wrapper"])
def test_the_serving_engines_refuse_a_gemma4_with_pruned_heads(model: str, adapter_of, attach_context,
                                                               calib) -> None:
    """Neither engine lists Gemma 4 among the model types whose implementation reads an explicit ``head_dim``
    (:attr:`EngineArchitectures.head_dim`), so both refuse; the wrapper is judged by its text config."""
    from economize.backends.sglang.serving.structured import WidthPruneSglang
    from economize.backends.vllm.serving.structured import WidthPruneVllm
    from economize.techniques.width_prune.serving import serving_obstacle

    pruned = make_gemma4(model)
    config = WidthPruneConfig(method="flap", structure="uniform", sparsity=0.5, compensate=False, **ALL_LAYERS)
    technique = _attach(pruned, config, adapter_of, attach_context, calib)
    for engine in (WidthPruneVllm.architectures, WidthPruneSglang.architectures):
        obstacle = serving_obstacle(technique, pruned, engine)
        assert obstacle is not None and "hidden_size // num_attention_heads (64 // 2)" in obstacle


# ---- refusals ----------------------------------------------------------------------------------------------------


def test_residual_pruning_is_refused(adapter_of) -> None:
    config = WidthPruneConfig(method="minitron", sparsity=0.5, hidden_sparsity=0.25, **ALL_LAYERS)
    with pytest.raises(UnsupportedModelError, match=r"residual stream.*post-attention and post-feedforward norms"):
        WidthPruneTechnique(config, adapter_of(make_gemma4()))


@pytest.mark.parametrize("model", ["dense", "wrapper"])
def test_slicegpt_is_refused(model: str, adapter_of) -> None:
    with pytest.raises(UnsupportedModelError, match="slicegpt changes the residual stream"):
        SliceGPTTechnique(SliceGPTConfig(), adapter_of(make_gemma4(model)))


def test_kv_sharing_is_refused(adapter_of) -> None:
    with pytest.raises(UnsupportedModelError, match="KV sharing"):
        WidthPruneTechnique(WidthPruneConfig(sparsity=0.5), adapter_of(make_family_model("gemma4_text")))
