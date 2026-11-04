"""``width_prune``: structured removal of head groups, FFN channels and residual channels (FLAP, LLM-Pruner, Minitron).

The references are written from the papers. Each criterion's scores are recomputed from hooked activations or a
separate backward pass of the dense model. Pruning without compensation must equal the dense model with the input
columns of the removed heads and channels zeroed, and FLAP's compensation must equal the dense model whose removed
inputs are replaced by their calibration mean (An et al., 2024, Section 3.3). Minitron's depth criterion is checked
against a brute-force search over all windows.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import economize
from economize.backends import UnsupportedBackendError
from economize.errors import ConfigError, IncompatibleTechniquesError
from economize.techniques._structured import layout_for
from economize.techniques.kv_quant import KIVIConfig
from economize.techniques.prune import DepthPruneConfig, PruneConfig
from economize.techniques.quant import QuantConfig
from economize.techniques.slicegpt import SliceGPTConfig
from economize.techniques.width_prune import (
    FLAP,
    LayerShape,
    LLMPruner,
    Minitron,
    Scores,
    WidthPruneConfig,
    WidthPruneTechnique,
    contiguous_windows,
    keep_count,
    minitron_depth_select,
    select_adaptive,
)
from helpers import VOCAB, make_family_model, make_tiny_model, perturb, reference_greedy

TOKENS = torch.randint(0, VOCAB, (2, 16), generator=torch.Generator().manual_seed(3))


def _attach(model: nn.Module, config: WidthPruneConfig, adapter_of, attach_context, calib) -> WidthPruneTechnique:
    adapter = adapter_of(model)
    technique = WidthPruneTechnique(config, adapter)
    technique.attach(model, adapter, attach_context(calib))
    return technique


def _inputs(model: nn.Module, calib: torch.Tensor, name: str) -> list[torch.Tensor]:
    """The inputs of module ``name`` of every layer during one dense forward of ``calib``."""
    seen: list[torch.Tensor] = []
    handles = [layer.get_submodule(name).register_forward_pre_hook(lambda _m, a: seen.append(a[0].double()))
               for layer in model.model.layers]
    model(calib)
    for handle in handles:
        handle.remove()
    return seen


def _score(criterion: Any, model: nn.Module, config: WidthPruneConfig, adapter_of, calib) -> Scores:
    adapter = adapter_of(model)
    layers = config.pruned_layers(adapter.num_layers)
    return criterion.score(adapter, layout_for(adapter, "width_prune"), [calib], layers, config)


# ---- criteria ----------------------------------------------------------------------------------------------------


def test_flap_scores_are_the_fluctuation_times_the_column_norms(adapter_of, calib) -> None:
    model = make_tiny_model()
    config = WidthPruneConfig(method="flap", structure="uniform")
    attn, mlp = _inputs(model, calib, "self_attn.o_proj"), _inputs(model, calib, "mlp.down_proj")
    scores = _score(FLAP(), model, config, adapter_of, calib)
    for i, layer in enumerate(model.model.layers, start=1):
        o, down = layer.self_attn.o_proj.weight.double(), layer.mlp.down_proj.weight.double()
        x = attn[i - 1].reshape(-1, o.shape[1])
        head = (torch.var(x, 0) * o.pow(2).sum(0)).pow(2)
        torch.testing.assert_close(scores.groups[i], head.view(2, -1).sum(1))
        y = mlp[i - 1].reshape(-1, down.shape[1])
        torch.testing.assert_close(scores.channels[i], torch.var(y, 0) * down.pow(2).sum(0))
        torch.testing.assert_close(scores.attn_mean[i], x.mean(0))
        torch.testing.assert_close(scores.mlp_mean[i], y.mean(0))


def test_adaptive_flap_standardizes_and_averages(adapter_of, calib) -> None:
    model = make_tiny_model()
    uniform = _score(FLAP(), model, WidthPruneConfig(method="flap", structure="uniform"), adapter_of, calib)
    adaptive = _score(FLAP(), model, WidthPruneConfig(method="flap"), adapter_of, calib)
    attn = _inputs(model, calib, "self_attn.o_proj")
    assert adaptive.comparable and not uniform.comparable
    for i, layer in enumerate(model.model.layers, start=1):
        o = layer.self_attn.o_proj.weight.double()
        head = (torch.var(attn[i - 1].reshape(-1, o.shape[1]), 0) * o.pow(2).sum(0)).pow(2)
        standard = (head - head.mean()) / head.std()
        torch.testing.assert_close(adaptive.groups[i], standard.view(2, -1).mean(1))
        channels = uniform.channels[i]
        torch.testing.assert_close(adaptive.channels[i], (channels - channels.mean()) / channels.std())


def test_llm_pruner_scores_are_the_taylor_importance_of_each_group(adapter_of, calib) -> None:
    reference = make_tiny_model().requires_grad_(True)
    with torch.enable_grad():
        logits = reference(calib).logits[:, :-1]
        F.cross_entropy(logits.reshape(-1, VOCAB), calib[:, 1:].reshape(-1)).backward()
    config = WidthPruneConfig(method="llm_pruner", keep_first=1, keep_last=1)
    scores = _score(LLMPruner(), make_tiny_model(), config, adapter_of, calib)
    assert sorted(scores.groups) == [2, 3, 4, 5]
    hd, group = 16, 2  # 4 query heads of width 16 in 2 key/value groups
    for i in scores.groups:
        layer = reference.model.layers[i - 1]
        attn, mlp = layer.self_attn, layer.mlp

        def taylor(linear: nn.Linear) -> torch.Tensor:
            return (linear.weight * linear.weight.grad).abs().double()

        for g in range(2):
            q_rows, kv_rows = slice(g * group * hd, (g + 1) * group * hd), slice(g * hd, (g + 1) * hd)
            want = (taylor(attn.q_proj)[q_rows].sum() + taylor(attn.k_proj)[kv_rows].sum()
                    + taylor(attn.v_proj)[kv_rows].sum() + taylor(attn.o_proj)[:, q_rows].sum())
            torch.testing.assert_close(scores.groups[i][g], want)
        want = taylor(mlp.gate_proj).sum(1) + taylor(mlp.up_proj).sum(1) + taylor(mlp.down_proj).sum(0)
        torch.testing.assert_close(scores.channels[i], want)


def test_llm_pruner_leaves_no_gradients_behind(adapter_of, calib) -> None:
    model = make_tiny_model()
    _score(LLMPruner(), model, WidthPruneConfig(method="llm_pruner", keep_first=0, keep_last=0), adapter_of, calib)
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("chunk", [1, 2, 5])
def test_llm_pruner_scores_do_not_depend_on_the_gradient_chunk(chunk: int, adapter_of, calib) -> None:
    def score(config: WidthPruneConfig, model: nn.Module) -> Scores:
        adapter = adapter_of(model)
        return LLMPruner().score(adapter, layout_for(adapter, "width_prune"), list(calib.split(4)),
                                 config.pruned_layers(6), config)

    want = score(WidthPruneConfig(method="llm_pruner", keep_first=1, keep_last=0), make_tiny_model())
    model = make_tiny_model()
    got = score(WidthPruneConfig(method="llm_pruner", keep_first=1, keep_last=0, gradient_chunk=chunk), model)
    assert sorted(got.groups) == sorted(want.groups) == [2, 3, 4, 5, 6]
    for i in want.groups:
        torch.testing.assert_close(got.groups[i], want.groups[i])
        torch.testing.assert_close(got.channels[i], want.channels[i])
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())


def test_llm_pruner_holds_the_gradients_of_one_chunk_at_a_time(adapter_of, calib) -> None:
    model = make_tiny_model()
    layers = [model.model.layers[i] for i in range(6)]
    held: list[int] = []

    def count(_module: nn.Module, _args: Any, _output: Any) -> None:
        held.append(sum(1 for layer in layers if layer.mlp.down_proj.weight.requires_grad))

    handles = [layer.register_forward_hook(count) for layer in layers]
    _score(LLMPruner(), model, WidthPruneConfig(method="llm_pruner", keep_first=0, keep_last=0, gradient_chunk=2),
           adapter_of, calib)
    for handle in handles:
        handle.remove()
    assert set(held) == {2}


def test_minitron_scores_are_sequence_means_with_a_batch_l2(adapter_of, calib) -> None:
    model = make_tiny_model()
    config = WidthPruneConfig(method="minitron", hidden_sparsity=0.25)
    attn, mlp = _inputs(model, calib, "self_attn.o_proj"), _inputs(model, calib, "mlp.down_proj")
    norms: list[torch.Tensor] = []
    modules = [m for layer in model.model.layers for m in (layer.input_layernorm, layer.post_attention_layernorm)]
    handles = [m.register_forward_hook(lambda _m, _a, out: norms.append(out.double()))
               for m in (*modules, model.model.norm)]
    model(calib)
    for handle in handles:
        handle.remove()
    scores = _score(Minitron(), model, config, adapter_of, calib)
    for i in range(1, 7):
        x = attn[i - 1]
        heads = x.view(*x.shape[:-1], 4, 16).norm(dim=-1).mean(1).norm(dim=0)
        torch.testing.assert_close(scores.groups[i], heads.view(2, 2).norm(dim=1))
        torch.testing.assert_close(scores.channels[i], mlp[i - 1].abs().mean(1).norm(dim=0))
    assert scores.hidden is not None
    torch.testing.assert_close(scores.hidden, sum(n.abs().mean(1).norm(dim=0) for n in norms))


# ---- selection ---------------------------------------------------------------------------------------------------


def test_keep_count_rounds_like_the_references() -> None:
    assert keep_count(10, 0.25) == 8
    assert keep_count(128, 0.3, 8) == 88
    assert keep_count(4, 1.0) == 1
    assert keep_count(4, 0.0) == 4


def test_the_adaptive_threshold_matches_the_parameter_budget() -> None:
    # A group of one query head of width 3 holds (2 + 2) * 3 / 3 = 4 channel-equivalents; 24 in all, 12 kept.
    shape = LayerShape(heads=2, kv_heads=2, head_dim=3, intermediate=4)
    scores = Scores(
        groups={1: torch.tensor([5.0, -1.0]), 2: torch.tensor([-4.0, 2.0])},
        channels={1: torch.tensor([4.0, 3.0, -2.0, -3.0]), 2: torch.tensor([1.0, 0.0, -5.0, -6.0])},
    )
    selection = select_adaptive(scores, {1: shape, 2: shape}, 0.5)
    assert selection.groups[1].tolist() == [0] and selection.groups[2].tolist() == [1]
    assert selection.channels[1].tolist() == [0, 1] and selection.channels[2].tolist() == [0]


# ---- surgery -----------------------------------------------------------------------------------------------------


def _replace_removed(model: nn.Module, technique: WidthPruneTechnique, compensate: bool) -> None:
    """Zero (or set to the calibration mean) the inputs of the removed heads and channels of the dense ``model``."""
    for i, layer in enumerate(model.model.layers):
        heads = technique.groups_kept[i].repeat_interleave(2 * 16)
        channels = technique.channels_kept[i]
        means = (technique.attn_mean[i], technique.mlp_mean[i]) if compensate else (None, None)
        maps = ((layer.self_attn.o_proj, heads, means[0]), (layer.mlp.down_proj, channels, means[1]))
        for linear, kept, mean in maps:
            def pre(_m: nn.Module, args: Any, kept: torch.Tensor = kept, mean: Any = mean) -> Any:
                fill = torch.zeros_like(args[0]) if mean is None else mean.to(args[0]).expand_as(args[0])
                return (torch.where(kept.to(args[0].device), args[0], fill),)

            linear.register_forward_pre_hook(pre)


def test_uniform_pruning_is_the_dense_model_with_the_removed_inputs_zeroed(adapter_of, attach_context, calib) -> None:
    pruned = make_tiny_model()
    config = WidthPruneConfig(method="flap", structure="uniform", sparsity=0.5, compensate=False)
    technique = _attach(pruned, config, adapter_of, attach_context, calib)
    assert technique.summary["kv_heads"] == [1] * 6 and technique.summary["intermediate"] == [64] * 6
    dense = make_tiny_model()
    _replace_removed(dense, technique, compensate=False)
    torch.testing.assert_close(pruned(TOKENS).logits, dense(TOKENS).logits, atol=1e-5, rtol=1e-5)


def test_compensation_replaces_the_removed_inputs_by_their_mean(adapter_of, attach_context, calib) -> None:
    pruned = make_tiny_model()
    config = WidthPruneConfig(method="flap", structure="uniform", sparsity=0.5)
    technique = _attach(pruned, config, adapter_of, attach_context, calib)
    assert technique.summary["compensated"] == {"attention": True, "mlp": True}
    assert pruned.model.layers[0].self_attn.o_proj.bias is not None
    dense = make_tiny_model()
    _replace_removed(dense, technique, compensate=True)
    torch.testing.assert_close(pruned(TOKENS).logits, dense(TOKENS).logits, atol=1e-5, rtol=1e-5)


def test_compensation_lowers_the_held_out_loss(adapter_of, attach_context, calib, held_out, nll) -> None:
    plain, compensated = make_tiny_model(), make_tiny_model()
    for model, compensate in ((plain, False), (compensated, True)):
        _attach(model, WidthPruneConfig(method="flap", structure="uniform", sparsity=0.5, compensate=compensate),
                adapter_of, attach_context, calib)
    assert nll(compensated, held_out) < nll(plain, held_out)


def test_zero_sparsity_is_the_dense_model(adapter_of, attach_context, calib) -> None:
    pruned = make_tiny_model()
    _attach(pruned, WidthPruneConfig(method="minitron", sparsity=0.0), adapter_of, attach_context, calib)
    assert torch.equal(pruned(TOKENS).logits, make_tiny_model()(TOKENS).logits)


def test_minitron_prunes_the_residual_stream_into_a_plain_checkpoint(tmp_path: Path, adapter_of, attach_context,
                                                                     calib) -> None:
    from transformers import AutoModelForCausalLM

    pruned = make_tiny_model()
    config = WidthPruneConfig(method="minitron", sparsity=0.5, hidden_sparsity=0.25)
    technique = _attach(pruned, config, adapter_of, attach_context, calib)
    summary = technique.summary
    assert summary["width"] == 48 and summary["uniform"] and summary["plain_checkpoint"]
    assert pruned.config.hidden_size == 48 and pruned.config.head_dim == 16
    assert pruned.config.num_attention_heads == 2 and pruned.config.intermediate_size == 64
    assert summary["parameters"] == sum(p.numel() for p in pruned.parameters())
    pruned.save_pretrained(tmp_path / "plain")
    again = AutoModelForCausalLM.from_pretrained(tmp_path / "plain", dtype=torch.float32).eval()
    assert torch.equal(again(TOKENS).logits, pruned(TOKENS).logits)


def test_protected_layers_keep_their_widths(adapter_of, attach_context, calib) -> None:
    pruned = make_tiny_model()
    config = WidthPruneConfig(method="llm_pruner", sparsity=0.5, keep_first=2, keep_last=1)
    technique = _attach(pruned, config, adapter_of, attach_context, calib)
    assert technique.summary["kv_heads"] == [2, 2, 1, 1, 1, 2]
    assert technique.summary["intermediate"] == [128, 128, 64, 64, 64, 128]
    assert not technique.summary["uniform"] and not technique.summary["plain_checkpoint"]
    assert pruned.config.intermediate_size == 128  # the config keeps the widths it cannot describe


def test_adaptive_layers_of_different_widths_generate(calib) -> None:
    model = economize.apply(make_tiny_model(), WidthPruneConfig(method="flap", sparsity=0.4), calibration=[calib])
    summary = model.techniques["width_prune"].summary
    assert summary["structure"] == "adaptive" and len(set(summary["intermediate"])) > 1
    removed = summary["removed_parameters"] / (6 * (4 * 64 * 16 + 2 * 2 * 64 * 16 + 3 * 64 * 128))
    assert 0.3 < removed < 0.5
    prompt = TOKENS[:1, :8]
    out = model.generate(prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=6, do_sample=False)
    assert torch.equal(out[0, 8:], reference_greedy(economize.unwrap.base_model_of(model), prompt, 6))


def test_round_to_and_separate_ratios(adapter_of, attach_context, calib) -> None:
    pruned = make_tiny_model()
    config = WidthPruneConfig(method="minitron", attn_sparsity=0.0, mlp_sparsity=0.3, round_to=16)
    technique = _attach(pruned, config, adapter_of, attach_context, calib)
    assert technique.summary["kv_heads"] == [2] * 6 and technique.summary["intermediate"] == [96] * 6


def test_configuration_errors(adapter_of) -> None:
    with pytest.raises(ConfigError, match="adaptive"):
        WidthPruneConfig(method="minitron", structure="adaptive")
    with pytest.raises(ConfigError, match="attn_sparsity"):
        WidthPruneConfig(method="flap", attn_sparsity=0.2)
    with pytest.raises(ConfigError, match="residual"):
        WidthPruneConfig(method="flap", hidden_sparsity=0.2)
    with pytest.raises(ConfigError, match="compensation"):
        WidthPruneConfig(method="llm_pruner", compensate=True)
    with pytest.raises(ConfigError, match="gradient_chunk"):
        WidthPruneConfig(method="minitron", gradient_chunk=2)
    with pytest.raises(ConfigError):
        WidthPruneConfig(method="llm_pruner", gradient_chunk=-1)
    with pytest.raises(ConfigError):
        WidthPruneConfig(sparsity=1.5)
    with pytest.raises(ConfigError, match="protect all 6"):
        WidthPruneConfig(method="llm_pruner").validate(adapter_of(make_tiny_model()))


# ---- save / load and the recovery LoRA -----------------------------------------------------------------------------


@pytest.mark.parametrize("lora_rank", [0, 4])
def test_a_saved_pruning_replays_without_calibration(lora_rank: int, tmp_path: Path, calib) -> None:
    config = WidthPruneConfig(method="llm_pruner", sparsity=0.5, keep_first=1, keep_last=1, lora_rank=lora_rank)
    model = economize.apply(make_tiny_model(), config, calibration=[calib])
    if lora_rank:
        perturb(model.techniques["width_prune"].lora)
    expected = model(TOKENS).logits
    economize.save(model, tmp_path / "pruned")
    again = economize.load(make_tiny_model(), tmp_path / "pruned")
    assert torch.equal(again(TOKENS).logits, expected)


def test_a_loaded_recovery_lora_lists_its_parameters_like_a_fresh_one(tmp_path: Path, calib) -> None:
    # A resumed run matches the optimizer state to the trainable parameters by position.
    config = WidthPruneConfig(method="llm_pruner", sparsity=0.5, keep_first=1, keep_last=1, lora_rank=4)
    model = economize.apply(make_tiny_model(), config, calibration=[calib])
    economize.save(model, tmp_path / "pruned")
    again = economize.load(make_tiny_model(), tmp_path / "pruned")
    fresh = [name for name, _ in model.techniques["width_prune"].named_parameters()]
    loaded = [name for name, _ in again.techniques["width_prune"].named_parameters()]
    assert loaded == fresh and fresh != sorted(fresh)


def test_the_recovery_lora_is_the_only_trainable_part() -> None:
    assert WidthPruneTechnique.trainable_for(WidthPruneConfig(lora_rank=0)) == ()
    assert WidthPruneTechnique.trainable_for(WidthPruneConfig(lora_rank=8)) == ("lora",)


def test_a_merged_recovery_lora_is_the_hooked_one_folded_into_the_weights(tmp_path: Path, calib) -> None:
    import dataclasses

    config = WidthPruneConfig(method="llm_pruner", sparsity=0.5, keep_first=1, keep_last=1, lora_rank=4)
    model = economize.apply(make_tiny_model(), config, calibration=[calib])
    perturb(model.techniques["width_prune"].lora)
    expected = model(TOKENS).logits
    economize.save(model, tmp_path / "pruned")
    checkpoint = economize.read_checkpoint(tmp_path / "pruned")
    merged_config = dataclasses.replace(checkpoint.configs[0], merge_lora=True)
    merged = economize.attach(make_tiny_model(), economize.Checkpoint([merged_config], checkpoint.states))
    lora = merged.techniques["width_prune"].lora
    assert lora is not None and not lora.installed
    torch.testing.assert_close(merged(TOKENS).logits, expected, rtol=1e-4, atol=1e-4)  # float32 rounding of W + BA
    assert WidthPruneTechnique.trainable_for(merged_config) == ()
    weights = QuantConfig(format="int8", symmetric=True, group_size=None)
    states = {"width_prune": checkpoint.states["width_prune"]}
    quantized = economize.attach(make_tiny_model(), economize.Checkpoint([merged_config, weights], states))
    assert list(quantized.techniques) == ["width_prune", "quant"]
    with pytest.raises(ConfigError, match="merge_lora"):
        WidthPruneConfig(method="llm_pruner", merge_lora=True)


def test_logit_distill_trains_the_pruned_model_against_a_teacher(calib) -> None:
    from economize.train import fit
    from economize.train.objectives import LogitDistill

    student = economize.apply(make_tiny_model(), WidthPruneConfig(method="minitron", sparsity=0.5),
                              calibration=[calib])
    data = [calib[:4], calib[4:]]
    with torch.enable_grad():
        with pytest.raises(ConfigError, match="teacher_model"):
            fit(student, data, LogitDistill(parameters="all"), steps=1, output_dir=None)
        before = student.base_model.model.layers[0].mlp.up_proj.weight.clone()
        fit(student, data, LogitDistill(parameters="all"), teacher_model=make_tiny_model(), steps=2, output_dir=None)
    assert not torch.equal(student.base_model.model.layers[0].mlp.up_proj.weight, before)


def test_logit_distill_of_the_lora_leaves_the_base_weights(calib) -> None:
    from economize.train import fit
    from economize.train.objectives import LogitDistill

    config = WidthPruneConfig(method="llm_pruner", sparsity=0.5, keep_first=1, keep_last=1, lora_rank=4)
    student = economize.apply(make_tiny_model(), config, calibration=[calib])
    lora = student.techniques["width_prune"].lora
    base = {k: v.clone() for k, v in student.base_model.state_dict().items()}
    with torch.enable_grad():
        fit(student, [calib], LogitDistill(), teacher_model=make_tiny_model(), steps=2, output_dir=None)
    assert all(torch.equal(v, base[k]) for k, v in student.base_model.state_dict().items())
    assert any(delta.lora_B.abs().sum() > 0 for _, delta in lora._targets)


# ---- depth -------------------------------------------------------------------------------------------------------


def test_contiguous_windows() -> None:
    assert contiguous_windows([1, 2, 3, 5, 6], 2) == [[1, 2], [2, 3], [5, 6]]
    assert contiguous_windows([1, 3], 2) == []


def _without(model: nn.Module, window: list[int]) -> list[Any]:
    def skip(_m: nn.Module, args: Any, kwargs: Any, out: Any) -> Any:
        hidden = args[0] if args else kwargs["hidden_states"]
        return (hidden, *out[1:]) if isinstance(out, tuple) else hidden

    return [model.model.layers[i - 1].register_forward_hook(skip, with_kwargs=True) for i in window]


def test_minitron_depth_is_the_best_window_by_brute_force(adapter_of, calib) -> None:
    model = make_tiny_model()
    losses = {}
    for window in contiguous_windows(range(1, 7), 2):
        handles = _without(model, window)
        logits = model(calib).logits[:, :-1]
        losses[tuple(window)] = float(F.cross_entropy(logits.reshape(-1, VOCAB), calib[:, 1:].reshape(-1)))
        for handle in handles:
            handle.remove()
    best = min(losses, key=losses.__getitem__)
    assert minitron_depth_select(adapter_of(model), [calib], 2) == list(best)
    assert minitron_depth_select(adapter_of(model), [calib], 2, candidates=[1, 2, 3]) in ([1, 2], [2, 3])
    with pytest.raises(ConfigError, match="consecutive"):
        minitron_depth_select(adapter_of(model), [calib], 2, candidates=[1, 3])


def test_minitron_depth_pruning_removes_one_window(calib) -> None:
    model = economize.apply(make_tiny_model(), DepthPruneConfig(method="minitron", num_blocks=2), calibration=[calib])
    assert economize.unwrap.base_model_of(model).config.num_hidden_layers == 4


# ---- composition -------------------------------------------------------------------------------------------------


def test_width_pruning_composes_with_quantization_and_unstructured_pruning(calib) -> None:
    width = WidthPruneConfig(method="minitron", sparsity=0.5)
    for weights in (QuantConfig(format="int8", symmetric=True, group_size=None), PruneConfig(method="wanda")):
        model = economize.apply(make_tiny_model(), width, weights, calibration=[calib])
        assert list(model.techniques) == ["width_prune", weights.technique]
        assert model.techniques["width_prune"].summary["intermediate"] == [64] * 6
        prompt = TOKENS[:1, :8]
        out = model.generate(prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=4, do_sample=False)
        assert out.shape == (1, 12)


@pytest.mark.parametrize("structure", ["adaptive", "uniform"])
def test_a_drafters_layer_copies_are_pruned_like_the_layers(structure: str, tmp_path: Path, calib) -> None:
    # The trained copies of a full-adaptation drafter write the cache of the layers they stand in for: pruning the
    # heads of a layer must prune its copies the same way, and drafting must stay lossless on the pruned model.
    from economize import DecodeConfig, JumpConfig
    from economize.arch.base import adapter_for
    from economize.core.serialization import Checkpoint, read_checkpoint
    from economize.techniques.early_exit import EarlyExitConfig
    from economize.techniques.width_prune.stats import layer_shape

    width = (WidthPruneConfig(method="flap", sparsity=0.4, structure="adaptive") if structure == "adaptive" else
             WidthPruneConfig(method="flap", sparsity=0.4, attn_sparsity=0.5, structure="uniform"))
    for drafter, name in ((JumpConfig(target_layer=2, source_layer=4, adaptation="full"), "jump"),
                          (EarlyExitConfig(exit_layer=4, adaptation="full"), "early_exit")):
        trained = economize.apply(make_tiny_model(), drafter)
        perturb(trained.techniques[name])
        economize.save(trained, tmp_path / name)
        saved = read_checkpoint(tmp_path / name)
        base = make_tiny_model()
        fresh = WidthPruneTechnique(width, adapter_for(base)).state_dict()
        model = economize.attach(base, Checkpoint([*saved.configs, width], {**saved.states, "width_prune": fresh}),
                                 calibration=[calib], check_teacher=False)
        prune, copies = model.techniques["width_prune"], model.techniques[name].layer_copies()
        adapter = adapter_for(economize.unwrap.base_model_of(model))
        layout = layout_for(adapter, "width_prune")
        assert prune.summary["pruned_layer_copies"] == sorted(copies) and copies
        assert any(heads < 2 for heads in prune.summary["kv_heads"])
        for i, copy in copies.items():
            assert layer_shape(layout, copy) == layer_shape(layout, adapter.layer(i))
        state = model.techniques[name].state_dict()
        assert any(state[k].shape != v.shape for k, v in saved.states[name].items())  # narrowed, not rebuilt dense
        prompt = TOKENS[:1, :8]
        result = model.decode(prompt, 12, decoding=DecodeConfig(cycle_length=3))
        assert result.tokens.tolist() == reference_greedy(economize.unwrap.base_model_of(model), prompt, 12).tolist()


def test_copies_get_the_compensation_biases_of_a_bias_free_family(tmp_path: Path, calib) -> None:
    # FLAP compensates with biases that a bias-free family (Llama) only has once the config asks for them: the copies
    # of a full-adaptation drafter, rebuilt from that config, must get them too, or the saved state does not load.
    from economize import DecodeConfig, JumpConfig
    from economize.arch.base import adapter_for
    from economize.core.serialization import Checkpoint, read_checkpoint

    width = WidthPruneConfig(method="flap", sparsity=0.4)
    trained = economize.apply(make_family_model("llama"), JumpConfig(target_layer=2, source_layer=4, adaptation="full"))
    perturb(trained.techniques["jump"])
    economize.save(trained, tmp_path / "jump")
    saved = read_checkpoint(tmp_path / "jump")
    base = make_family_model("llama")
    fresh = WidthPruneTechnique(width, adapter_for(base)).state_dict()
    model = economize.attach(base, Checkpoint([*saved.configs, width], {**saved.states, "width_prune": fresh}),
                             calibration=[calib], check_teacher=False)
    copies = model.techniques["jump"].layer_copies()
    adapter = adapter_for(economize.unwrap.base_model_of(model))
    for i, copy in copies.items():
        assert dict(copy.named_parameters()).keys() == dict(adapter.layer(i).named_parameters()).keys()
        assert any(name.endswith("o_proj.bias") for name, _ in copy.named_parameters())
    economize.save(model, tmp_path / "pruned")
    loaded = economize.load(make_family_model("llama"), tmp_path / "pruned")  # strict
    prompt = TOKENS[:1, :8]
    result = loaded.decode(prompt, 12, decoding=DecodeConfig(cycle_length=3))
    assert result.tokens.tolist() == reference_greedy(economize.unwrap.base_model_of(loaded), prompt, 12).tolist()


def test_the_recovery_merge_holds_one_merged_map_at_a_time() -> None:
    from economize.techniques.width_prune.recovery import RecoveryLoRA

    assert inspect.isgeneratorfunction(inspect.unwrap(RecoveryLoRA.merged))


def test_conflicts(calib) -> None:
    with pytest.raises(IncompatibleTechniquesError, match="residual"):
        economize.apply(make_tiny_model(), WidthPruneConfig(method="minitron", hidden_sparsity=0.25),
                        SliceGPTConfig(rotation="global"), calibration=[calib])
    lora = WidthPruneConfig(method="llm_pruner", keep_first=1, keep_last=1, lora_rank=4)
    with pytest.raises(IncompatibleTechniquesError, match="recovery LoRA"):
        economize.apply(make_tiny_model(), lora, QuantConfig(format="int8", group_size=None), calibration=[calib])
    with pytest.raises(IncompatibleTechniquesError, match="head geometry"):
        economize.apply(make_tiny_model(), WidthPruneConfig(method="flap"),
                        KIVIConfig(bits=2, group_size=4, residual_length=4), calibration=[calib])
    model = economize.apply(make_tiny_model(), WidthPruneConfig(method="minitron", sparsity=0.5),
                            KIVIConfig(bits=2, group_size=4, residual_length=4), calibration=[calib])
    assert "kivi" in " ".join(model.techniques)


# ---- serving -----------------------------------------------------------------------------------------------------


def _saved(name: str, tmp_path: Path) -> Any:
    from transformers import AutoModelForCausalLM

    folder = tmp_path / name
    (make_family_model(name) if name != "qwen2_tiny" else make_tiny_model()).save_pretrained(folder)
    return AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32).eval()


def _served_logits(plan: Any) -> torch.Tensor:
    from transformers import AutoModelForCausalLM

    folder = getattr(plan, "model", None) or plan.model_path
    return AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32).eval()(TOKENS).logits


def test_compensated_flap_serves_on_vllm_for_llama(tmp_path: Path, calib) -> None:
    from economize.backends.sglang.serving import check_servable as sglang_servable
    from economize.backends.vllm.serving import to_vllm

    config = WidthPruneConfig(method="flap", structure="uniform", sparsity=0.5)
    model = economize.apply(_saved("llama", tmp_path), config, calibration=[calib])
    plan = to_vllm(model, tmp_path / "out")
    assert plan.techniques == {"width_prune": "native"}
    served = json.loads((Path(plan.model) / "config.json").read_text())
    assert served["attention_bias"] and served["mlp_bias"] and served["head_dim"] == 16
    torch.testing.assert_close(_served_logits(plan), economize.unwrap.base_model_of(model)(TOKENS).logits,
                               atol=1e-5, rtol=1e-5)
    with pytest.raises(UnsupportedBackendError, match="down-map bias"):
        sglang_servable(model)


def test_the_served_checkpoint_merges_the_recovery_lora(tmp_path: Path, calib) -> None:
    from economize.backends.sglang.serving import to_sglang
    from economize.backends.vllm.serving import to_vllm

    config = WidthPruneConfig(method="llm_pruner", sparsity=0.5, keep_first=0, keep_last=0, lora_rank=4)
    model = economize.apply(_saved("llama", tmp_path), config, calibration=[calib])
    perturb(model.techniques["width_prune"].lora)
    expected = model(TOKENS).logits
    for convert in (to_vllm, to_sglang):
        plan = convert(model, tmp_path / convert.__name__)
        assert plan.techniques == {"width_prune": "native"}
        torch.testing.assert_close(_served_logits(plan), expected, atol=1e-5, rtol=1e-5)


def test_head_pruning_of_qwen2_serves_on_sglang_only(tmp_path: Path, calib) -> None:
    from economize.backends.sglang.serving import to_sglang
    from economize.backends.vllm.serving import check_servable

    config = WidthPruneConfig(method="minitron", sparsity=0.5)
    model = economize.apply(_saved("qwen2_tiny", tmp_path), config, calibration=[calib])
    with pytest.raises(UnsupportedBackendError, match="head size"):
        check_servable(model)
    assert to_sglang(model, tmp_path / "out").techniques == {"width_prune": "native"}
    mlp_only = economize.apply(_saved("qwen2_tiny", tmp_path), WidthPruneConfig(method="minitron", attn_sparsity=0.0,
                                                                                 mlp_sparsity=0.5),
                               calibration=[calib])
    assert "width_prune" in check_servable(mlp_only)  # MLP-only pruning keeps hidden_size // heads == head_dim


def test_engines_refuse_layers_of_different_widths(tmp_path: Path, calib) -> None:
    from economize.backends.sglang.serving import check_servable as sglang_servable
    from economize.backends.vllm.serving import check_servable

    model = economize.apply(_saved("llama", tmp_path), WidthPruneConfig(method="flap"), calibration=[calib])
    for check in (check_servable, sglang_servable):
        with pytest.raises(UnsupportedBackendError, match="one config"):
            check(model)


def test_backend_support_is_real_where_served() -> None:
    from economize.core.compatibility import SupportStatus

    def statuses(config: WidthPruneConfig) -> dict[str, SupportStatus]:
        rows = economize.backend_support(config)
        return {name: rows[name].status for name in ("transformers", "vllm", "sglang")}

    real = SupportStatus.REAL
    assert statuses(WidthPruneConfig(method="minitron")) == {"transformers": real, "vllm": real, "sglang": real}
    assert statuses(WidthPruneConfig(method="flap"))["vllm"] is SupportStatus.UNSUPPORTED
    assert statuses(WidthPruneConfig(method="flap", structure="uniform"))["sglang"] is SupportStatus.UNSUPPORTED
    assert statuses(WidthPruneConfig(method="flap", structure="uniform", compensate=False))["sglang"] is real
    assert statuses(WidthPruneConfig(method="llm_pruner"))["vllm"] is SupportStatus.UNSUPPORTED
