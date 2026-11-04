"""Weight pruning (Wanda, SparseGPT, magnitude; unstructured and 2:4) and depth pruning (ShortGPT, SLEB)."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from economize.core.technique import Capability
from economize.errors import ConfigError, EconomizeError
from economize.techniques.prune import (
    PRUNE_METHODS,
    DepthPruneConfig,
    PruneConfig,
    Pruner,
    block_influence,
    check_nm,
    depth_prune_model,
    magnitude_mask,
    nm_mask,
    parse_pattern,
    prune_model,
    remove_layers,
    shortgpt_select,
    sleb_select,
    sparsegpt_prune,
    token_batches,
    unstructured_mask,
    wanda_mask,
)
from economize.techniques.prune.calibration import SquaredNormObserver, linear_modules, observing
from economize.techniques.prune.semi_structured import semi_structured_available, to_semi_structured
from helpers import reference_greedy

requires_weights_contract = pytest.mark.skipif(
    not hasattr(Capability, "WEIGHTS"), reason="needs the economize v0.2 Technique contract (Capability.WEIGHTS)"
)


def _output_error(weight: torch.Tensor, pruned: torch.Tensor, inputs: torch.Tensor) -> float:
    return (inputs @ (weight - pruned).T).pow(2).sum().item()


# ---- masks ---------------------------------------------------------------------------------------------------------


def test_parse_pattern() -> None:
    assert parse_pattern(0.5).fraction == 0.5 and not parse_pattern(0.5).structured
    two_four = parse_pattern("2:4")
    assert (two_four.n, two_four.m, two_four.fraction, str(two_four)) == (2, 4, 0.5, "2:4")
    for bad in (1.0, -0.1, "4:2", "0:4", "half", True):
        with pytest.raises(ConfigError):
            parse_pattern(bad)


def test_unstructured_mask_removes_exact_fraction_per_row() -> None:
    score = torch.rand(16, 40, generator=torch.Generator().manual_seed(0))
    mask = unstructured_mask(score, 0.3)
    assert (mask.sum(-1) == 12).all()
    kept_min = score.masked_fill(mask, float("inf")).min(-1).values
    removed_max = score.masked_fill(~mask, -float("inf")).max(-1).values
    assert (removed_max <= kept_min).all()
    assert unstructured_mask(score, 0.3, per_row=False).sum() == int(16 * 40 * 0.3)


def test_nm_mask_and_check() -> None:
    score = torch.rand(8, 32, generator=torch.Generator().manual_seed(0))
    mask = nm_mask(score, 2, 4)
    groups = mask.reshape(8, 8, 4)
    assert (groups.sum(-1) == 2).all()
    pruned = torch.ones(8, 32).masked_fill(mask, 0)
    assert check_nm(pruned, 2, 4) and not check_nm(torch.ones(8, 32), 2, 4)
    with pytest.raises(ConfigError):
        nm_mask(torch.rand(4, 6), 2, 4)


def test_wanda_mask_is_per_row_selection_of_weight_times_input_norm() -> None:
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(24, 48, generator=generator)
    norm = torch.rand(48, generator=generator) * 10
    mask = wanda_mask(weight, norm, parse_pattern(0.5))
    score = weight.abs() * norm.sqrt()
    expected = torch.zeros_like(mask).scatter_(-1, score.argsort(-1)[:, :24], True)
    assert torch.equal(mask, expected)
    # Scaling the norm (the reference divides by the number of sequences) changes nothing.
    assert torch.equal(wanda_mask(weight, norm * 7.3, parse_pattern(0.5)), mask)


def test_magnitude_mask_is_per_matrix() -> None:
    weight = torch.cat([torch.full((1, 8), 0.1), torch.full((1, 8), 1.0)])
    mask = magnitude_mask(weight, parse_pattern(0.5))
    assert mask[0].all() and not mask[1].any()


# ---- SparseGPT -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sparsity", [0.5, "2:4"])
def test_sparsegpt_error_below_magnitude_and_wanda(correlated_inputs, sparsity) -> None:
    pattern = parse_pattern(sparsity)
    weight = torch.randn(64, 128, generator=torch.Generator().manual_seed(1))
    inputs = correlated_inputs(128)
    pruned = sparsegpt_prune(weight, inputs.T @ inputs, pattern, blocksize=32)
    magnitude = weight.masked_fill(magnitude_mask(weight, pattern), 0)
    wanda = weight.masked_fill(wanda_mask(weight, inputs.pow(2).sum(0), pattern), 0)
    errors = {name: _output_error(weight, w, inputs) for name, w in [("sparsegpt", pruned), ("wanda", wanda)]}
    errors["magnitude"] = _output_error(weight, magnitude, inputs)
    assert errors["sparsegpt"] < errors["wanda"] < errors["magnitude"], errors
    if pattern.structured:
        assert check_nm(pruned, 2, 4)
    assert (pruned == 0).float().mean().item() == pytest.approx(0.5, abs=0.01)


# ---- model level ---------------------------------------------------------------------------------------------------


def _linear(adapter, layer: int, name: str):
    return linear_modules(adapter, adapter.layer(layer))[name]


def _expected_wanda_masks(model, adapter, layer, names, batches, pattern) -> dict:
    """Wanda masks of ``layer`` computed from the input norms of a full forward of ``model``."""
    modules = {name: _linear(adapter, layer, name) for name in names}
    with observing(modules, SquaredNormObserver) as stats:
        for batch in batches:
            model(batch)
    return {(layer, name): wanda_mask(modules[name].weight, stats[name].sum, pattern) for name in names}


def test_wanda_model_masks_follow_sequential_calibration(tiny2, adapter_of, calib) -> None:
    """Layer 1 is scored on the dense model's inputs, layer 2 on the outputs of the pruned layer 1."""
    names = ("self_attn.q_proj", "mlp.gate_proj", "mlp.down_proj")
    batches = token_batches(calib)
    pattern = parse_pattern(0.5)
    dense = copy.deepcopy(tiny2)
    expected = _expected_wanda_masks(dense, adapter_of(dense), 1, names, batches, pattern)
    first_only = copy.deepcopy(tiny2)
    prune_model(adapter_of(first_only), PruneConfig(method="wanda", sparsity=0.5, layers=(1,)), batches)
    expected |= _expected_wanda_masks(first_only, adapter_of(first_only), 2, names, batches, pattern)

    adapter = adapter_of(tiny2)
    summary = prune_model(adapter, PruneConfig(method="wanda", sparsity=0.5), batches)
    assert summary["sparsity"] == pytest.approx(0.5) and summary["pattern"] == "0.5"
    for (layer, name), mask in expected.items():
        assert torch.equal(_linear(adapter, layer, name).weight == 0, mask), (layer, name)



def _sparsegpt_block(method: str) -> dict[str, int]:
    """A column block that fits the tiny model; only SparseGPT reads ``blocksize``."""
    return {"blocksize": 32} if method == "sparsegpt" else {}

@pytest.mark.parametrize("method", ["magnitude", "wanda", "sparsegpt"])
def test_two_four_pattern_holds_for_every_method(tiny2, adapter_of, calib, held_out, nll, method) -> None:
    adapter = adapter_of(tiny2)
    dense = nll(tiny2, held_out)
    config = PruneConfig(method=method, sparsity="2:4", **_sparsegpt_block(method))
    summary = prune_model(adapter, config, token_batches(calib))
    assert summary["sparsity"] == pytest.approx(0.5) and summary["pattern"] == "2:4"
    for index in (1, 2):
        for module in linear_modules(adapter, adapter.layer(index)).values():
            assert check_nm(module.weight, 2, 4)
    pruned = nll(tiny2, held_out)
    assert torch.isfinite(torch.tensor(pruned)) and abs(pruned - dense) < 3.0


def test_targets_exclude_and_layers(tiny2, adapter_of, calib) -> None:
    adapter = adapter_of(tiny2)
    config = PruneConfig(method="magnitude", sparsity=0.5, targets=("mlp.*",), exclude=("*down*",), layers=(2,))
    summary = prune_model(adapter, config, token_batches(calib))
    assert summary["weights"] == 2 * 128 * 64
    for index in (1, 2):
        for name, module in linear_modules(adapter, adapter.layer(index)).items():
            touched = index == 2 and name in ("mlp.gate_proj", "mlp.up_proj")
            assert ((module.weight == 0).float().mean().item() == pytest.approx(0.5)) == touched, (index, name)


def test_sparsegpt_model_error_below_magnitude(tiny_factory, adapter_of, calib, held_out) -> None:
    errors = {}
    reference = tiny_factory(num_layers=2)(held_out).logits
    for method in ("magnitude", "wanda", "sparsegpt"):
        model = tiny_factory(num_layers=2)
        config = PruneConfig(method=method, sparsity=0.5, **_sparsegpt_block(method))
        prune_model(adapter_of(model), config, token_batches(calib))
        errors[method] = (model(held_out).logits - reference).pow(2).mean().item()
    assert errors["sparsegpt"] < errors["magnitude"], errors


def test_calibrated_methods_need_data(tiny2, adapter_of) -> None:
    with pytest.raises(ConfigError):
        prune_model(adapter_of(tiny2), PruneConfig(method="wanda"))
    prune_model(adapter_of(tiny2), PruneConfig(method="magnitude"))  # data-free


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "obs"},
        {"sparsity": 1.0},
        {"sparsity": "4:2"},
        {"sparsity": "2:4", "blocksize": 30},
        {"semi_structured": True, "sparsity": 0.5},
        {"targets": ("",)},
        {"layers": (0,)},
        {"percdamp": -0.1},
    ],
)
def test_prune_config_validation(kwargs) -> None:
    with pytest.raises(ConfigError):
        PruneConfig(**kwargs)


def test_semi_structured_conversion(tiny2, adapter_of) -> None:
    if not semi_structured_available():
        with pytest.raises(ConfigError):
            to_semi_structured(_linear(adapter_of(tiny2), 1, "mlp.up_proj"))
        pytest.skip("semi-structured 2:4 kernels need a CUDA GPU with compute capability >= 8.0")
    model = tiny2.to("cuda", torch.float16)
    adapter = adapter_of(model)
    ids = torch.randint(0, 128, (1, 64), device="cuda")
    prune_model(adapter, PruneConfig(method="magnitude", sparsity="2:4", targets=("mlp.*",)))
    dense = model(ids).logits
    prune_model(adapter, PruneConfig(method="magnitude", sparsity="2:4", targets=("mlp.*",), semi_structured=True))
    assert torch.allclose(model(ids).logits, dense, atol=5e-2)


# ---- depth pruning -------------------------------------------------------------------------------------------------


def _make_identity(adapter, index: int) -> None:
    """Zero the maps that write into the residual stream, so layer ``index`` returns its input exactly."""
    modules = linear_modules(adapter, adapter.layer(index))
    modules["self_attn.o_proj"].weight.zero_()
    modules["mlp.down_proj"].weight.zero_()


def test_block_influence_matches_hidden_states(tiny_factory, adapter_of, calib) -> None:
    model = tiny_factory(num_layers=4)
    batches = token_batches(calib)
    influence = block_influence(adapter_of(model), batches)
    expected = torch.zeros(3)
    for batch in batches:
        states = model(batch, output_hidden_states=True).hidden_states
        for i in range(3):  # states[L] has the final norm applied, so the last layer is not checked
            expected[i] += (1 - F.cosine_similarity(states[i], states[i + 1], dim=-1)).sum()
    assert torch.allclose(influence[:3], expected / calib.numel(), atol=1e-5)
    assert ((influence > 0) & (influence < 2)).all()


def test_shortgpt_removes_lowest_influence_block(tiny_factory, adapter_of, calib, held_out) -> None:
    model = tiny_factory(num_layers=4)
    adapter = adapter_of(model)
    _make_identity(adapter, 3)
    reference = model(held_out).logits
    influence = block_influence(adapter, token_batches(calib))
    assert influence[2] < 1e-6 and int(influence.argmin()) == 2
    assert shortgpt_select(influence, 2) == sorted([3, int(influence.argsort()[1]) + 1])
    removed = depth_prune_model(adapter, DepthPruneConfig("shortgpt", 1), token_batches(calib))
    assert removed == [3] and adapter.num_layers == 3 and model.config.num_hidden_layers == 3
    assert torch.allclose(model(held_out).logits, reference, atol=1e-5)


def _naive_nll(model, batches) -> float:
    total = 0.0
    for batch in batches:
        logits = model(batch).logits[:, :-1]
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1), reduction="sum").item()
    return total


def test_sleb_matches_naive_iterative_reevaluation(tiny_factory, adapter_of, calib) -> None:
    model = tiny_factory(num_layers=4)
    batches = token_batches(calib[:4])
    alive, expected = [1, 2, 3, 4], []
    for _ in range(2):
        losses = {}
        for candidate in alive:
            trial = copy.deepcopy(model)
            remove_layers(adapter_of(trial), [i for i in (1, 2, 3, 4) if i not in alive or i == candidate])
            losses[candidate] = _naive_nll(trial, batches)
        best = min(losses, key=losses.get)
        alive.remove(best)
        expected.append(best)
    assert sleb_select(adapter_of(model), batches, 2) == sorted(expected)


def test_sleb_removes_identity_block(tiny_factory, adapter_of, calib) -> None:
    model = tiny_factory(num_layers=4)
    adapter = adapter_of(model)
    _make_identity(adapter, 2)
    baseline = _naive_nll(model, token_batches(calib[:4]))
    chosen = sleb_select(adapter, token_batches(calib[:4]), 1)
    trial = copy.deepcopy(model)
    remove_layers(adapter_of(trial), chosen)
    assert _naive_nll(trial, token_batches(calib[:4])) <= baseline + 1e-3


def test_removed_layers_keep_kv_cache_decoding_exact(tiny_factory, adapter_of, calib, prompt) -> None:
    model = tiny_factory(num_layers=4)
    adapter = adapter_of(model)
    remove_layers(adapter, [2, 3])
    assert [layer.self_attn.layer_idx for layer in adapter.layers] == [0, 1]
    cached = model.generate(prompt.view(1, -1), max_new_tokens=8, do_sample=False)[0, prompt.numel() :]
    assert torch.equal(cached, reference_greedy(model, prompt, 8))


def test_depth_config_validation(tiny2, adapter_of) -> None:
    with pytest.raises(ConfigError):
        DepthPruneConfig(method="random")
    with pytest.raises(ConfigError):
        DepthPruneConfig(num_blocks=0)
    with pytest.raises(ConfigError):
        DepthPruneConfig(num_blocks=2).validate(adapter_of(tiny2))


# ---- registered techniques -----------------------------------------------------------------------------------------


@requires_weights_contract
def test_prune_technique_attach_and_detach(tiny2, adapter_of, calib, attach_context) -> None:
    from economize.techniques.prune import PruneTechnique

    adapter = adapter_of(tiny2)
    technique = PruneTechnique(PruneConfig(method="wanda", sparsity="2:4"), adapter)
    assert technique.capabilities == Capability.WEIGHTS and technique.order == 10 and not technique.lossless
    with pytest.raises(ConfigError, match="calibration"):
        technique.attach(tiny2, adapter, attach_context(None))
    technique.attach(tiny2, adapter, attach_context(calib))
    assert technique.summary["sparsity"] == pytest.approx(0.5)
    with pytest.raises(EconomizeError):
        technique.detach()


@requires_weights_contract
def test_semi_structured_prune_rejects_later_weight_transforms(tiny2, adapter_of) -> None:
    from economize.errors import IncompatibleTechniquesError
    from economize.techniques.prune import PruneTechnique
    from economize.techniques.quant import QuantConfig, QuantTechnique

    adapter = adapter_of(tiny2)
    quant = QuantTechnique(QuantConfig(), adapter)
    PruneTechnique(PruneConfig(sparsity="2:4"), adapter).check_compatible([quant])
    with pytest.raises(IncompatibleTechniquesError):
        PruneTechnique(PruneConfig(sparsity="2:4", semi_structured=True), adapter).check_compatible([quant])


@requires_weights_contract
def test_depth_prune_technique(tiny_factory, adapter_of, calib, attach_context) -> None:
    from economize.techniques.prune import DepthPruneTechnique

    model = tiny_factory(num_layers=4)
    adapter = adapter_of(model)
    _make_identity(adapter, 2)
    technique = DepthPruneTechnique(DepthPruneConfig("shortgpt", 1), adapter)
    assert technique.order < 10
    technique.attach(model, adapter, attach_context(calib))
    assert technique.summary == {"method": "shortgpt", "removed_layers": [2], "num_layers": 3}
    assert technique.config.layers == (2,)  # the choice is recorded, so a saved checkpoint repeats it


@requires_weights_contract
def test_depth_prune_explicit_layers_need_no_calibration(tiny_factory, adapter_of, calib, attach_context,
                                                          held_out) -> None:
    from economize.techniques.prune import DepthPruneTechnique

    chosen = tiny_factory(num_layers=4)
    adapter = adapter_of(chosen)
    _make_identity(adapter, 3)
    technique = DepthPruneTechnique(DepthPruneConfig("shortgpt", 1), adapter)
    technique.attach(chosen, adapter, attach_context(calib))
    replay = tiny_factory(num_layers=4)
    _make_identity(adapter_of(replay), 3)
    again = DepthPruneTechnique(technique.config, adapter_of(replay))
    assert not again.needs_calibration()
    again.attach(replay, adapter_of(replay), attach_context(None))
    assert again.summary["removed_layers"] == [3] and replay.config.num_hidden_layers == 3
    assert torch.equal(replay(held_out).logits, chosen(held_out).logits)


def test_depth_config_explicit_layers_validation(tiny2, adapter_of) -> None:
    with pytest.raises(ConfigError, match="num_blocks"):
        DepthPruneConfig(num_blocks=2, layers=(1,))
    with pytest.raises(ConfigError):
        DepthPruneConfig(layers=(0,))
    with pytest.raises(ConfigError):
        DepthPruneConfig(layers=(3,)).validate(adapter_of(tiny2))
    assert DepthPruneConfig.from_dict({"num_blocks": 1, "layers": [2]}).layers == (2,)


# ---- per-method options ----
@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"method": "wanda", "blocksize": 32}, "blocksize"),
        ({"method": "magnitude", "percdamp": 0.1}, "percdamp"),
    ],
)
def test_prune_rejects_knobs_the_method_does_not_read(kwargs: dict, name: str) -> None:
    with pytest.raises(ConfigError, match=name):
        PruneConfig(**kwargs)


def test_sparsegpt_checks_its_own_block() -> None:
    PruneConfig(method="sparsegpt", blocksize=32, percdamp=0.1, sparsity="2:4")
    with pytest.raises(ConfigError, match="multiple of M"):
        PruneConfig(method="sparsegpt", blocksize=30, sparsity="2:4")


def test_pruner_observer_is_abstract() -> None:
    class NoObserver(Pruner):
        def prune(self, weight, stats, pattern, config):  # type: ignore[no-untyped-def]
            return weight

    with pytest.raises(TypeError):
        NoObserver()  # type: ignore[abstract]
    assert PRUNE_METHODS.get("magnitude")().observer(torch.nn.Linear(2, 2)) is None


# ---- SparseGPT rejects N:M patterns that do not tile the input dimension ----
@pytest.mark.parametrize("pattern", ["2:4", "1:4"])
@pytest.mark.parametrize("columns", [6, 10])
def test_sparsegpt_rejects_nm_pattern_on_indivisible_input(pattern: str, columns: int) -> None:
    """Before the fix 2:4 silently left a short last group (not N:M) and 1:4 raised a RuntimeError from topk."""
    weight = torch.randn(4, columns, generator=torch.Generator().manual_seed(0))
    x = torch.randn(32, columns, generator=torch.Generator().manual_seed(1))
    with pytest.raises(ConfigError, match="multiple"):
        sparsegpt_prune(weight, x.T @ x, parse_pattern(pattern))


def test_sparsegpt_rejects_blocksize_not_multiple_of_m() -> None:
    weight = torch.randn(4, 12, generator=torch.Generator().manual_seed(0))
    x = torch.randn(32, 12, generator=torch.Generator().manual_seed(1))
    with pytest.raises(ConfigError, match="blocksize"):
        sparsegpt_prune(weight, x.T @ x, parse_pattern("2:4"), blocksize=6)


def test_sparsegpt_nm_on_divisible_input_is_exact_nm() -> None:
    weight = torch.randn(4, 12, generator=torch.Generator().manual_seed(0))
    x = torch.randn(32, 12, generator=torch.Generator().manual_seed(1))
    pruned = sparsegpt_prune(weight, x.T @ x, parse_pattern("2:4"))
    assert ((pruned.reshape(4, 3, 4) == 0).sum(-1) == 2).all()
