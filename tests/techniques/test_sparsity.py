"""TEAL activation sparsity: the pure functions, calibration against the target sparsity, the hooks, the technique."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import economize
from economize import DecodeConfig
from economize.arch import adapter_for
from economize.core.technique import Capability
from economize.errors import ConfigError
from economize.runtime import phase
from economize.techniques.lookup import LookupConfig
from economize.techniques.prune import token_batches
from economize.techniques.sparsity import (
    PhasePolicy,
    TealConfig,
    block_sparsity,
    calibrate_teal,
    greedy_allocation,
    greedy_error,
    install_teal,
    magnitude_quantile,
    measure_sparsity,
    sparsify,
    sparsify_tokens,
    teal_modules,
)
from helpers import VOCAB, make_family_model, make_tiny_model, reference_greedy

requires_activations_contract = pytest.mark.skipif(
    not hasattr(Capability, "ACTIVATIONS"),
    reason="needs the economize v0.2 Technique contract (Capability.ACTIVATIONS)",
)

# Cheap greedy settings: the search costs one layer forward per candidate map and accepted move.
FAST = {"greedy_step": 0.1, "greedy_batches": 2}


# ---- pure functions ------------------------------------------------------------------------------------------------


def test_sparsify_zeroes_entries_at_or_below_threshold() -> None:
    x = torch.tensor([-2.0, -0.5, 0.0, 0.3, 0.5, 1.0])
    assert torch.equal(sparsify(x, 0.5), torch.tensor([-2.0, 0.0, 0.0, 0.0, 0.0, 1.0]))


def test_sparsify_tokens_prefill_modes() -> None:
    x = torch.randn(2, 6, 4, generator=torch.Generator().manual_seed(0))
    t = 0.7
    assert torch.equal(sparsify_tokens(x, t, "dense"), x)
    assert torch.equal(sparsify_tokens(x, t, "all"), sparsify(x, t))
    half = sparsify_tokens(x, t, "second_half")
    assert torch.equal(half[:, :3], x[:, :3])
    assert torch.equal(half[:, 3:], sparsify(x[:, 3:], t))
    one = x[:, :1]
    for mode in ("dense", "second_half", "all"):  # single-token calls (decoding) are always sparsified
        assert torch.equal(sparsify_tokens(one, t, mode), sparsify(one, t))


@pytest.mark.parametrize("q", [0.0, 0.1, 0.25, 0.5, 0.9])
def test_magnitude_quantile_zeroes_the_fraction(q: float) -> None:
    magnitudes = torch.randn(10_000, generator=torch.Generator().manual_seed(0)).abs().sort().values
    threshold = magnitude_quantile(magnitudes, q)
    assert (magnitudes <= threshold).float().mean().item() == pytest.approx(q, abs=1e-4)
    if q == 0:
        assert threshold == 0.0


def test_greedy_allocation_reaches_target_with_size_scaled_steps() -> None:
    sizes = {"small": 100, "large": 300}
    # Sparsifying "large" is cheap and "small" is expensive, so the search should favour "large".
    levels, path = greedy_allocation(
        lambda lv: 10 * lv["small"] ** 2 + lv["large"] ** 2, sizes, 0.5, step=0.1, unit=100
    )
    increment = 0.1 * 100 / 400  # every move adds step * unit / total parameters to the block sparsity
    assert block_sparsity(levels, sizes) == pytest.approx(len(path) * increment)
    assert 0.5 - 1e-9 <= block_sparsity(levels, sizes) < 0.5 + increment
    assert [s.block_sparsity for s in path] == pytest.approx([increment * (i + 1) for i in range(len(path))])
    assert levels["large"] > levels["small"]
    assert levels["small"] / 0.1 == pytest.approx(round(levels["small"] / 0.1))
    assert levels["large"] / (0.1 / 3) == pytest.approx(round(levels["large"] / (0.1 / 3)))


def test_greedy_allocation_stops_at_full_sparsity() -> None:
    levels, _ = greedy_allocation(lambda lv: sum(lv.values()), {"a": 10, "b": 10}, 1.0, step=0.5)
    assert levels == {"a": 1.0, "b": 1.0}


def test_greedy_error_measures_only_the_trailing_positions() -> None:
    reference = torch.zeros(1, 8, 2)
    early = reference.clone()
    early[:, :6] = 1.0
    assert greedy_error(reference, early, 0.25).item() == 0.0
    late = reference.clone()
    late[:, 7] = torch.tensor([3.0, 4.0])
    assert greedy_error(reference, late, 0.25).item() == pytest.approx(3.5)  # mean over dims of |3|, |4|


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sparsity": 1.5},
        {"allocation": "random"},
        {"prefill": "first_half"},
        {"backend": "cutlass-teal"},
        {"greedy_batches": -1},
        {"greedy_step": 0.0},
        {"targets": ()},
        {"layers": (0,)},
    ],
)
def test_config_validation(kwargs: dict) -> None:
    with pytest.raises(ConfigError):
        TealConfig(**kwargs)


def test_config_validate_rejects_missing_layer(tiny2, adapter_of) -> None:
    with pytest.raises(ConfigError):
        TealConfig(layers=(3,)).validate(adapter_of(tiny2))


# ---- model level ---------------------------------------------------------------------------------------------------


def test_teal_modules_order_and_selection(tiny2, adapter_of) -> None:
    adapter = adapter_of(tiny2)
    keys = list(teal_modules(adapter, TealConfig()))
    assert len(keys) == 2 * 7 and keys[0].startswith("1.") and keys[-1].startswith("2.")
    selected = teal_modules(adapter, TealConfig(layers=(2,), targets=("mlp.*",), exclude=("*down*",)))
    assert sorted(selected) == ["2.mlp.gate_proj", "2.mlp.up_proj"]


@pytest.mark.parametrize("allocation", ["uniform", "greedy"])
@pytest.mark.parametrize("target", [0.25, 0.5])
def test_calibrated_sparsity_matches_target(tiny2, adapter_of, calib, held_out, allocation: str, target: float) -> None:
    adapter = adapter_of(tiny2)
    config = TealConfig(sparsity=target, allocation=allocation, prefill="all", **FAST)
    batches = token_batches(calib)
    calibration = calibrate_teal(adapter, config, batches)
    # Greedy overshoots by at most one move: step * hidden**2 / layer parameters.
    assert target - 1e-6 <= calibration.model_sparsity <= target + 0.1 * 64**2 / 36_864 + 1e-6
    if allocation == "greedy":
        assert len(set(calibration.sparsities.values())) > 1
    installed, _ = install_teal(adapter, calibration.thresholds, config)
    try:
        on_calibration = measure_sparsity(adapter, config, lambda: [tiny2(b) for b in batches])
        on_held_out = measure_sparsity(adapter, config, lambda: tiny2(held_out))
    finally:
        installed.remove()
    assert on_calibration["model"] == pytest.approx(target, abs=0.02)
    assert on_held_out["model"] == pytest.approx(target, abs=0.04)


def test_greedy_allocation_beats_uniform(tiny_factory, adapter_of, calib, held_out) -> None:
    """At 50% sparsity the greedy allocation changes the logits less than the uniform one, as in the
    paper. Over six random models the gap was 30-50% of the uniform error; two are checked here."""
    for seed in range(2):
        model = tiny_factory(num_layers=2, seed=seed)
        adapter = adapter_of(model)
        reference = model(held_out).logits
        errors = {}
        for allocation in ("uniform", "greedy"):
            config = TealConfig(sparsity=0.5, allocation=allocation, prefill="all", greedy_step=0.1, greedy_batches=1)
            calibration = calibrate_teal(adapter, config, token_batches(calib[:4]))
            installed, _ = install_teal(adapter, calibration.thresholds, config)
            errors[allocation] = (model(held_out).logits - reference).pow(2).mean().item()
            installed.remove()
        assert errors["greedy"] < 0.9 * errors["uniform"], errors


def test_install_and_remove_restore_logits_exactly(tiny2, adapter_of, calib, held_out) -> None:
    adapter = adapter_of(tiny2)
    config = TealConfig(sparsity=0.5, allocation="uniform", prefill="all")
    reference = tiny2(held_out).logits
    installed, live = install_teal(adapter, calibrate_teal(adapter, config, token_batches(calib)).thresholds, config)
    sparse = tiny2(held_out).logits
    assert not torch.allclose(sparse, reference)
    assert (sparse - reference).abs().max() < 50 and torch.isfinite(sparse).all()
    for threshold in live.values():  # thresholds are read at call time
        threshold.zero_()
    assert torch.allclose(tiny2(held_out).logits, reference, atol=1e-5)
    installed.remove()
    assert torch.equal(tiny2(held_out).logits, reference)


def test_dense_prefill_sparsifies_only_single_token_calls(tiny2, adapter_of, calib, held_out) -> None:
    adapter = adapter_of(tiny2)
    config = TealConfig(sparsity=0.5, allocation="uniform")  # prefill="dense", the default
    step = held_out[:, :1]
    dense_prompt, dense_step = tiny2(held_out).logits, tiny2(step).logits
    installed, _ = install_teal(adapter, calibrate_teal(adapter, config, token_batches(calib)).thresholds, config)
    try:
        assert torch.equal(tiny2(held_out).logits, dense_prompt)
        assert not torch.equal(tiny2(step).logits, dense_step)
    finally:
        installed.remove()


def test_install_skips_dense_maps(tiny2, adapter_of) -> None:
    adapter = adapter_of(tiny2)
    config = TealConfig()
    keys = list(teal_modules(adapter, config))
    thresholds = {key: (0.0 if i % 2 else 0.1) for i, key in enumerate(keys)}
    installed, live = install_teal(adapter, thresholds, config)
    assert len(installed.handles) == len(live) == (len(keys) + 1) // 2
    installed.remove()


# ---- registered technique ------------------------------------------------------------------------------------------


@requires_activations_contract
def test_technique_attach_detach_and_state_round_trip(tiny2, adapter_of, calib, held_out, attach_context) -> None:
    from economize.techniques.sparsity import TealTechnique

    adapter = adapter_of(tiny2)
    config = TealConfig(sparsity=0.4, allocation="greedy", prefill="all", **FAST)
    reference = tiny2(held_out).logits
    technique = TealTechnique(config, adapter)
    assert not technique.calibrated
    technique.attach(tiny2, adapter, attach_context(calib))
    assert technique.calibrated and technique.summary["installed"]
    assert technique.summary["model_sparsity"] == pytest.approx(0.4, abs=0.012)
    sparse = tiny2(held_out).logits
    state = {k: v.clone() for k, v in technique.state_dict().items()}
    assert set(state) == {"sparsities", "thresholds"} and state["thresholds"].shape == (14,)
    technique.detach()
    assert torch.equal(tiny2(held_out).logits, reference)

    restored = TealTechnique(config, adapter)
    restored.load_state_dict(state, strict=True)
    assert restored.calibrated
    restored.attach(tiny2, adapter, None)  # loaded thresholds: no calibration data needed
    assert torch.equal(tiny2(held_out).logits, sparse)
    restored.detach()


@requires_activations_contract
def test_technique_needs_calibration_data(tiny2, adapter_of, attach_context) -> None:
    from economize.techniques.sparsity import TealTechnique

    adapter = adapter_of(tiny2)
    technique = TealTechnique(TealConfig(), adapter)
    with pytest.raises(ConfigError, match="calibration"):
        technique.attach(tiny2, adapter, attach_context(None))
    assert technique.capabilities == Capability.ACTIVATIONS and not technique.lossless


@requires_activations_contract
def test_loading_state_after_attach_reinstalls_hooks(tiny2, adapter_of, calib, held_out, attach_context) -> None:
    from economize.techniques.sparsity import TealTechnique

    adapter = adapter_of(tiny2)
    config = TealConfig(sparsity=0.5, allocation="uniform", prefill="all")
    first = TealTechnique(config, adapter)
    first.attach(tiny2, adapter, attach_context(calib))
    target = tiny2(held_out).logits
    state = first.state_dict()
    first.detach()
    second = TealTechnique(TealConfig(sparsity=0.25, allocation="uniform", prefill="all"), adapter)
    second.attach(tiny2, adapter, attach_context(calib))
    second.load_state_dict(state)
    assert torch.equal(tiny2(held_out).logits, target)
    second.detach()


# ---- stacking with the weight techniques ---------------------------------------------------------------------------


def test_weight_and_activation_stack_end_to_end(tiny2, adapter_of, calib, held_out, nll) -> None:
    """prune -> low rank -> int4 GPTQ -> TEAL in recipe order, each calibrated on the already transformed model."""
    from economize.techniques.lowrank import LowRankConfig, lowrank_model
    from economize.techniques.prune import PruneConfig, prune_model
    from economize.techniques.quant import QuantConfig, quantize_model

    adapter = adapter_of(tiny2)
    batches = token_batches(calib)
    dense = nll(tiny2, held_out)
    prune_model(adapter, PruneConfig(method="wanda", sparsity=0.3), batches)
    lowrank_model(adapter, LowRankConfig(compression=0.1, targets=("mlp.*",)), batches)
    quantize_model(adapter, QuantConfig(format="int4", method="gptq", group_size=32), batches)
    config = TealConfig(sparsity=0.25, allocation="uniform", prefill="all")
    installed, _ = install_teal(adapter, calibrate_teal(adapter, config, batches).thresholds, config)
    stacked = nll(tiny2, held_out)
    installed.remove()
    assert torch.isfinite(torch.tensor(stacked)) and abs(stacked - dense) < 2.0
    assert stacked != dense


# ---- exact passes stay dense under prefill="dense" ----
@pytest.mark.parametrize("name", ["prefill", "refresh"])
def test_dense_mode_keeps_single_token_exact_passes_dense(name: str) -> None:
    x = torch.randn(2, 1, 8)
    with phase(name):
        assert torch.equal(sparsify_tokens(x, 0.5, "dense"), x)
        assert torch.equal(sparsify_tokens(x, 0.5, "all"), sparsify(x, 0.5))
        assert torch.equal(sparsify_tokens(x, 0.5, "second_half"), sparsify(x, 0.5))


@pytest.mark.parametrize("name", [None, "decode", "draft"])
def test_dense_mode_sparsifies_single_token_steps(name: str | None) -> None:
    x = torch.randn(2, 1, 8)
    with phase(name):
        assert torch.equal(sparsify_tokens(x, 0.5, "dense"), sparsify(x, 0.5))
        several = torch.randn(2, 3, 8)
        # decoding and drafting steps are classified by their phase, whatever their shape; only calls outside the
        # decoding loop fall back to the number of tokens per sequence
        expected = several if name is None else sparsify(several, 0.5)
        assert torch.equal(sparsify_tokens(several, 0.5, "dense"), expected)


def test_lossless_decoding_with_empty_drafts_matches_dense_greedy() -> None:
    """A lookup drafter proposes no tokens when no n-gram matches, so many refresh passes cover a single token.
    Before the fix these were sparsified and the "lossless" output differed from greedy decoding of the dense model."""
    calib = torch.randint(0, VOCAB, (8, 32), generator=torch.Generator().manual_seed(0))
    base = make_tiny_model()
    generator = torch.Generator().manual_seed(1)
    prompts = [torch.randint(1, VOCAB, (n,), generator=generator) for n in (12, 7, 20)]
    with torch.no_grad():
        expected = [reference_greedy(base, p, 24) for p in prompts]
        model = economize.apply(
            base,
            LookupConfig(max_ngram=3),
            TealConfig(sparsity=0.9, allocation="uniform"),
            decoding=DecodeConfig(cycle_length=4),
            calibration=calib,
        )
        try:
            for prompt, reference in zip(prompts, expected):
                assert torch.equal(model.decode(prompt, 24).tokens, reference)
        finally:
            model.detach()


# ---- calls are classified by the decoding phase, not by the shape of a map's input ----
def _moe_with_teal(prefill: str) -> tuple[nn.Module, object, TealConfig, object]:
    model = make_family_model("qwen2_moe")
    adapter = adapter_for(model)
    config = TealConfig(sparsity=0.5, allocation="uniform", prefill=prefill)
    calib = torch.randint(3, VOCAB, (4, 16), generator=torch.Generator().manual_seed(0))
    installed, _ = install_teal(adapter, calibrate_teal(adapter, config, token_batches(calib)).thresholds, config)
    return model, adapter, config, installed


@pytest.mark.parametrize("name", ["decode", "draft"])
def test_flattened_inputs_of_decoding_steps_are_sparsified(name: str) -> None:
    flat = torch.randn(3, 8, generator=torch.Generator().manual_seed(0))  # three rows' tokens, one per row
    with phase(name):
        assert torch.equal(sparsify_tokens(flat, 0.5, "dense"), sparsify(flat, 0.5))


def test_policy_from_config_strings() -> None:
    assert TealConfig().policy == PhasePolicy()
    assert TealConfig(prefill="all").policy == PhasePolicy(prefill="all", refresh="all", outside="all")
    custom = PhasePolicy(draft="dense")
    assert TealConfig(prefill=custom).policy is custom
    assert TealConfig(prefill={"refresh": "second_half"}).policy == PhasePolicy(refresh="second_half")
    assert TealConfig.from_dict(TealConfig(prefill=custom).to_dict()).policy == custom
    with pytest.raises(ConfigError):
        PhasePolicy(decode="half")
    with pytest.raises(ConfigError):
        TealConfig(prefill="half")


def test_policy_modes_by_phase() -> None:
    policy = PhasePolicy(prefill="second_half", refresh="dense", decode="all", draft="dense", outside="all")
    assert [policy.mode(p, 4) for p in ("prefill", "refresh", "decode", "draft")] == [
        "second_half", "dense", "all", "dense"
    ]
    assert policy.mode(None, 1) == "all" and policy.mode(None, 4) == "all"
    assert PhasePolicy(decode="dense").mode(None, 1) == "dense"  # a single-token call outside decoding is a step


def test_batched_decoding_step_sparsifies_the_moe_maps_with_flattened_inputs() -> None:
    model, adapter, config, installed = _moe_with_teal("dense")
    flat = [k for k in teal_modules(adapter, config) if "shared_expert" in k or k.endswith(".gate")]
    assert flat, "the tiny MoE has a router and a shared expert"
    step = torch.randint(3, VOCAB, (3, 1), generator=torch.Generator().manual_seed(1))
    try:
        with torch.no_grad():
            measured = measure_sparsity(adapter, config, lambda: model(input_ids=step))
            for key in flat:
                assert measured[key] > 0.2, key
            prompt = measure_sparsity(adapter, config, lambda: model(input_ids=step.repeat(1, 4)))
            assert all(prompt[key] == 0 for key in flat)
    finally:
        installed.remove()


def test_second_half_follows_sequence_positions_in_flattened_inputs() -> None:
    model, adapter, config, installed = _moe_with_teal("second_half")
    key = next(k for k in teal_modules(adapter, config) if "shared_expert." in k)
    module = teal_modules(adapter, config)[key]
    seen: list[torch.Tensor] = []
    handle = module.register_forward_pre_hook(lambda _m, args: seen.append(args[0].detach().clone()))
    ids = torch.randint(3, VOCAB, (2, 6), generator=torch.Generator().manual_seed(2))
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        handle.remove()
        installed.remove()
    (x,) = seen
    per_sequence = x.view(2, 6, -1)
    assert (per_sequence[:, :3] != 0).all()  # the first half of every sequence stays dense
    assert (per_sequence[:, 3:] == 0).float().mean() > 0.2


def test_summary_uses_block_sparsity(tiny2, calib, attach_context) -> None:
    from economize.techniques.sparsity import TealTechnique

    adapter = adapter_for(tiny2)
    technique = TealTechnique(TealConfig(sparsity=0.4), adapter)
    technique.attach(tiny2, adapter, attach_context(calibration=calib))
    try:
        sizes = {k: m.weight.numel() for k, m in teal_modules(adapter, technique.config).items()}
        assert technique.summary["model_sparsity"] == block_sparsity(technique.sparsity_map(), sizes)
    finally:
        technique.detach()


def test_loaded_thresholds_update_the_installed_hooks_in_place(tiny2, calib, attach_context) -> None:
    """A checkpoint loaded after attach changes the thresholds the hooks read, without reinstalling them when the
    same maps stay sparsified; a map whose threshold turns zero is unhooked."""
    from economize.techniques.sparsity import TealTechnique

    adapter = adapter_for(tiny2)
    technique = TealTechnique(TealConfig(sparsity=0.4), adapter)
    technique.attach(tiny2, adapter, attach_context(calibration=calib))
    try:
        installed, live = technique._installed, dict(technique._live)
        state = {k: v.clone() for k, v in technique.state_dict().items()}
        state["thresholds"] = state["thresholds"] * 2
        technique.load_state_dict(state)
        assert technique._installed is installed
        for key, tensor in live.items():
            assert technique._live[key] is tensor and float(tensor) == technique.threshold_map()[key]
        state["thresholds"][0] = 0.0
        technique.load_state_dict(state)
        assert technique.keys[0] not in technique._live and len(technique._live) == len(live) - 1
    finally:
        technique.detach()
