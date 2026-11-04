"""Paper-fidelity and extension points of the dynamic-depth techniques.

* LoRA-Drop reuses the output of the cycle's full token at a dropped layer; anchors of rejected tokens are discarded.
* Mixture-of-Depths reads the raw stream by default and rejects ``kv="routed"`` until the runtime supports it.
* FlexiDepth routers come from a registry; the paper's router is the default.
* CALM's measures declare their modules through a hook, and Learn then Test certifies a threshold.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

import economize
from economize.errors import ConfigError, IncompatibleTechniquesError
from economize.techniques.dynamic_depth import (
    CONFIDENCE_MEASURES,
    FLEXI_ROUTERS,
    AnchorStore,
    CalmConfig,
    CalmObjective,
    ConfidenceMeasure,
    FlexiDepthConfig,
    FlexiRouter,
    MoDConfig,
    hoeffding_bentkus_pvalue,
    ltt_threshold,
    runtime_supports_routed_kv,
    token_mismatch,
)
from helpers import make_tiny_model


# ---- LoRA-Drop anchors --------------------------------------------------------------------------------------------
def _pass(store: AnchorStore, positions: list[int], full: list[bool]) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.tensor(positions, dtype=torch.float32)[None, :, None].expand(1, -1, 2) + 100.0
    pos = torch.tensor([positions])
    return store.lookup(out, torch.tensor([full]), pos, torch.ones_like(pos, dtype=torch.bool))


def test_anchor_store_reuses_the_last_full_token_and_forgets_rejected_ones() -> None:
    store = AnchorStore()
    anchor, found = _pass(store, [0], [True])
    assert found.all() and anchor[0, 0, 0] == 100
    anchor, found = _pass(store, [1], [False])
    assert found.all() and anchor[0, 0, 0] == 100  # the anchor of position 1 is position 0
    anchor, found = _pass(store, [2, 3, 4], [True, False, True])  # a verification pass
    assert anchor[0, :, 0].tolist() == [102, 102, 104]
    anchor, _ = _pass(store, [3, 4], [False, True])  # 3 and 4 were rejected and are recomputed
    assert anchor[0, :, 0].tolist() == [102, 104]
    anchor, _ = _pass(store, [5], [False])
    assert anchor[0, 0, 0] == 104
    assert store.positions is not None and store.positions.shape[1] <= 2  # pruned to what a later pass can use


def test_anchor_store_falls_back_without_an_anchor() -> None:
    _, found = _pass(AnchorStore(), [1, 2], [False, True])
    assert found.tolist() == [[False, True]]


def test_loradrop_dropped_token_is_the_anchor_output_plus_the_update() -> None:
    from economize.techniques.dynamic_depth import LoRADropConfig

    model = economize.apply(make_tiny_model(), LoRADropConfig(period=4, layers=(5,)))
    technique = model.techniques["loradrop"]
    policy = technique.layer_policy()
    policy._full = torch.tensor([[True, False, False]])
    policy._positions = torch.tensor([[0, 1, 2]])
    x, out = torch.randn(1, 3, 64), torch.randn(1, 3, 64)

    class Ctx:
        valid = None

    result = policy.transform(5, x, out, None, Ctx())
    assert torch.equal(result[0, 0], out[0, 0])
    expected = out[0, 0] + technique.update(5, x[0, 1:])
    torch.testing.assert_close(result[0, 1:], expected)


# ---- Mixture-of-Depths --------------------------------------------------------------------------------------------
def test_mod_defaults_follow_the_paper_and_routed_kv_needs_the_runtime() -> None:
    config = MoDConfig()
    assert config.router_input == "raw" and config.kv == "fill"
    model = economize.apply(make_tiny_model(), config)
    router = next(iter(model.techniques["mod"].routers.values()))
    assert router.normalize is False
    if not runtime_supports_routed_kv():
        with pytest.raises(ConfigError):
            economize.apply(make_tiny_model(), MoDConfig(kv="routed"))
    with pytest.raises(ConfigError):
        MoDConfig(router_input="layernorm")


# ---- FlexiDepth routers -------------------------------------------------------------------------------------------
def test_flexidepth_router_registry() -> None:
    assert {"paper", "bottleneck"} <= set(FLEXI_ROUTERS.names())
    model = economize.apply(make_tiny_model(), FlexiDepthConfig())
    routers = model.techniques["flexidepth"].routers
    assert all(isinstance(r, FlexiRouter) for r in routers.values())
    bottleneck = economize.apply(make_tiny_model(), FlexiDepthConfig(router="bottleneck"))
    assert not any(isinstance(r, FlexiRouter) for r in bottleneck.techniques["flexidepth"].routers.values())
    with pytest.raises(ConfigError):
        FlexiDepthConfig(router="unknown")


# ---- CALM ---------------------------------------------------------------------------------------------------------
def test_calm_measure_modules_are_registered_through_the_hook() -> None:
    @CONFIDENCE_MEASURES.register("test-bias")
    class BiasConfidence(ConfidenceMeasure):
        trainable = True

        @classmethod
        def modules(cls, technique):  # type: ignore[no-untyped-def]
            return {"bias_head": nn.Linear(technique.adapter.hidden_size, 1)}

        def logit(self, index, hidden):  # type: ignore[no-untyped-def]
            return self.technique.bias_head(hidden.float())[..., 0]

        def __call__(self, index, x, out):  # type: ignore[no-untyped-def]
            return torch.sigmoid(self.logit(index, out))

    try:
        model = economize.apply(make_tiny_model(), CalmConfig(confidence="test-bias"))
        technique = model.techniques["calm"]
        assert any(k.startswith("bias_head.") for k in technique.state_dict())
        assert CalmObjective().parameter_groups(model)[0].params
    finally:
        CONFIDENCE_MEASURES.unregister("test-bias")
    softmax = economize.apply(make_tiny_model(), CalmConfig(confidence="softmax"))
    with pytest.raises(IncompatibleTechniquesError):
        CalmObjective().parameter_groups(softmax)
    classifier = economize.apply(make_tiny_model(), CalmConfig(confidence="classifier"))
    assert set(classifier.techniques["calm"].state_dict()) == {"classifier.proj.weight", "classifier.proj.bias"}


def test_hoeffding_bentkus_pvalue() -> None:
    assert hoeffding_bentkus_pvalue(0.0, 200, 0.1) < 1e-6
    assert hoeffding_bentkus_pvalue(0.2, 200, 0.1) == 1.0 or hoeffding_bentkus_pvalue(0.2, 200, 0.1) > 0.5
    # monotone in the mean loss and in n
    assert hoeffding_bentkus_pvalue(0.02, 100, 0.1) < hoeffding_bentkus_pvalue(0.05, 100, 0.1)
    assert hoeffding_bentkus_pvalue(0.05, 400, 0.1) < hoeffding_bentkus_pvalue(0.05, 100, 0.1)
    assert math.isclose(hoeffding_bentkus_pvalue(0.0, 10, 0.5), min(0.5**10, math.e * 0.5**10))
    with pytest.raises(ConfigError):
        hoeffding_bentkus_pvalue(0.0, 10, 1.0)


def test_ltt_threshold_uses_fixed_sequence_testing() -> None:
    candidates = [0.9, 0.7, 0.5, 0.3]
    n = 300
    losses = [[0.0] * n, [0.01] * n, [0.3] * n, [0.0] * n]  # 0.3 is safe again but follows a failure
    assert ltt_threshold(candidates, losses, delta=0.1, epsilon=0.1) == 0.7
    assert ltt_threshold(candidates, [[0.5] * n] * 4, delta=0.1, epsilon=0.1) == 1.0
    with pytest.raises(ConfigError):
        ltt_threshold(candidates, [[2.0] * n] * 4, delta=0.1, epsilon=0.1)


def test_token_mismatch() -> None:
    assert token_mismatch(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])) == 0.0
    assert token_mismatch(torch.tensor([1, 5, 3]), torch.tensor([1, 2, 3, 4])) == 0.5


def test_calibrate_threshold_restores_the_config() -> None:
    from economize.techniques.dynamic_depth import calibrate_threshold

    model = economize.apply(make_tiny_model(), CalmConfig(confidence="state", threshold=0.5))
    prompts = [torch.randint(1, 50, (8,), generator=torch.Generator().manual_seed(i)) for i in range(3)]
    chosen = calibrate_threshold(model, prompts, 6, candidates=(0.999, 0.5), delta=0.5, epsilon=0.5)
    assert chosen in (0.999, 0.5, 1.0)
    assert model.techniques["calm"].config.threshold == 0.5
