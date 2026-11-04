"""CATS and Deja Vu: gated-MLP channel sparsity against references written from the papers, the attach/detach and
state round trips, the predictor fit and objective, the serving refusals and the row-masked Triton kernel (in the
Triton interpreter on the CPU, and on a GPU when one is present)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

import economize
from economize.errors import ConfigError, IncompatibleTechniquesError, UnsupportedModelError
from economize.techniques.mlp_sparsity import (
    MLP_SPARSITY_BACKENDS,
    CatsConfig,
    CatsTechnique,
    DejaVuConfig,
    DejaVuObjective,
    DejaVuTechnique,
    DenseMlpBackend,
    DenseOps,
    TritonMlpBackend,
    cats_mlp,
    dejavu_labels,
    dejavu_mlp,
    gated_mlps,
    kept_channels,
    predictor_recall,
    topk_mask,
)
from helpers import VOCAB, make_tiny_model

ROOT = Path(__file__).resolve().parents[2]
FAST_DEJAVU = dict(rank=16, fit_steps=60, fit_batch=128, fit_lr=1e-2)


def _gated(model: torch.nn.Module, layer: int = 1):
    adapter = economize.arch.adapter_for(model)
    return gated_mlps(adapter, CatsConfig())[layer]


def _hidden(model: torch.nn.Module, rows: int = 12, seed: int = 0) -> Tensor:
    d = model.config.hidden_size
    return torch.randn(rows, d, generator=torch.Generator().manual_seed(seed))


# ---- configuration -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [dict(sparsity=1.5), dict(sparsity=-0.1), dict(backend="nope"), dict(prefill="sometimes"), dict(mlp=""),
     dict(layers=(0,))],
)
def test_shared_config_validation(kwargs: dict) -> None:
    for config in (CatsConfig, DejaVuConfig):
        with pytest.raises(ConfigError):
            config(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [dict(rank=0), dict(predictor_input="attention"), dict(predictor_dtype="int8"), dict(fit_steps=-1),
     dict(fit_lr=0.0), dict(fit_batch=0), dict(max_tokens=0), dict(seed=-1), dict(lookahead_norm="")],
)
def test_dejavu_config_validation(kwargs: dict) -> None:
    with pytest.raises(ConfigError):
        DejaVuConfig(**kwargs)


def test_config_rejects_missing_layer(tiny2, adapter_of) -> None:
    with pytest.raises(ConfigError):
        CatsConfig(layers=(3,)).validate(adapter_of(tiny2))


def test_mixture_of_experts_or_fused_mlps_are_refused(tiny2, adapter_of) -> None:
    with pytest.raises(UnsupportedModelError, match="gated MLP"):
        CatsTechnique(CatsConfig(up="fused_gate_up"), adapter_of(tiny2))


def test_metadata_and_registration() -> None:
    assert MLP_SPARSITY_BACKENDS.get("dense") is DenseMlpBackend
    assert MLP_SPARSITY_BACKENDS.get("triton") is TritonMlpBackend
    assert DenseMlpBackend.simulation is not None and TritonMlpBackend.simulation is None
    assert CatsTechnique.info.trainable == () and DejaVuTechnique.info.trainable == ("predictors",)
    assert not CatsTechnique.info.lossless and not DejaVuTechnique.info.lossless
    assert economize.techniques.TECHNIQUES.get("cats") is CatsTechnique
    assert economize.techniques.TECHNIQUES.get("dejavu") is DejaVuTechnique


# ---- the MLP forwards against the papers ---------------------------------------------------------------------------


def test_cats_mlp_matches_equation_two(tiny2) -> None:
    """CATS Eq. 2: ``down((g * 1[|g| > t]) * up(x))`` with ``g = act(gate(x))``."""
    mlp = _gated(tiny2)
    x = _hidden(tiny2)
    g = mlp.act(mlp.gate(x))
    t = torch.quantile(g.abs().flatten(), 0.5)
    reference = mlp.down(torch.where(g.abs() > t, g, torch.zeros_like(g)) * mlp.up(x))
    out = cats_mlp(mlp, x, DenseOps(), "all", t)
    torch.testing.assert_close(out, reference, rtol=0, atol=1e-6)
    assert torch.equal(cats_mlp(mlp, x, DenseOps(), "all", torch.tensor(-1.0)), mlp.module(x))


def test_dejavu_mlp_computes_only_the_topk_channels(tiny2) -> None:
    mlp = _gated(tiny2)
    x = _hidden(tiny2)
    ffn = mlp.up.out_features
    scores = torch.randn(x.shape[0], ffn, generator=torch.Generator().manual_seed(3))
    k = kept_channels(ffn, 0.75)
    keep = scores.topk(k, -1).indices
    reference = torch.stack(
        [mlp.down.weight[:, keep[i]] @ (mlp.act(mlp.gate.weight[keep[i]] @ x[i]) * (mlp.up.weight[keep[i]] @ x[i]))
         for i in range(x.shape[0])]
    )
    torch.testing.assert_close(dejavu_mlp(mlp, x, DenseOps(), "all", scores, k), reference, rtol=1e-5, atol=1e-5)
    assert torch.allclose(dejavu_mlp(mlp, x, DenseOps(), "all", scores, ffn), mlp.module(x), atol=1e-6)


def test_labels_are_the_largest_contributions(tiny2) -> None:
    mlp = _gated(tiny2)
    x = _hidden(tiny2)
    labels = dejavu_labels(mlp, x, 5)
    contribution = (mlp.act(mlp.gate(x)) * mlp.up(x)).abs()
    assert labels.dtype == torch.bool and labels.sum(-1).eq(5).all()
    smallest_kept = contribution[labels].reshape(-1, 5).min(-1).values
    assert (smallest_kept >= contribution[~labels].reshape(x.shape[0], -1).max(-1).values).all()
    assert predictor_recall(contribution, labels, 5) == 1.0
    assert kept_channels(64, 0.5) == 32 and kept_channels(64, 1.0) == 1


def test_labels_do_not_depend_on_the_chunk_size(tiny2) -> None:
    mlp = _gated(tiny2)
    x = _hidden(tiny2)
    labels = dejavu_labels(mlp, x, 5)
    for chunk in (1, 3, x.shape[0]):
        assert torch.equal(dejavu_labels(mlp, x, 5, chunk=chunk), labels)
    batched = dejavu_labels(mlp, x.reshape(1, *x.shape), 5, chunk=2)
    assert batched.shape == (1, *labels.shape) and torch.equal(batched[0], labels)


def test_topk_mask() -> None:
    mask = topk_mask(torch.tensor([[0.1, 3.0, -1.0, 2.0]]), 2)
    assert mask.tolist() == [[False, True, False, True]]


# ---- CATS technique ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("target", [0.3, 0.5, 0.7])
def test_cats_threshold_drops_the_target_fraction(tiny2, adapter_of, calib, attach_context, target: float) -> None:
    adapter = adapter_of(tiny2)
    technique = CatsTechnique(CatsConfig(sparsity=target, prefill="all"), adapter)
    technique.attach(tiny2, adapter, attach_context(calib))
    seen: list[float] = []
    for index, t in technique.threshold_map().items():
        mlp = _gated(tiny2, index)
        captured: list[Tensor] = []
        handle = mlp.module.register_forward_pre_hook(lambda _m, a, captured=captured: captured.append(a[0]))
        tiny2(calib)
        handle.remove()
        g = mlp.act(mlp.gate(captured[0]))
        seen.append(float((g.abs() <= t).float().mean()))
    technique.detach()
    # layer 1 sees the calibration distribution exactly; deeper layers see sparsified inputs
    assert seen[0] == pytest.approx(target, abs=0.01)
    assert all(abs(s - target) < 0.08 for s in seen)


def test_cats_attach_detach_and_state_round_trip(tiny2, adapter_of, calib, held_out, attach_context) -> None:
    adapter = adapter_of(tiny2)
    config = CatsConfig(sparsity=0.5, prefill="all")
    reference = tiny2(held_out).logits
    technique = CatsTechnique(config, adapter)
    assert not technique.calibrated
    technique.attach(tiny2, adapter, attach_context(calib))
    assert technique.calibrated and technique.summary["installed"]
    sparse = tiny2(held_out).logits
    assert not torch.equal(sparse, reference)
    state = {k: v.clone() for k, v in technique.state_dict().items()}
    assert set(state) == {"thresholds"} and state["thresholds"].shape == (2,)
    technique.detach()
    assert torch.equal(tiny2(held_out).logits, reference)

    restored = CatsTechnique(config, adapter)
    restored.load_state_dict(state, strict=True)
    restored.attach(tiny2, adapter, None)  # loaded thresholds: no calibration data needed
    assert torch.equal(tiny2(held_out).logits, sparse)
    restored.detach()


def test_cats_loading_after_attach_updates_the_thresholds(tiny2, adapter_of, calib, held_out, attach_context) -> None:
    adapter = adapter_of(tiny2)
    first = CatsTechnique(CatsConfig(sparsity=0.6, prefill="all"), adapter)
    first.attach(tiny2, adapter, attach_context(calib))
    target = tiny2(held_out).logits
    state = first.state_dict()
    first.detach()
    second = CatsTechnique(CatsConfig(sparsity=0.2, prefill="all"), adapter)
    second.attach(tiny2, adapter, attach_context(calib))
    second.load_state_dict(state)
    assert torch.equal(tiny2(held_out).logits, target)
    second.detach()


def test_cats_needs_calibration(tiny2, adapter_of, attach_context) -> None:
    adapter = adapter_of(tiny2)
    with pytest.raises(ConfigError, match="calibration"):
        CatsTechnique(CatsConfig(), adapter).attach(tiny2, adapter, attach_context(None))


def test_dense_prefill_keeps_multi_token_calls_exact(tiny2, calib, held_out) -> None:
    reference = tiny2(held_out).logits
    model = economize.apply(tiny2, CatsConfig(sparsity=0.5), calibration=[calib])
    assert torch.equal(model(held_out).logits, reference)  # default prefill="dense": a forward pass outside decoding
    model.detach()
    assert torch.equal(tiny2(held_out).logits, reference)


def test_cats_and_dejavu_conflict_with_teal() -> None:
    from economize.techniques.sparsity import TealConfig

    for config in (CatsConfig(), DejaVuConfig()):
        with pytest.raises(IncompatibleTechniquesError):
            economize.apply(make_tiny_model(num_layers=2), config, TealConfig())
    with pytest.raises(IncompatibleTechniquesError):
        economize.apply(make_tiny_model(num_layers=2), CatsConfig(), DejaVuConfig())


def test_cats_composes_with_weight_quantization(calib, held_out) -> None:
    from economize.techniques.prune import PruneConfig

    base = make_tiny_model(num_layers=2)
    model = economize.apply(base, PruneConfig(sparsity=0.3), CatsConfig(sparsity=0.5, prefill="all"),
                            calibration=[calib])
    logits = model(held_out).logits
    assert torch.isfinite(logits).all()


# ---- Deja Vu technique ---------------------------------------------------------------------------------------------


def _fitted(model, adapter, calib, attach_context, **kwargs) -> DejaVuTechnique:
    config = DejaVuConfig(sparsity=0.5, prefill="all", **{**FAST_DEJAVU, **kwargs})
    technique = DejaVuTechnique(config, adapter)
    technique.attach(model, adapter, attach_context(calib))
    return technique


def _recall(technique: DejaVuTechnique, model, ids: Tensor) -> float:
    """Mean recall of the predictors on ``ids`` (the dense model's inputs of every MLP)."""
    technique.detach()
    recalls = []
    for layer in technique.layers:
        mlp = _gated(model, layer)
        captured: list[Tensor] = []
        handle = mlp.module.register_forward_pre_hook(lambda _m, a, captured=captured: captured.append(a[0]))
        model(ids)
        handle.remove()
        x = captured[0].reshape(-1, captured[0].shape[-1])
        labels = dejavu_labels(mlp, x, technique.kept(layer))
        recalls.append(float(predictor_recall(technique.scores(layer, x), labels, technique.kept(layer))))
    technique.install()
    return sum(recalls) / len(recalls)


def test_dejavu_fit_beats_a_random_predictor(tiny2, adapter_of, calib, held_out, attach_context) -> None:
    adapter = adapter_of(tiny2)
    untrained = DejaVuTechnique(DejaVuConfig(sparsity=0.5, prefill="all", **{**FAST_DEJAVU, "fit_steps": 0}), adapter)
    untrained.attach(tiny2, adapter, attach_context(None))  # no fit: no calibration needed
    chance = _recall(untrained, tiny2, held_out)
    untrained.detach()
    fitted = _fitted(tiny2, adapter, calib, attach_context)
    assert bool(fitted.fitted)
    assert chance == pytest.approx(0.5, abs=0.1)
    assert _recall(fitted, tiny2, held_out) > chance + 0.1
    fitted.detach()


@pytest.mark.parametrize("prefill", ["all", "second_half"])
def test_dejavu_long_prefill_in_pieces_matches_one_piece(
    tiny2, adapter_of, held_out, attach_context, monkeypatch, prefill
) -> None:
    adapter = adapter_of(tiny2)
    config = DejaVuConfig(sparsity=0.5, prefill=prefill, **{**FAST_DEJAVU, "fit_steps": 0})
    technique = DejaVuTechnique(config, adapter)
    technique.attach(tiny2, adapter, attach_context(None))
    whole = tiny2(held_out).logits
    monkeypatch.setattr("economize.techniques.mlp_sparsity.dejavu.PREFILL_CHUNK", 5)  # 48 tokens: 10 pieces
    torch.testing.assert_close(tiny2(held_out).logits, whole, rtol=0, atol=1e-5)
    technique.detach()


def test_dejavu_attach_detach_and_state_round_trip(tiny2, adapter_of, calib, held_out, attach_context) -> None:
    adapter = adapter_of(tiny2)
    reference = tiny2(held_out).logits
    technique = _fitted(tiny2, adapter, calib, attach_context)
    sparse = tiny2(held_out).logits
    assert not torch.equal(sparse, reference) and technique.summary["installed"]
    state = {k: v.clone() for k, v in technique.state_dict().items()}
    assert "fitted" in state and {f"predictors.{i}.0.weight" for i in ("1", "2")} <= set(state)
    technique.detach()
    assert torch.equal(tiny2(held_out).logits, reference)

    restored = DejaVuTechnique(technique.config, adapter)
    restored.load_state_dict(state, strict=True)
    restored.attach(tiny2, adapter, None)  # loaded predictors: no fit, no calibration data
    assert torch.equal(tiny2(held_out).logits, sparse)
    restored.detach()


@pytest.mark.parametrize("predictor_input", ["mlp", "lookahead"])
def test_dejavu_offloaded_fit_matches_the_on_device_fit(
    tiny2, adapter_of, calib, attach_context, predictor_input
) -> None:
    adapter = adapter_of(tiny2)
    states = []
    for offload in (False, True):
        technique = _fitted(tiny2, adapter, calib, attach_context, offload=offload, predictor_input=predictor_input)
        states.append(technique.state_dict())
        technique.detach()
    assert states[0].keys() == states[1].keys()
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key]), key


def test_dejavu_lookahead_reads_the_first_norm(tiny2, adapter_of, calib, held_out, attach_context) -> None:
    adapter = adapter_of(tiny2)
    technique = _fitted(tiny2, adapter, calib, attach_context, predictor_input="lookahead")
    logits = tiny2(held_out).logits
    assert torch.isfinite(logits).all()
    technique.detach()
    missing = DejaVuTechnique(DejaVuConfig(predictor_input="lookahead", lookahead_norm="nope", fit_steps=0), adapter)
    with pytest.raises(UnsupportedModelError, match="nope"):
        missing.attach(tiny2, adapter, None)
    missing.detach()


def test_dejavu_low_precision_predictors(tiny2, adapter_of, calib, attach_context) -> None:
    adapter = adapter_of(tiny2)
    technique = _fitted(tiny2, adapter, calib, attach_context, predictor_dtype="bfloat16", fit_steps=5)
    assert technique.predictor(1)[0].weight.dtype == torch.bfloat16
    technique.detach()


def test_dejavu_objective_trains_only_the_predictors(calib) -> None:
    from economize.train import fit
    from economize.train.objective import OBJECTIVES

    base = make_tiny_model(num_layers=2)
    base_before = {k: v.clone() for k, v in base.state_dict().items()}
    config = DejaVuConfig(sparsity=0.5, prefill="all", rank=16, fit_steps=0)
    model = economize.apply(base, config)
    technique = model.techniques["dejavu"]
    assert OBJECTIVES.get("dejavu") is DejaVuObjective
    rows = torch.randint(0, VOCAB, (16, 24), generator=torch.Generator().manual_seed(0))
    with torch.enable_grad():
        result = fit(model, rows, DejaVuObjective(), steps=12, batch_size=4, lr=1e-2, warmup_steps=0, seed=0,
                     log_every=1)
    assert result.losses[-1] < result.losses[0]
    assert result.history and "recall" in result.history[-1]
    assert bool(technique.fitted)
    assert all(torch.equal(base_before[k], v) for k, v in base.state_dict().items())


def test_dejavu_objective_needs_the_installed_technique() -> None:
    from economize.train.objective import Batch

    model = economize.apply(make_tiny_model(num_layers=2), DejaVuConfig(rank=4, fit_steps=0))
    technique = model.techniques["dejavu"]
    technique.detach()
    with pytest.raises(UnsupportedModelError, match="installed"):
        DejaVuObjective().loss(model, Batch(torch.zeros(1, 4, dtype=torch.long)), None)


# ---- serving -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("config", [CatsConfig(), DejaVuConfig()], ids=["cats", "dejavu"])
def test_serving_adapters_refuse_honestly(config) -> None:
    from economize.backends.sglang.serving.base import SGLANG_TECHNIQUES
    from economize.backends.vllm.serving.base import VLLM_TECHNIQUES, Support
    from economize.core.compatibility import SupportStatus

    for registry in (VLLM_TECHNIQUES, SGLANG_TECHNIQUES):
        adapter = registry.get(config.technique)
        assert adapter.support is Support.UNSUPPORTED or adapter.support.name == "UNSUPPORTED"
        assert "HF-only" in adapter.reason
    rows = economize.backend_support(config)
    assert rows["vllm"].status is SupportStatus.UNSUPPORTED and rows["sglang"].status is SupportStatus.UNSUPPORTED
    assert rows["transformers"].status is not SupportStatus.UNSUPPORTED


# ---- Triton kernels ------------------------------------------------------------------------------------------------

_INTERPRETED = """
import torch
from economize.techniques.mlp_sparsity import row_masked_gemv
from economize.techniques.mlp_sparsity.kernel import RowGemvConfig
torch.manual_seed(0)
for dtype in (torch.float32, torch.float16):
    for m, n, k in ((1, 40, 96), (3, 72, 64), (16, 33, 80)):
        x = torch.randn(m, k).to(dtype)
        w = torch.randn(n, k).to(dtype)
        mask = torch.rand(m, n) < 0.4
        for cfg in (None, RowGemvConfig(block_n=16, block_k=32)):
            out = row_masked_gemv(x, w, mask, cfg)
            ref = (x.float() @ w.float().t()) * mask
            assert out.dtype == dtype and out.shape == (m, n)
            assert bool((out[~mask] == 0).all()), "dropped outputs must be exact zeros"
            tol = 1e-4 if dtype == torch.float32 else 2e-2
            torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol * k ** 0.5)
try:
    row_masked_gemv(torch.randn(17, 8), torch.randn(4, 8), torch.ones(17, 4, dtype=torch.bool))
except ValueError:
    pass
else:
    raise AssertionError("17 rows must be refused")
print("ok")
"""

_INTERPRETED_MLP = """
import torch
from economize.techniques.mlp_sparsity import (TritonMlpBackend, CatsConfig, DenseOps, cats_mlp, dejavu_mlp,
    gated_mlps, kept_channels)
import economize
from economize.techniques.sparsity.teal_gemv import GemvConfig
from helpers import make_tiny_model

class Interpreted(TritonMlpBackend):
    gemv_config = GemvConfig(block_n=32, block_k=32)  # the default heuristic asks the GPU for its SM count
    @classmethod
    def usable(cls):
        return True
    @classmethod
    def eligible(cls, module):
        return isinstance(module, torch.nn.Linear) and module.weight.dtype in (torch.float32, torch.float16)

torch.manual_seed(0)
model = make_tiny_model(num_layers=2)
mlp = gated_mlps(economize.arch.adapter_for(model), CatsConfig())[1]
x = torch.randn(3, model.config.hidden_size)
with torch.no_grad():
    ops = Interpreted().ops(mlp)
    assert not mlp.down.weight.is_contiguous() and mlp.down.weight.t().is_contiguous()
    g = mlp.act(mlp.gate(x))
    t = g.abs().median()
    torch.testing.assert_close(cats_mlp(mlp, x, ops, "all", t), cats_mlp(mlp, x, DenseOps(), "all", t),
                               rtol=1e-4, atol=1e-5)
    scores = torch.randn(3, mlp.up.out_features)
    k = kept_channels(mlp.up.out_features, 0.5)
    expected = dejavu_mlp(mlp, x, DenseOps(), "all", scores, k)
    torch.testing.assert_close(dejavu_mlp(mlp, x, ops, "all", scores, k), expected, rtol=1e-4, atol=1e-5)
    ops.remove()
    assert mlp.down.weight.is_contiguous()
print("ok")
"""


def _interpret(source: str) -> None:
    if importlib.util.find_spec("triton") is None:
        pytest.skip("needs Triton")
    path = os.pathsep.join([str(ROOT / "src"), str(ROOT / "tests")])
    env = {**os.environ, "TRITON_INTERPRET": "1", "PYTHONPATH": path}
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(source)], env=env, capture_output=True, text=True,
                            timeout=600, cwd=ROOT)
    assert result.returncode == 0 and result.stdout.strip().endswith("ok"), result.stderr[-4000:]


def test_row_masked_gemv_in_the_interpreter() -> None:
    _interpret(_INTERPRETED)


def test_triton_ops_match_dense_ops_in_the_interpreter() -> None:
    _interpret(_INTERPRETED_MLP)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("m", [1, 2, 9, 16])
def test_row_masked_gemv_bf16_on_gemma4_31b_shapes(m: int) -> None:
    """bfloat16 on the Gemma-4-31B up map ``[21504, 5376]``: within the rounding of cuBLAS on the masked output."""
    from economize.techniques.mlp_sparsity import row_masked_gemv

    generator = torch.Generator("cuda").manual_seed(0)
    weight = torch.randn(21504, 5376, device="cuda", dtype=torch.bfloat16, generator=generator)
    x = torch.randn(m, 5376, device="cuda", dtype=torch.bfloat16, generator=generator)
    mask = torch.rand(m, 21504, device="cuda", generator=generator) < 0.5
    out = row_masked_gemv(x, weight, mask)
    reference = (x.float() @ weight.float().t()) * mask
    cublas = F.linear(x, weight) * mask
    error = float((out.float() - reference).norm() / reference.norm())
    assert (out[~mask] == 0).all()
    assert error < 2 * float((cublas.float() - reference).norm() / reference.norm()) + 1e-3


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("config", [CatsConfig, DejaVuConfig])
def test_triton_backend_decodes_like_the_dense_backend(config) -> None:
    """Greedy decoding with the Triton backend matches the dense backend (bfloat16 noise aside) on a tiny model."""
    calib = torch.randint(0, VOCAB, (8, 32), generator=torch.Generator().manual_seed(0))
    prompt = torch.randint(0, VOCAB, (1, 8), generator=torch.Generator().manual_seed(1)).cuda()
    extra = dict(rank=16, fit_steps=20) if config is DejaVuConfig else {}
    outputs = []
    for backend in ("dense", "triton"):
        base = make_tiny_model(num_layers=2).to("cuda", torch.bfloat16)
        model = economize.apply(base, config(sparsity=0.5, backend=backend, **extra), calibration=[calib])
        outputs.append(model.generate(prompt, max_new_tokens=16, do_sample=False))
    assert (outputs[0] == outputs[1]).float().mean() > 0.8
