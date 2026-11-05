"""The default backend ``"auto"`` of TEAL, CATS and Deja Vu: the reported support, the refusal of maps the kernels
cannot serve, the simulation report of the lenient ``"triton"`` backends, and (on a GPU) that the default runs the
sparse kernels with tuned launch parameters and matches the masked dense reference."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import economize
from economize import SupportStatus
from economize.errors import ConfigError
from economize.techniques.mlp_sparsity import (
    MLP_SPARSITY_BACKENDS,
    AutoMlpBackend,
    CatsConfig,
    DejaVuConfig,
    GatedMlp,
    KernelPlan,
    RowGemvConfig,
    RowTuner,
    TritonMlpBackend,
    clear_row_tuning_cache,
    masked,
    row_masked_gemv,
)
from economize.techniques.sparsity import TEAL_BACKENDS, AutoBackend, TealConfig
from economize.techniques.sparsity.auto_backend import describe_problems, kernel_problems
from economize.techniques.sparsity.teal_gemv import GemvConfig, triton_available
from economize.techniques.sparsity.triton_backend import kernel_ineligibility
from helpers import VOCAB, make_tiny_model

CONFIGS = [TealConfig, CatsConfig, DejaVuConfig]
IDS = ["teal", "cats", "dejavu"]
FAST = {
    TealConfig: dict(layers=(1,), greedy_step=0.2, greedy_batches=2),
    CatsConfig: {},
    DejaVuConfig: dict(rank=16, fit_steps=20),
}


def _calibration() -> list[torch.Tensor]:
    return [torch.randint(0, VOCAB, (4, 32), generator=torch.Generator().manual_seed(0))]


# ---- reported support ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
def test_default_configs_report_no_simulated_path(config) -> None:
    assert config().backend == "auto"
    rows = economize.backend_support(config())
    assert rows["transformers"].status is SupportStatus.REAL
    assert all(row.status is not SupportStatus.SIMULATED for row in rows.values()), rows
    for row in rows.values():
        assert row.status is SupportStatus.REAL or row.detail, row  # every refusal names its reason


@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
def test_the_dense_reference_reports_its_saving_as_simulated(config) -> None:
    support = economize.backend_support(config(backend="dense"))["transformers"]
    assert support.status is SupportStatus.SIMULATED and support.simulation is not None


def test_auto_backends_are_registered_without_a_simulation() -> None:
    assert TEAL_BACKENDS.get("auto") is AutoBackend and MLP_SPARSITY_BACKENDS.get("auto") is AutoMlpBackend
    assert getattr(AutoBackend, "simulation", None) is None and getattr(AutoMlpBackend, "simulation", None) is None


# ---- refusals ------------------------------------------------------------------------------------------------------


def test_kernel_ineligibility_names_the_reason() -> None:
    assert kernel_ineligibility(nn.Linear(4, 4)) == "a weight on cpu"
    assert "not an nn.Linear" in str(kernel_ineligibility(nn.Conv1d(4, 4, 1)))
    if torch.cuda.is_available():
        assert "float32" in str(kernel_ineligibility(nn.Linear(4, 4, device="cuda")))
        assert kernel_ineligibility(nn.Linear(4, 4, device="cuda", dtype=torch.bfloat16)) is None


def test_describe_problems_summarizes_long_lists() -> None:
    problems = {f"map {i}": "a weight on cpu" for i in range(6)}
    text = describe_problems(problems)
    assert text.count("a weight on cpu") == 4 and text.endswith("and 2 more")
    assert describe_problems({"map": "x"}) == "map: x"


def test_kernel_problems_without_triton_refuse_every_map(monkeypatch: pytest.MonkeyPatch) -> None:
    from economize.techniques.sparsity import auto_backend

    monkeypatch.setattr(auto_backend, "triton_available", lambda: False)
    assert kernel_problems({"a": nn.Linear(2, 2), "b": nn.Linear(2, 2)}) == dict.fromkeys(
        ["a", "b"], "Triton or a CUDA device is not available"
    )


@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
def test_auto_refuses_a_cpu_model_before_calibrating(config) -> None:
    with pytest.raises(ConfigError, match="backend='dense'"):
        economize.apply(make_tiny_model(num_layers=2), config(**FAST[config]), calibration=_calibration())


@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
def test_lenient_triton_on_a_cpu_model_reports_a_simulation(config) -> None:
    """``backend="triton"`` falls back to masking the maps the kernels cannot serve and says so."""
    model = economize.apply(
        make_tiny_model(num_layers=2), config(backend="triton", **FAST[config]), calibration=_calibration()
    )
    simulation = model.simulations()[config().technique]
    assert "cannot serve" in simulation.reason


@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
def test_dense_reference_runs_on_a_cpu_model(config) -> None:
    model = economize.apply(
        make_tiny_model(num_layers=2), config(backend="dense", **FAST[config]), calibration=_calibration()
    )
    assert config().technique in model.simulations()


# ---- GPU: the default runs the kernels -----------------------------------------------------------------------------

_kernel = pytest.mark.skipif(not triton_available(), reason="needs Triton and a CUDA GPU")


def _count_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Wrap the kernel entry points where the backends call them and count the launches."""
    from economize.techniques.mlp_sparsity import backends as mlp_backends
    from economize.techniques.sparsity import triton_backend

    counts = {"teal_gemv": 0, "row_masked_gemv": 0}

    def counting(module: object, name: str) -> None:
        original = getattr(module, name)

        def wrapper(*args, **kwargs):
            counts[name] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapper)

    counting(triton_backend, "teal_gemv")
    counting(mlp_backends, "teal_gemv")
    counting(mlp_backends, "row_masked_gemv")
    return counts


@pytest.mark.gpu
@_kernel
@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
def test_default_decodes_with_the_kernels_like_the_dense_reference(config, monkeypatch: pytest.MonkeyPatch) -> None:
    """Greedy bfloat16 decoding: the default launches the sparse kernels at every step, reports no simulation and
    matches the dense reference on the step logits within bfloat16 rounding."""
    extra = {**FAST[config], "layers": (1, 2)} if config is TealConfig else FAST[config]
    prompt = torch.randint(0, VOCAB, (1, 8), generator=torch.Generator().manual_seed(1)).cuda()
    calibration = [c.cuda() for c in _calibration()]
    logits, tokens = {}, {}
    for backend in ("dense", "auto"):
        torch.manual_seed(0)
        base = make_tiny_model(num_layers=2).to("cuda", torch.bfloat16)
        model = economize.apply(base, config(sparsity=0.5, backend=backend, **extra), calibration=calibration)
        if backend == "auto":
            assert model.simulations() == {}
            counts = _count_calls(monkeypatch)
        out = model.generate(
            prompt, max_new_tokens=12, do_sample=False, output_logits=True, return_dict_in_generate=True
        )
        tokens[backend], logits[backend] = out.sequences, torch.stack(out.logits).float()
    assert counts["teal_gemv"] >= 11  # every decoding step (the prefill takes the exact dense path)
    if config is not TealConfig:
        assert counts["row_masked_gemv"] >= 11
    reference = logits["dense"]
    assert float((logits["auto"] - reference).norm() / reference.norm()) < 3e-2
    assert (tokens["auto"] == tokens["dense"]).float().mean() > 0.8


# ---- launch plans and autotuning of the MLP kernels ----------------------------------------------------------------


def test_kernel_plan_picks_the_config_by_call_size() -> None:
    one, many = RowGemvConfig(16, 256, 8, 2), RowGemvConfig(64, 64, 4, 2)
    plan = KernelPlan(one, many)
    assert plan.config(1) == one and plan.config(5) == many and plan.max_rows == 16
    assert KernelPlan().config(1) is None  # the kernel's heuristic


def test_the_lenient_triton_backend_keeps_its_class_configs() -> None:
    linear = nn.Linear(8, 8)
    assert TritonMlpBackend().row_plan(linear) == KernelPlan(None, None)
    assert TritonMlpBackend().gemv_plan(linear) == KernelPlan(None, None)


@pytest.mark.gpu
@_kernel
def test_row_tuner_caches_a_config_that_matches_the_reference() -> None:
    clear_row_tuning_cache()
    tuner = RowTuner(grid_one_row=((16, 32), (128, 256), (4,)), grid_many_rows=((32,), (64,), (4,)), reps=3)
    weight = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16) * 0.02
    tuning = tuner.tune(weight)
    assert tuner.tune(weight) is tuning and tuning.one_row_speedup > 0 and tuning.many_rows_speedup > 0
    x = torch.randn(1, 1024, device="cuda", dtype=torch.bfloat16)
    keep = torch.rand(1, 2048, device="cuda") < 0.5
    reference = (x.float() @ weight.float().t()) * keep
    out = row_masked_gemv(x, weight, keep, tuning.one_row).float()
    assert float((out - reference).norm() / reference.norm()) < 1e-2
    assert bool((out[~keep] == 0).all())


@pytest.mark.gpu
@_kernel
def test_a_one_row_plan_sends_larger_calls_to_the_dense_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A map whose ``tl.dot`` kernel loses keeps the kernel for single rows only; the output is the same."""

    class OneRow(TritonMlpBackend):
        def row_plan(self, linear: nn.Linear) -> KernelPlan[RowGemvConfig]:
            return KernelPlan(max_rows=1)

        def gemv_plan(self, linear: nn.Linear) -> KernelPlan[GemvConfig]:
            return KernelPlan(max_rows=1)

    torch.manual_seed(0)
    mlp = make_tiny_model(num_layers=1).to("cuda", torch.bfloat16).model.layers[0].mlp
    gated = GatedMlp(mlp, CatsConfig())
    ops = OneRow().ops(gated)
    counts = _count_calls(monkeypatch)
    for rows, launches in ((1, 1), (4, 0)):
        x = torch.randn(rows, mlp.up_proj.in_features, device="cuda", dtype=torch.bfloat16)
        keep = torch.rand(rows, mlp.up_proj.out_features, device="cuda") < 0.5
        before = dict(counts)
        h = ops.rows(x, mlp.up_proj, keep)
        out = ops.cols(h, mlp.down_proj)
        assert counts["row_masked_gemv"] - before["row_masked_gemv"] == launches
        assert counts["teal_gemv"] - before["teal_gemv"] == launches
        torch.testing.assert_close(h, masked(mlp.up_proj(x), keep), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(out, mlp.down_proj(h), atol=2e-2, rtol=2e-2)
    ops.remove()


@pytest.mark.gpu
@_kernel
def test_auto_mlp_backend_installs_tuned_plans() -> None:
    torch.manual_seed(0)
    mlp = make_tiny_model(num_layers=1).to("cuda", torch.bfloat16).model.layers[0].mlp
    gated = GatedMlp(mlp, CatsConfig())
    backend = AutoMlpBackend()
    plan = backend.row_plan(mlp.up_proj)
    assert plan.one_row == backend.row_tuner.tune(mlp.up_proj.weight).one_row
    assert backend.gemv_plan(mlp.down_proj).one_row == backend.gemv_tuner.tune(mlp.down_proj.weight).one_row
    ops = backend.ops(gated)
    assert ops.realized
    ops.remove()
