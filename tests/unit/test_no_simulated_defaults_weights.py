"""Weight compression has no simulated default: with its default configuration, every weight technique runs on a
real kernel or is refused with a reason, on every backend (:func:`economize.backend_support`).

The simulated paths survive only as explicit references (``quant(backend="fake")``,
``prune(semi_structured=False)``), which still say so.
"""

from __future__ import annotations

import pytest
import torch

import economize
from economize import SupportStatus
from economize.errors import ConfigError
from economize.techniques.prune import PruneConfig
from economize.techniques.quant import DraftQuantConfig, QuantConfig
from helpers import make_tiny_model

#: The weight-compression techniques and the backends on which their default configuration runs a real kernel.
REAL_DEFAULTS: dict[str, set[str]] = {
    "quant": {"transformers", "vllm", "sglang", "onnx", "gguf", "mlx"},
    "draft_quant": {"transformers"},
    "prune": {"transformers"},
    "lowrank": {"transformers", "vllm", "onnx"},
    "littlebit": {"transformers"},
    "depth_prune": {"transformers", "vllm", "sglang", "onnx", "gguf", "mlx", "trtllm"},
}
BACKENDS = {"transformers", "vllm", "sglang", "onnx", "gguf", "mlx", "trtllm"}


def test_every_backend_is_reported() -> None:
    assert set(economize.backend_support("quant")) >= BACKENDS


@pytest.mark.parametrize("name", sorted(REAL_DEFAULTS))
def test_no_default_is_simulated(name: str) -> None:
    for backend, support in economize.backend_support(name).items():
        assert support.status is not SupportStatus.SIMULATED, (name, backend, support.detail)
        assert support.simulation is None, (name, backend)
        if support.status is SupportStatus.UNSUPPORTED:
            assert len(support.detail) > 20, (name, backend, support.detail)  # an honest reason, not a blank


@pytest.mark.parametrize("name", sorted(REAL_DEFAULTS))
def test_real_defaults_are_the_documented_ones(name: str) -> None:
    real = {backend for backend, support in economize.backend_support(name).items()
            if support.status is SupportStatus.REAL}
    assert real & BACKENDS == REAL_DEFAULTS[name]


@pytest.mark.parametrize(
    ("backend", "words"),
    [
        ("vllm", "2:4 GEMM"),
        ("sglang", "2:4 scheme"),
        ("onnx", "semi_structured=False"),
        ("trtllm", "weight_sparsity"),
    ],
)
def test_prune_refusals_name_the_missing_kernel_and_the_reference(backend: str, words: str) -> None:
    detail = economize.backend_support("prune")[backend].detail
    assert words in detail and "semi_structured=False" in detail


def test_draft_quant_refusals_say_why() -> None:
    support = economize.backend_support("draft_quant")
    for backend in ("vllm", "sglang"):
        assert support[backend].status is SupportStatus.UNSUPPORTED
        assert "draft" in support[backend].detail


# ---- the explicit references ---------------------------------------------------------------------------------------
REFERENCES = [
    ("quant", QuantConfig(backend="fake")),
    ("prune", PruneConfig(semi_structured=False)),
    ("prune", PruneConfig(sparsity=0.5, semi_structured=False)),
]


@pytest.mark.parametrize(("name", "config"), REFERENCES)
def test_explicit_references_still_report_simulated(name: str, config: object) -> None:
    support = economize.backend_support(config)["transformers"]  # type: ignore[arg-type]
    assert support.status is SupportStatus.SIMULATED and support.simulation is not None, name
    assert "saves no" in support.detail


@pytest.mark.parametrize("backend", ["vllm", "sglang", "onnx", "gguf", "mlx", "trtllm"])
def test_the_prune_reference_is_served_dense_and_reported_simulated(backend: str) -> None:
    support = economize.backend_support(PruneConfig(semi_structured=False))[backend]
    assert support.status is SupportStatus.SIMULATED, (backend, support.detail)


def test_draft_quant_reference_is_simulated() -> None:
    support = economize.backend_support(DraftQuantConfig(backend="fake"))["transformers"]
    assert support.status is SupportStatus.SIMULATED


# ---- configurations without a real kernel are refused, not simulated ------------------------------------------------
@pytest.mark.parametrize("sparsity", [0.5, "4:8"])
def test_patterns_without_a_sparse_kernel_are_refused(sparsity: float | str) -> None:
    support = economize.backend_support(PruneConfig(sparsity=sparsity))["transformers"]
    assert support.status is SupportStatus.UNSUPPORTED
    assert "sparse tensor cores run only 2:4" in support.detail and "semi_structured=False" in support.detail


def test_refused_pattern_is_rejected_before_any_weight_changes() -> None:
    model = make_tiny_model(num_layers=2)
    before = {name: weight.clone() for name, weight in model.state_dict().items()}
    with pytest.raises(ConfigError, match="semi_structured=False"):
        economize.apply(model, PruneConfig(method="magnitude", sparsity=0.5))
    assert all(torch.equal(before[name], weight) for name, weight in model.state_dict().items())


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the CPU refusal")
def test_default_prune_on_cpu_asks_for_a_gpu_or_the_reference() -> None:
    with pytest.raises(ConfigError, match=r"compute capability >= 8\.0.*semi_structured=False"):
        economize.apply(make_tiny_model(num_layers=2), PruneConfig(method="magnitude"))


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the CPU refusal")
def test_default_quant_on_cpu_lists_the_extras_to_install() -> None:
    with pytest.raises(ConfigError, match=r"backend='auto' found no real kernel") as caught:
        economize.apply(make_tiny_model(num_layers=2), QuantConfig())
    message = str(caught.value)
    assert "economize[" in message and "backend='fake'" in message


def test_auto_refuses_a_layout_no_kernel_stores() -> None:
    with pytest.raises(ConfigError, match=r"no real kernel that stores.*backend=.fake."):
        QuantConfig(format="int3")
