"""Hub kernels: op resolution against a fake ``kernels.get_kernel`` (no network), fallbacks, and ``kernelize``."""

from __future__ import annotations

import logging
import sys
import types
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from economize.integrations import kernels as ek
from helpers import make_tiny_model

OP = "test_scaled"


class FakeHub:
    """Stands in for the ``kernels`` package: records ``get_kernel`` calls, serves the configured modules."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.modules: dict[str, types.ModuleType] = {}

    def get_kernel(self, repo_id: str, **kwargs: Any) -> types.ModuleType:
        self.calls.append((repo_id, kwargs))
        if repo_id not in self.modules:
            raise FileNotFoundError(f"no build of {repo_id} for this platform")
        return self.modules[repo_id]


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeHub]:
    fake = FakeHub()
    package = types.ModuleType("kernels")
    package.get_kernel = fake.get_kernel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kernels", package)
    monkeypatch.delenv(ek.ENV_VAR, raising=False)
    ek.clear_cache()
    yield fake
    ek.clear_cache()
    if OP in ek.KERNEL_OPS:
        ek.KERNEL_OPS.unregister(OP)


def _scaled(x: torch.Tensor, scale: float = 2.0) -> torch.Tensor:
    """x times scale."""
    return x * scale


def _register(*sources: ek.HubKernel) -> None:
    ek.KERNEL_OPS.register(OP, ek.KernelOp(_scaled, sources))


def _cpu_kernel(repo: str = "org/fast", function: str = "scaled", **kwargs: Any) -> ek.HubKernel:
    return ek.HubKernel(repo, function, version=1, devices=("cpu",), **kwargs)


def test_hub_kernel_is_loaded_once_per_device(hub: FakeHub) -> None:
    hub.modules["org/fast"] = types.SimpleNamespace(scaled=lambda x, scale=2.0: x * scale + 1)  # type: ignore[assignment]
    _register(_cpu_kernel())
    scaled = ek.op(OP)
    x = torch.ones(3)
    assert torch.equal(scaled(x), x * 2 + 1) and torch.equal(scaled(x, scale=3.0), x * 3 + 1)
    assert hub.calls == [("org/fast", {"version": 1, "revision": None, "trust_remote_code": False})]
    assert scaled.__doc__ == "x times scale." and repr(scaled) == f"op({OP!r})"


def test_fallbacks(hub: FakeHub, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    x = torch.ones(2)
    _register(ek.HubKernel("org/gpu-only", "scaled", devices=("cuda",)))
    assert ek.resolve(OP, "cpu") is _scaled and not hub.calls  # no kernel for the device: no download attempt

    add_failing = _cpu_kernel("org/missing")
    ek.add_hub_kernel(OP, add_failing)
    with caplog.at_level(logging.WARNING, logger=ek.__name__):
        assert torch.equal(ek.op(OP)(x), x * 2) and torch.equal(ek.op(OP)(x), x * 2)
    assert [repo for repo, _ in hub.calls] == ["org/missing"]  # the failure is cached with the fallback
    assert sum("failed to load" in record.getMessage() for record in caplog.records) == 1

    hub.modules["org/working"] = types.SimpleNamespace(scaled=lambda x: x - 1)  # type: ignore[assignment]
    ek.add_hub_kernel(OP, _cpu_kernel("org/working"), first=False)  # tried after the failing one
    assert torch.equal(ek.op(OP)(x), x - 1)

    monkeypatch.setenv(ek.ENV_VAR, "0")
    ek.clear_cache()
    calls = len(hub.calls)
    assert ek.resolve(OP, torch.device("cpu")) is _scaled and len(hub.calls) == calls and not ek.hub_kernels_enabled()


def test_without_the_kernels_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "kernels", None)
    ek.clear_cache()
    try:
        ek.KERNEL_OPS.register(OP, ek.KernelOp(_scaled, (_cpu_kernel(),)))
        assert not ek.hub_kernels_enabled() and ek.resolve(OP, "cpu") is _scaled
    finally:
        ek.KERNEL_OPS.unregister(OP)
        ek.clear_cache()


def test_builtin_ops_match_their_definitions(hub: FakeHub) -> None:
    x = torch.randn(3, 5, 16, generator=torch.Generator().manual_seed(0))
    gate, up = x[..., :8], x[..., 8:]
    assert torch.allclose(ek.op("silu_and_mul")(x), F.silu(gate) * up)
    assert torch.allclose(ek.op("gelu_tanh_and_mul")(x), F.gelu(gate, approximate="tanh") * up)
    weight, threshold = torch.randn(4, 16), torch.tensor(0.5)
    expected = F.linear(x * (x.abs() > threshold), weight)
    assert torch.allclose(ek.op("masked_linear")(x, weight, threshold), expected)
    assert not hub.calls  # the built-in hub kernels serve GPUs only


def test_out_parameter_kernels_are_adapted(hub: FakeHub) -> None:
    """``kernels-community/activation`` writes into ``out`` (the vLLM calling convention)."""

    def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
        out.copy_(ek.silu_and_mul(x))

    hub.modules["kernels-community/activation"] = types.SimpleNamespace(silu_and_mul=silu_and_mul)  # type: ignore[assignment]
    loaded = ek.resolve("silu_and_mul", "cuda")  # resolved for a GPU, called here on CPU tensors
    x = torch.randn(2, 6)
    assert loaded is not ek.silu_and_mul and torch.equal(loaded(x), ek.silu_and_mul(x))
    assert hub.calls[0][0] == "kernels-community/activation" and hub.calls[0][1]["version"] == 1


# ---- kernelize -----------------------------------------------------------------------------------------------------
@pytest.fixture
def real_kernels() -> Any:
    return pytest.importorskip("kernels")


def test_kernelize_covers_the_base_and_the_added_modules(
    real_kernels: Any, applied_jump: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    model = applied_jump(adaptation="full", exit_layer=5)
    seen: list[tuple[Any, dict[str, Any]]] = []
    kernelize = real_kernels.kernelize

    def spy(module: Any, **kwargs: Any) -> Any:
        seen.append((module, kwargs))
        return kernelize(module, **kwargs)

    monkeypatch.setattr(real_kernels, "kernelize", spy)
    prompt = torch.arange(1, 9)[None]
    before = model(prompt).logits
    assert ek.kernelize(model) is model
    assert model.base_model.use_kernels and torch.equal(model(prompt).logits, before)  # no CPU kernels: same maths
    assert [module for module, _ in seen] == [model.techniques]
    assert seen[0][1] == {"mode": real_kernels.Mode.INFERENCE, "device": "cpu", "use_fallback": True}
    layer_names = {getattr(type(m), "kernel_layer_name", None) for m in model.techniques.modules()}
    assert "RMSNorm" in layer_names  # the trainable block copies carry kernelizable layers of their own


def test_kernelize_a_plain_model_and_mode_names(real_kernels: Any) -> None:
    Mode = real_kernels.Mode
    base = make_tiny_model()
    assert ek.kernelize(base, mode="inference") is base and base.use_kernels
    assert ek._mode(Mode, "training|torch_compile", False) == Mode.TRAINING | Mode.TORCH_COMPILE
    assert ek._mode(Mode, None, True) == Mode.TRAINING and ek._mode(Mode, Mode.INFERENCE, True) == Mode.INFERENCE
