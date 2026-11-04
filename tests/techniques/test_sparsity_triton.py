"""The ``"triton"`` TEAL backend: the sparse GEMV against the masked dense matmul, the installed maps, the model.

Everything is skipped unless Triton imports and a CUDA GPU is present.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from economize.runtime import phase
from economize.techniques.prune import token_batches
from economize.techniques.sparsity import (
    TEAL_BACKENDS,
    DenseBackend,
    PhasePolicy,
    TealConfig,
    TritonBackend,
    calibrate_teal,
    install_teal,
    sparsify,
)
from economize.techniques.sparsity.teal_gemv import MAX_ROWS, default_config, teal_gemv, triton_available
from economize.techniques.sparsity.triton_backend import kernel_eligible

pytestmark = pytest.mark.skipif(not triton_available(), reason="needs Triton and a CUDA GPU")

# (out, in) of the Gemma-4-31B linear maps.
GEMMA4_31B = {
    "q_proj": (8192, 5376),
    "k_proj": (4096, 5376),
    "o_proj": (5376, 8192),
    "gate_proj": (21504, 5376),
    "down_proj": (5376, 21504),
}


def _relative_error(out: Tensor, reference: Tensor) -> float:
    return float((out.float() - reference).norm() / reference.norm())


def _inputs(m: int, k: int, seed: int) -> Tensor:
    return torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=torch.Generator("cuda").manual_seed(seed))


def _threshold(x: Tensor, sparsity: float) -> Tensor:
    return torch.quantile(x.float().abs().flatten(), sparsity).reshape(()).contiguous()


def test_registered() -> None:
    assert TEAL_BACKENDS.get("triton") is TritonBackend
    TealConfig(backend="triton")


@pytest.mark.parametrize("m", [1, 2, 9, 16])
@pytest.mark.parametrize("name", list(GEMMA4_31B))
def test_kernel_matches_masked_dense(name: str, m: int) -> None:
    n, k = GEMMA4_31B[name]
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=torch.Generator("cuda").manual_seed(0))
    weight_t = weight.t().contiguous()
    x = _inputs(m, k, seed=m)
    for sparsity in (0.0, 0.5):
        t = _threshold(x, sparsity) if sparsity else torch.zeros((), device="cuda")
        masked = sparsify(x, t)
        reference = masked.float() @ weight.float().t()
        cublas = F.linear(masked, weight)
        out = teal_gemv(x, weight_t, t)
        assert out.shape == (m, n) and out.dtype == torch.bfloat16
        # within bfloat16 rounding of the output, as cuBLAS
        assert _relative_error(out, reference) < 2 * _relative_error(cublas, reference) + 1e-3


def test_kernel_masks_like_pytorch_at_the_threshold() -> None:
    """``x.abs() > t`` in PyTorch rounds a float32 ``t`` to bfloat16; the kernel drops the same inputs."""
    k = 4096
    x = torch.linspace(0.4, 0.6, k, device="cuda").to(torch.bfloat16).reshape(1, k)
    t = torch.tensor(0.5003, device="cuda")  # between two bfloat16 values
    weight_t = torch.zeros(k, 8, device="cuda", dtype=torch.bfloat16)
    weight_t[:, 0] = 1
    kept = int((x.abs() > t).sum())
    assert float(teal_gemv(torch.ones_like(x) * (x.abs() > t), weight_t, torch.zeros((), device="cuda"))[0, 0]) == kept
    out = teal_gemv(x, weight_t, t)[0, 0]
    assert torch.allclose(out.float(), sparsify(x, t).float().sum(), rtol=1e-2)


def test_kernel_split_configs_agree() -> None:
    n, k = 1024, 4096
    weight_t = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.02
    for m in (1, 5):
        x = _inputs(m, k, seed=3)
        t = _threshold(x, 0.4)
        reference = sparsify(x, t).float() @ weight_t.float()
        base = default_config(m, n, k, x.device)
        for split in (1, 2, 8):
            config = type(base)(base.block_n, base.block_k, split, base.num_warps, base.num_stages)
            assert _relative_error(teal_gemv(x, weight_t, t, config), reference) < 1e-2


def test_kernel_rejects_more_rows() -> None:
    weight_t = torch.zeros(64, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        teal_gemv(
            torch.zeros(MAX_ROWS + 1, 64, device="cuda", dtype=torch.bfloat16), weight_t, torch.zeros((), device="cuda")
        )


# ---- installed maps ------------------------------------------------------------------------------------------------


def _linear(n: int, k: int, bias: bool = True) -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(k, n, bias=bias).to("cuda", torch.bfloat16)


@pytest.mark.parametrize("rows", [1, 2, 9, 16, 64])
def test_installed_map_matches_dense_backend(rows: int) -> None:
    linear = _linear(5376, 2048)
    x = _inputs(rows, 2048, seed=rows).reshape(1, rows, 2048)
    t = _threshold(x, 0.5)
    expected = {}
    dense = DenseBackend().install({"map": linear}, {"map": t}, "all")
    for name in (None, "decode", "refresh"):
        with phase(name):
            expected[name] = linear(x)
    dense.remove()
    installed = TritonBackend().install({"map": linear}, {"map": t}, "all")
    try:
        for name, reference in expected.items():
            with phase(name):
                out = linear(x)
            if rows > MAX_ROWS:  # the dense fallback: the same cuBLAS call on the same values
                assert torch.allclose(out, reference, atol=1e-2, rtol=1e-2)
            else:
                assert _relative_error(out, reference.float()) < 1e-2
    finally:
        installed.remove()


def test_install_stores_the_weight_transposed_in_place_and_remove_restores_it() -> None:
    linear = _linear(4096, 4096, bias=False)
    original = linear.weight.detach().clone()
    t = torch.zeros((), device="cuda")
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    installed = TritonBackend().install({"map": linear}, {"map": t}, "all")
    assert torch.cuda.memory_allocated() == before  # replaced, not duplicated
    assert linear.weight.shape == (4096, 4096) and linear.weight.t().is_contiguous()
    assert torch.equal(linear.weight, original)
    assert isinstance(linear.weight, nn.Parameter)
    installed.remove()
    assert linear.weight.is_contiguous() and torch.equal(linear.weight, original)
    assert "forward" not in linear.__dict__
    assert torch.cuda.memory_allocated() == before


def test_phase_policy_dense_calls_skip_the_kernel() -> None:
    linear = _linear(256, 512)
    x = _inputs(4, 512, seed=0).reshape(1, 4, 512)
    t = _threshold(x, 0.5)
    installed = TritonBackend().install({"map": linear}, {"map": t}, PhasePolicy())  # exact passes stay dense
    try:
        with phase("refresh"):
            assert torch.equal(linear(x), F.linear(x, linear.weight, linear.bias))
        with phase("decode"):
            assert not torch.allclose(linear(x), F.linear(x, linear.weight, linear.bias))
    finally:
        installed.remove()


def test_ineligible_maps_get_the_dense_hook() -> None:
    fp32 = nn.Linear(64, 32).cuda()
    bf16 = _linear(32, 64)
    assert not kernel_eligible(fp32) and kernel_eligible(bf16)
    t = torch.tensor(0.5, device="cuda")
    x = torch.randn(1, 1, 64, device="cuda")
    installed = TritonBackend().install({"fp32": fp32, "bf16": bf16}, {"fp32": t, "bf16": t}, "all")
    try:
        assert "forward" not in fp32.__dict__ and "forward" in bf16.__dict__
        assert torch.equal(fp32(x), F.linear(sparsify(x, t), fp32.weight, fp32.bias))
    finally:
        installed.remove()
    assert "forward" not in bf16.__dict__ and torch.equal(fp32(x), F.linear(x, fp32.weight, fp32.bias))


def test_cuda_graph_replay_reads_the_live_threshold_and_input() -> None:
    linear = _linear(4096, 4096)
    t = torch.zeros((), device="cuda")
    installed = TritonBackend().install({"map": linear}, {"map": t}, "all")
    try:
        static_x = _inputs(1, 4096, seed=0).reshape(1, 1, 4096)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            linear(static_x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = linear(static_x)
        for seed, sparsity in ((1, 0.4), (2, 0.6)):
            static_x.copy_(_inputs(1, 4096, seed=seed).reshape(1, 1, 4096))
            t.copy_(_threshold(static_x, sparsity))
            graph.replay()
            assert torch.equal(static_out, linear(static_x))
    finally:
        installed.remove()


# ---- model level ---------------------------------------------------------------------------------------------------


def test_tiny_model_logits_match_dense_backend(tiny2, adapter_of, calib, held_out) -> None:
    adapter = adapter_of(tiny2)
    config = TealConfig(sparsity=0.5, allocation="uniform", prefill="all")
    thresholds = calibrate_teal(adapter, config, token_batches(calib)).thresholds
    model = tiny2.to("cuda", torch.bfloat16)
    adapter = adapter_of(model)
    ids = held_out.cuda()
    calls = {"step": ids[:, :1], "verify": ids[:1, :9], "prefill": ids}  # 2, 9 and 48 rows
    logits = {}
    for backend in ("dense", "triton"):
        installed, _ = install_teal(adapter, thresholds, replace(config, backend=backend))
        try:
            logits[backend] = {name: model(x).logits.float() for name, x in calls.items()}
        finally:
            installed.remove()
    dense_logits = {name: model(x).logits.float() for name, x in calls.items()}
    for name in calls:
        reference, out = logits["dense"][name], logits["triton"][name]
        assert (reference - dense_logits[name]).abs().max() > 0.05  # the thresholds change the logits
        assert _relative_error(out, reference) < 2e-2, name
        assert (out.argmax(-1) == reference.argmax(-1)).float().mean() > 0.95, name
