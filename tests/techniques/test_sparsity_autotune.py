"""The ``"triton_autotune"`` TEAL backend: registration, tuning cache, and outputs equal to the ``"dense"`` backend."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import Tensor, nn

from economize.runtime import phase
from economize.techniques.prune import token_batches
from economize.techniques.sparsity import (
    TEAL_BACKENDS,
    AutotuneTritonBackend,
    DenseBackend,
    TealConfig,
    TritonBackend,
    calibrate_teal,
    clear_tuning_cache,
    install_teal,
    triton_autotune,
)
from economize.techniques.sparsity.teal_gemv import MAX_ROWS, GemvConfig, triton_available


class _SmallGrid(AutotuneTritonBackend):
    """A two-config grid per path keeps the tests to a few compilations."""

    grid_one_row = ((64, 128), (128,), (8,), (2,))
    grid_many_rows = ((128,), (32,), (4,), (4,))
    calls = 2
    reps = 3


def test_registered() -> None:
    assert TEAL_BACKENDS.get("triton_autotune") is AutotuneTritonBackend
    assert issubclass(AutotuneTritonBackend, TritonBackend)
    TealConfig(backend="triton_autotune")


def test_grid_respects_tile_limits() -> None:
    grid = triton_autotune._grid((32, 256), (64, 128), (1, 64), (2,), k=4096)
    assert all(c.block_n * c.block_k <= 32768 for c in grid)
    assert all(4096 // c.split_k >= c.block_k for c in grid)
    assert GemvConfig(256, 128, 64, 2, 2) not in grid  # 4096 // 64 = 64 channels per split < block_k


needs_gpu = pytest.mark.skipif(not triton_available(), reason="needs Triton and a CUDA GPU")


def _relative_error(out: Tensor, reference: Tensor) -> float:
    return float((out.float() - reference.float()).norm() / reference.float().norm())


@pytest.mark.gpu
@needs_gpu
def test_tuning_is_cached_per_shape() -> None:
    clear_tuning_cache()
    backend = _SmallGrid()
    w = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16)
    tuning = backend.tune(w)
    assert tuning.one_row.block_k == 128 and tuning.one_row_speedup > 0
    assert (tuning.many_rows is None) == (tuning.many_rows_speedup <= 1.0)
    assert backend.tune(torch.randn_like(w)) is tuning
    assert len(triton_autotune._CACHE) == 1
    clear_tuning_cache()
    assert not triton_autotune._CACHE


@pytest.mark.gpu
@needs_gpu
@pytest.mark.parametrize("rows", [1, 2, 9, 16, 64])
def test_installed_map_matches_dense_backend(rows: int) -> None:
    clear_tuning_cache()
    torch.manual_seed(0)
    linear = nn.Linear(2048, 5376).to("cuda", torch.bfloat16)
    x = torch.randn(1, rows, 2048, device="cuda", dtype=torch.bfloat16)
    t = torch.quantile(x.float().abs().flatten(), 0.5).reshape(()).contiguous()
    dense = DenseBackend().install({"map": linear}, {"map": t}, "all")
    with phase("decode"):
        reference = linear(x)
    dense.remove()
    installed = _SmallGrid().install({"map": linear}, {"map": t}, "all")
    try:
        patch = installed.handles[0]
        assert isinstance(patch, triton_autotune._KernelPatch)
        assert 1 in patch.configs and patch.max_rows in (1, MAX_ROWS)
        with phase("decode"):
            out = linear(x)
        assert _relative_error(out, reference) < 1e-2
    finally:
        installed.remove()
        clear_tuning_cache()


@pytest.mark.gpu
@needs_gpu
def test_tiny_model_logits_match_dense_backend(tiny2, adapter_of, calib, held_out) -> None:
    clear_tuning_cache()
    adapter = adapter_of(tiny2)
    config = TealConfig(sparsity=0.5, allocation="uniform", prefill="all")
    thresholds = calibrate_teal(adapter, config, token_batches(calib)).thresholds
    model = tiny2.to("cuda", torch.bfloat16)
    adapter = adapter_of(model)
    ids = held_out.cuda()
    calls = {"step": ids[:, :1], "verify": ids[:1, :9], "prefill": ids}
    logits = {}
    for backend in ("dense", "triton_autotune"):
        installed, _ = install_teal(adapter, thresholds, replace(config, backend=backend))
        try:
            logits[backend] = {name: model(x).logits.float() for name, x in calls.items()}
        finally:
            installed.remove()
    for name in calls:
        reference, out = logits["dense"][name], logits["triton_autotune"][name]
        assert _relative_error(out, reference) < 2e-2, name
        assert (out.argmax(-1) == reference.argmax(-1)).float().mean() > 0.95, name
    clear_tuning_cache()
