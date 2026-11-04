"""Storage kernels for 2:4 weights (``SEMI_STRUCTURED_KERNELS``) and the per-shape ``"auto"`` timing policy."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from economize.errors import ConfigError
from economize.techniques.prune import (
    SEMI_STRUCTURED_KERNELS,
    PruneConfig,
    SemiStructuredKernel,
    prune_model,
    semi_kernels,
)
from economize.techniques.prune.masks import nm_mask
from economize.techniques.prune.semi_structured import semi_structured_available


def test_builtin_kernels_registered() -> None:
    names = set(SEMI_STRUCTURED_KERNELS.names())
    assert {"default", "cusparselt", "cutlass", "dense", "auto"} <= names
    assert all(isinstance(SEMI_STRUCTURED_KERNELS.get(n), SemiStructuredKernel) for n in names)
    assert set(semi_kernels.AUTO_CANDIDATES) <= names


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sparsity": "2:4", "semi_structured": True, "semi_structured_kernel": "nope"},
        {"sparsity": "2:4", "semi_structured_kernel": "auto"},
    ],
)
def test_kernel_config_validation(kwargs) -> None:
    with pytest.raises(ConfigError):
        PruneConfig(**kwargs)


def test_kernel_config_default_is_backward_compatible() -> None:
    assert PruneConfig(sparsity="2:4", semi_structured=True).semi_structured_kernel == "default"
    assert PruneConfig(sparsity="2:4", semi_structured=True, semi_structured_kernel="auto").semi_structured


def test_registering_a_kernel_extends_config_choices() -> None:
    kernel = SemiStructuredKernel(convert=lambda w: w, available=lambda d: True)
    SEMI_STRUCTURED_KERNELS.register("test_identity", kernel)
    try:
        cfg = PruneConfig(sparsity="2:4", semi_structured=True, semi_structured_kernel="test_identity")
        assert cfg.semi_structured_kernel == "test_identity"
    finally:
        SEMI_STRUCTURED_KERNELS.unregister("test_identity")


def _weight_24(n: int, k: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    w = torch.randn(n, k, device="cuda", dtype=dtype) * 0.02
    return w * nm_mask(w.float().abs(), 2, 4).to(dtype)


needs_sm80 = pytest.mark.skipif(not semi_structured_available(), reason="2:4 kernels need CUDA sm80+")


@pytest.mark.gpu
@needs_sm80
@pytest.mark.parametrize("name", ["default", "cusparselt", "cutlass", "dense", "auto"])
@pytest.mark.parametrize("m", [1, 8, 33])
def test_kernel_matches_dense(name: str, m: int) -> None:
    kernel = SEMI_STRUCTURED_KERNELS.get(name)
    device = torch.device("cuda")
    if not kernel.available(device):
        pytest.skip(f"{name} unavailable")
    w = _weight_24(256, 512)
    x = torch.randn(m, 512, device="cuda", dtype=torch.bfloat16)
    stored = kernel.convert(w)
    out = F.linear(x, stored)
    ref = F.linear(x.float(), w.float())
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.gpu
@needs_sm80
def test_auto_choice_is_cached_and_valid() -> None:
    semi_kernels.clear_auto_cache()
    w = _weight_24(512, 1024)
    first = semi_kernels.auto_choice(w)
    assert first in semi_kernels.AUTO_CANDIDATES
    assert len(semi_kernels._AUTO_CACHE) == 1
    assert semi_kernels.auto_choice(_weight_24(512, 1024)) == first
    assert len(semi_kernels._AUTO_CACHE) == 1


@pytest.mark.gpu
@needs_sm80
def test_auto_falls_back_to_dense_without_sparse_candidates() -> None:
    semi_kernels.clear_auto_cache()
    w = _weight_24(256, 512)
    assert semi_kernels.auto_choice(w, candidates=("dense",)) == "dense"
    semi_kernels.clear_auto_cache()


@pytest.mark.gpu
@needs_sm80
@pytest.mark.parametrize("name", ["auto", "dense", "cutlass"])
def test_prune_model_with_kernel_matches_masked_dense(tiny2, adapter_of, name: str) -> None:
    model = tiny2.to("cuda", torch.float16)
    adapter = adapter_of(model)
    ids = torch.randint(0, 128, (1, 64), device="cuda")
    prune_model(adapter, PruneConfig(method="magnitude", sparsity="2:4", targets=("mlp.*",)))
    dense = model(ids).logits
    prune_model(
        adapter,
        PruneConfig(
            method="magnitude", sparsity="2:4", targets=("mlp.*",), semi_structured=True, semi_structured_kernel=name
        ),
    )
    torch.testing.assert_close(model(ids).logits, dense, atol=5e-2, rtol=0)
