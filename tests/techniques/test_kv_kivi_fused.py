"""``kivi_fused``: the KIVI cache whose packed codes the ``"economize"`` attention kernel reads.

The oracle is ``kivi`` with the same settings. The dense buffers of both caches hold the same dequantized values,
so on CPU (every pass falls back to ``sdpa``) the tokens are equal, and on a GPU (``-m gpu``) the packed kernel
reproduces the dense kernel bit for bit, which the tests check on the tokens and on the path log.
"""

from __future__ import annotations

import pytest
import torch

import economize
from economize import DecodeConfig, JumpConfig
from economize.attention import install, kernel_usage
from economize.errors import IncompatibleTechniquesError
from economize.techniques.kv_kivi_fused import FusedKIVICache, FusedKIVIConfig
from economize.techniques.kv_quant import FP8KVConfig, KIVIConfig
from helpers import VOCAB, make_tiny_model, perturb

# The tiny model has head dim 16: groups of 8 tokens / channels and a residual of 8 tokens quantize most of a
# 16-token generation after prompts of 5 to 13 tokens.
SETTINGS = {"group_size": 8, "residual_length": 8}


def _prompts(device: str) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(5)
    return [torch.randint(3, VOCAB, (n,), generator=generator).to(device) for n in (13, 5, 9)]


def _greedy(attn: str, config: object, device: str, decoding: DecodeConfig, jump: bool = False) -> list[list[int]]:
    configs = [config]
    if jump:
        configs.insert(0, JumpConfig(target_layer=2, source_layer=4))
    model = economize.apply(make_tiny_model(attn).to(device), *configs)
    if jump:
        perturb(model.techniques["jump"], 0.05)
    out = [r.tokens.tolist() for r in model.decode_batch(_prompts(device), 16, decoding=decoding)]
    model.detach()
    return out


def test_needs_a_kernel_that_reads_kivi_codes() -> None:
    with pytest.raises(IncompatibleTechniquesError, match="kivi"):
        economize.apply(make_tiny_model("sdpa"), FusedKIVIConfig())


def test_is_exclusive_with_other_kv_quantizers() -> None:
    install()
    for other in (KIVIConfig(), FP8KVConfig()):
        with pytest.raises(IncompatibleTechniquesError):
            economize.apply(make_tiny_model("economize"), FusedKIVIConfig(), other)


@torch.no_grad()
@pytest.mark.parametrize("bits", [2, 4])
def test_cpu_fallback_decodes_like_kivi(bits: int) -> None:
    install()
    decoding = DecodeConfig(cycle_length=1)
    expected = _greedy("sdpa", KIVIConfig(bits=bits, **SETTINGS), "cpu", decoding)
    assert _greedy("economize", FusedKIVIConfig(bits=bits, **SETTINGS), "cpu", decoding) == expected


@torch.no_grad()
def test_packed_codes_match_the_dense_buffers() -> None:
    """The stored codes dequantize to exactly the values the cache keeps in its dense buffers."""
    from economize.attention.kivi import dequantize

    install()
    model = economize.apply(make_tiny_model("economize"), FusedKIVIConfig(bits=2, **SETTINGS))
    seen: list[FusedKIVICache] = []
    original = FusedKIVICache.pass_begin

    def spy(self, frontier):
        seen.append(self)
        return original(self, frontier)

    FusedKIVICache.pass_begin = spy  # type: ignore[method-assign]
    try:
        model.decode(_prompts("cpu")[0], 16)
    finally:
        FusedKIVICache.pass_begin = original  # type: ignore[method-assign]
    cache = seen[-1]
    assert cache.keys_quantized[0] > 0 and cache.values_quantized[0] > 0
    for layer, store in cache._stores.items():
        keys, values = cache.inner.read(layer)
        kq, vq = cache.keys_quantized[0], cache.values_quantized[0]
        k = dequantize(store.key_codes, store.key_scale, store.key_zero, 2, "key", 8, keys.dtype)
        v = dequantize(store.value_codes, store.value_scale, store.value_zero, 2, "value", store.value_group,
                       values.dtype)
        assert torch.equal(k[:, :, :kq], keys[:, :, :kq])
        assert torch.equal(v[:, :, :vq], values[:, :, :vq])
    model.detach()


@torch.no_grad()
def test_device_counts_live_on_the_device_that_reads_them() -> None:
    """A model split over several GPUs reads the counts on the GPU of each layer, and every copy stays in sync."""
    install()
    model = economize.apply(make_tiny_model("economize"), FusedKIVIConfig(bits=2, **SETTINGS))
    seen: list[FusedKIVICache] = []
    original = FusedKIVICache.pass_begin

    def spy(self, frontier):
        seen.append(self)
        return original(self, frontier)

    FusedKIVICache.pass_begin = spy  # type: ignore[method-assign]
    try:
        model.decode(_prompts("cpu")[0], 16)
    finally:
        FusedKIVICache.pass_begin = original  # type: ignore[method-assign]
    cache = seen[-1]
    cpu = cache._device_counts(torch.device("cpu"))
    other = cache._device_counts(torch.device("meta"))
    assert cpu[0].device.type == "cpu" and other[0].device.type == "meta"
    assert cache._device_counts(torch.device("cpu")) is cpu
    cache.keys_quantized[0] += 8
    cache._sync_counts()
    assert cpu[0].tolist() == cache.keys_quantized
    model.detach()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("cycle", [1, 4])
@torch.no_grad()
def test_gpu_kernel_decodes_like_kivi(bits: int, cycle: int) -> None:
    install()
    decoding = DecodeConfig(cycle_length=cycle)
    dense = _greedy("economize", KIVIConfig(bits=bits, **SETTINGS), "cuda", decoding, jump=cycle > 1)
    assert dense == _greedy("sdpa", KIVIConfig(bits=bits, **SETTINGS), "cuda", decoding, jump=cycle > 1)
    kernel_usage(reset=True)
    got = _greedy("economize", FusedKIVIConfig(bits=bits, **SETTINGS), "cuda", decoding, jump=cycle > 1)
    assert got == dense
    paths = {row["path"] for row in kernel_usage(reset=True)}
    assert "kernel_kivi" in paths
