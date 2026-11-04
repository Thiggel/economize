"""``kv_fp8_fused``: the fp8 KV cache whose bytes the ``"economize"`` attention kernel decodes in registers.

The oracle is ``kv_fp8`` (``scale="none"``) on the ``sdpa`` model: attention sees the same fp8 values in both, so
greedy tokens are equal and logits agree to fp32 kernel rounding. On CPU every pass falls back to ``sdpa`` through
the upcast in the attention function; on a GPU (``-m gpu``) decoding steps and verification passes run the kernel.
"""

from __future__ import annotations

import pytest
import torch

import economize
from economize import DecodeConfig, JumpConfig
from economize.attention import install, kernel_calls
from economize.errors import ConfigError, IncompatibleTechniquesError
from economize.techniques.kv_fp8_fused import FusedFP8Cache, FusedFP8KVConfig
from economize.techniques.kv_quant import FP8KVConfig, KIVIConfig
from helpers import VOCAB, make_tiny_model, perturb

pytestmark = pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="needs torch fp8 dtypes")


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


def test_config_rejects_scales() -> None:
    with pytest.raises(ConfigError):
        FusedFP8KVConfig(scale="token")
    with pytest.raises(ConfigError):
        FusedFP8KVConfig(format="e3m4")


def test_needs_a_kernel_that_reads_fp8() -> None:
    with pytest.raises(IncompatibleTechniquesError, match="economize"):
        economize.apply(make_tiny_model("sdpa"), FusedFP8KVConfig())


def test_is_exclusive_with_other_kv_quantizers() -> None:
    install()
    with pytest.raises(IncompatibleTechniquesError):
        economize.apply(make_tiny_model("economize"), FusedFP8KVConfig(), KIVIConfig())


@torch.no_grad()
@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_cpu_fallback_decodes_like_kv_fp8(fmt: str) -> None:
    install()
    decoding = DecodeConfig(cycle_length=1)
    expected = _greedy("sdpa", FP8KVConfig(format=fmt), "cpu", decoding)
    assert _greedy("economize", FusedFP8KVConfig(format=fmt), "cpu", decoding) == expected


@torch.no_grad()
def test_hands_fp8_views_to_attention_and_keeps_reads_in_the_model_dtype() -> None:
    install()
    model = economize.apply(make_tiny_model("economize"), FusedFP8KVConfig())
    seen: list[torch.dtype] = []
    original = FusedFP8Cache.update

    def spy(self, *args, **kwargs):
        keys, values = original(self, *args, **kwargs)
        seen.append(keys.dtype)
        return keys, values

    FusedFP8Cache.update = spy  # type: ignore[method-assign]
    try:
        model.decode(_prompts("cpu")[0], 4)
    finally:
        FusedFP8Cache.update = original  # type: ignore[method-assign]
    assert seen and set(seen) == {torch.float8_e4m3fn}
    cache = model.techniques["kv_fp8_fused"].wrap_cache(model.runner.new_cache(32))
    assert isinstance(cache, FusedFP8Cache)
    stats_model = economize.apply(make_tiny_model("economize"), FusedFP8KVConfig())
    stats_model.decode(_prompts("cpu")[0], 8)
    assert stats_model.techniques["kv_fp8_fused"].kv_stats()["kv_compression"] == pytest.approx(4.0)
    model.detach()
    stats_model.detach()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("cycle", [1, 4])
@torch.no_grad()
def test_gpu_kernel_decodes_like_kv_fp8(graphs: bool, cycle: int) -> None:
    install()
    decoding = DecodeConfig(cycle_length=cycle, cuda_graphs=graphs)
    expected = _greedy("sdpa", FP8KVConfig(), "cuda", decoding, jump=cycle > 1)
    before = kernel_calls()["kernel"]
    got = _greedy("economize", FusedFP8KVConfig(), "cuda", decoding, jump=cycle > 1)
    assert got == expected
    assert kernel_calls()["kernel"] > before
