"""The packed KIVI storage: the cache really holds low-bit codes, and it decodes like the simulated storage."""

from __future__ import annotations

import pytest
import torch

import economize
from economize.attention.interface import install
from economize.errors import ConfigError
from economize.runtime import phase
from economize.techniques.kv_evict import KVEvictConfig
from economize.techniques.kv_kivi_fused import FusedKIVIConfig
from economize.techniques.kv_quant import KIVI_STORAGES, KIVIConfig, PackedKIVICache, fake_quantize
from helpers import VOCAB, make_tiny_model

PREFIX, TOTAL, SLOTS = 12, 40, 64
LAYERS, KV_HEADS, DIM = 6, 2, 16  # make_tiny_model: 6 layers, 2 KV heads of width 16


def _sequence(seed: int = 0, length: int = TOTAL) -> torch.Tensor:
    return torch.randint(0, VOCAB, (1, length), generator=torch.Generator().manual_seed(seed))


def _passes(runner, cache, seq: torch.Tensor, chunk: int) -> torch.Tensor:
    """Teacher-forced logits of ``seq`` through ``cache``: a prompt of :data:`PREFIX` tokens, then passes of ``chunk``
    tokens."""
    with phase("prefill"):
        ctx = runner.context(cache, torch.arange(PREFIX)[None], trusted=True)
        out = [runner.logits(runner.forward(seq[:, :PREFIX], ctx).final)[0]]
    for t in range(PREFIX, seq.shape[1], chunk):
        ctx = runner.context(cache, torch.arange(t, min(t + chunk, seq.shape[1]))[None], trusted=True)
        with phase("decode"):
            out.append(runner.logits(runner.forward(seq[:, t : t + chunk], ctx).final)[0])
    return torch.cat(out).float()


def _decode(model: economize.EfficientModel, seq: torch.Tensor, capacity: int = SLOTS, chunk: int = 1):
    """:func:`_passes` through a new cache of the KIVI technique of ``model``, and that cache."""
    cache = model.techniques["kivi"].wrap_cache(model.runner.new_cache(capacity))
    return _passes(model.runner, cache, seq, chunk), cache


def _packed_bytes(slots: int, bits: int, group: int, ring: int, element: int) -> int:
    """Bytes of one layer of the packed storage: codes, fp16 scales and zero points, and the two rings."""
    value_groups = DIM // group if DIM % group == 0 else 1
    codes = 2 * slots * DIM * bits // 8
    key_params = 2 * -(-slots // group) * DIM * 2
    value_params = 2 * slots * value_groups * 2
    rings = 2 * (ring + 1) * DIM * element
    return KV_HEADS * (codes + key_params + value_params + rings)


def test_storages_are_registered() -> None:
    assert {"simulated", "packed"} <= set(KIVI_STORAGES.names())
    assert KIVI_STORAGES.get("packed").simulation is None
    assert KIVI_STORAGES.get("simulated").simulation is not None
    assert KIVIConfig().kivi_storage == "packed"
    with pytest.raises(ConfigError):
        KIVIConfig(storage="paged")
    with pytest.raises(ConfigError):
        KIVIConfig(headroom=0)


@pytest.mark.parametrize(("bits", "group"), [(2, 16), (4, 16), (2, 4)])
def test_the_cache_holds_the_packed_bytes(bits: int, group: int) -> None:
    """Every covered layer lives in codes, fp16 scales and zero points and a short ring, the inner cache allocates
    nothing for it, and the bytes against the dense bf16 cache match the formula."""
    slots, residual, headroom = 1024, 16, 16
    base = make_tiny_model().to(torch.bfloat16)
    config = KIVIConfig(bits=bits, group_size=group, residual_length=residual, headroom=headroom)
    model = economize.apply(base, config)
    _, cache = _decode(model, _sequence(), capacity=slots)
    assert isinstance(cache, PackedKIVICache)
    assert cache.inner.keys == {} and cache.inner.values == {}  # no dense buffer behind the codes
    ring = residual + headroom
    assert cache.residual_ring == ring
    num_slots = cache.num_slots
    expected = LAYERS * _packed_bytes(num_slots, bits, group, ring, element=2)
    assert cache.allocated_bytes() == expected
    dense = LAYERS * 2 * KV_HEADS * num_slots * DIM * 2  # K and V in bf16
    ratio = cache.allocated_bytes() / dense
    # the codes are bits/16 of bf16, the fp16 scale and zero point of a group of K and of V add 2/group
    asymptote = bits / 16 + 2 / group
    assert ratio == pytest.approx(asymptote, abs=(ring + 2) / num_slots)
    if (bits, group) == (2, 16):
        assert ratio < 0.3
    for layer in range(LAYERS):
        assert cache.has_layer(layer)
    model.detach()


@pytest.mark.parametrize("attn", ["sdpa", "eager", "economize"])
def test_packed_decodes_like_the_simulated_storage(attn: str) -> None:
    """Dequantized entries equal those of the simulated (fake-quant) storage bit for bit, through every attention
    path: the transient dense copy (``sdpa``, ``eager``) and the packed format (``"economize"``)."""
    install()
    seq = _sequence()
    logits = {}
    for storage in ("simulated", "packed"):
        config = KIVIConfig(bits=2, group_size=4, residual_length=8, headroom=8, storage=storage)
        model = economize.apply(make_tiny_model(attn=attn), config)
        logits[storage], _ = _decode(model, seq)
        model.detach()
    torch.testing.assert_close(logits["packed"], logits["simulated"], atol=1e-5, rtol=0)


def test_packed_generation_matches_simulated_over_a_batch() -> None:
    prompts = [torch.randint(1, VOCAB, (n,), generator=torch.Generator().manual_seed(n)) for n in (12, 7, 20)]
    lengths = [24, 17, 30]
    tokens = {}
    for storage in ("simulated", "packed"):
        model = economize.apply(make_tiny_model(), KIVIConfig(bits=2, group_size=4, residual_length=8, storage=storage))
        tokens[storage] = [r.tokens for r in model.decode_batch(prompts, lengths)]
        model.detach()
    assert all(torch.equal(a, b) for a, b in zip(tokens["packed"], tokens["simulated"]))


@pytest.mark.parametrize("chunk", [1, 4])
def test_the_residual_is_exact_and_the_rest_is_quantized(chunk: int) -> None:
    """The first layer's entries (which depend only on the tokens) read back exactly for the last ``R`` committed
    tokens and as their fake-quantized values before."""
    residual, group = 8, 4
    seq = _sequence()
    base = make_tiny_model()
    plain = economize.apply(base)
    runner = plain.runner
    reference = runner.new_cache(SLOTS)
    _passes(runner, reference, seq, chunk)  # the same passes, so the same kernels produce the entries
    keys, values = (t[..., :TOTAL, :] for t in reference.read(0))
    plain.detach()
    model = economize.apply(base, KIVIConfig(bits=2, group_size=group, residual_length=residual, headroom=8))
    _, cache = _decode(model, seq, chunk=chunk)
    cache.pass_begin([TOTAL])  # commits the last pass: quantize what it pushed out of the residual
    kq, vq = cache.keys_quantized[0], cache.values_quantized[0]
    assert (kq, vq) == (TOTAL // residual * residual, TOTAL - residual)
    got_keys, got_values = (t[..., :TOTAL, :] for t in cache.read(0))
    assert torch.equal(got_keys[..., kq:, :], keys[..., kq:, :])
    assert torch.equal(got_values[..., vq:, :], values[..., vq:, :])
    assert not torch.equal(got_keys[..., :kq, :], keys[..., :kq, :])
    b, h = keys.shape[:2]
    # keys per channel in groups of `group` tokens, values per token in groups of `group` channels
    want_keys = fake_quantize(keys[..., :kq, :].reshape(b, h, kq // group, group, DIM), 2, dim=3)
    want_values = fake_quantize(values[..., :vq, :].reshape(b, h, vq, DIM // group, group), 2, dim=4)
    want_keys, want_values = want_keys.reshape(b, h, kq, DIM), want_values.reshape(b, h, vq, DIM)
    torch.testing.assert_close(got_keys[..., :kq, :], want_keys, atol=1e-6, rtol=0)
    torch.testing.assert_close(got_values[..., :vq, :], want_values, atol=1e-6, rtol=0)
    model.detach()


def test_a_pass_longer_than_the_headroom_is_refused() -> None:
    model = economize.apply(make_tiny_model(), KIVIConfig(bits=2, group_size=4, residual_length=4, headroom=2))
    with pytest.raises(ConfigError, match="headroom"):
        _decode(model, _sequence(), chunk=8)
    model.detach()


def test_scales_beyond_the_fp16_range_are_refused() -> None:
    base = make_tiny_model()
    base.model.layers[0].self_attn.k_proj.weight.mul_(1e7)
    model = economize.apply(base, KIVIConfig(bits=2, group_size=4, residual_length=4))
    with pytest.raises(ConfigError, match="simulated"):
        _decode(model, _sequence())
    model.detach()


def test_packed_storage_falls_back_beside_other_kv_techniques() -> None:
    """Another KV technique wraps the dense cache, so the default storage falls back to simulated; asking for the
    packed one explicitly is refused."""
    evict = KVEvictConfig(recent=8, storage="masked")
    model = economize.apply(make_tiny_model(), KIVIConfig(), evict)
    assert model.techniques["kivi"].storage == "simulated"
    model.detach()
    with pytest.raises(ConfigError, match="storage"):
        economize.apply(make_tiny_model(), KIVIConfig(storage="packed"), evict)


def test_packed_storage_is_reported_real_and_decodes_eagerly() -> None:
    model = economize.apply(make_tiny_model(), KIVIConfig())
    technique = model.techniques["kivi"]
    assert technique.storage == "packed" and technique.simulation() is None and not technique.graph_safe
    model.detach()
    simulated = economize.apply(make_tiny_model(), KIVIConfig(storage="simulated"))
    assert simulated.techniques["kivi"].simulation() is not None and simulated.techniques["kivi"].graph_safe
    simulated.detach()


def test_fused_kivi_keeps_its_own_storage() -> None:
    with pytest.raises(ConfigError):
        FusedKIVIConfig(storage="packed")
    assert FusedKIVIConfig().kivi_storage == "simulated"
