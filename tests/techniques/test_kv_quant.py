"""KV-cache quantization: KIVI (2/4-bit, keys per channel, values per token, full-precision residual) and fp8."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import economize
from economize import DecodeConfig
from economize.errors import ConfigError, IncompatibleTechniquesError
from economize.runtime import phase
from economize.techniques.kv_evict import KVEvictConfig
from economize.techniques.kv_quant import FP8KVConfig, KIVIConfig, fake_quantize, fp8_round_trip, kivi_bytes
from helpers import VOCAB, make_tiny_model, reference_greedy

PREFIX, TOTAL = 12, 40


def _sequence(seed: int = 0, length: int = TOTAL) -> torch.Tensor:
    return torch.randint(0, VOCAB, (1, length), generator=torch.Generator().manual_seed(seed))


def _prompts() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in (12, 7, 20)]


def _teacher_forced(model: economize.EfficientModel, name: str, seq: torch.Tensor) -> torch.Tensor:
    """Logits of ``seq`` from a prefill of :data:`PREFIX` tokens and single-token decode steps through the cache
    of technique ``name``."""
    runner = model.runner
    cache = model.techniques[name].wrap_cache(runner.new_cache(64))
    ctx = runner.context(cache, torch.arange(PREFIX)[None], trusted=True)
    with phase("prefill"):
        out = [runner.logits(runner.forward(seq[:, :PREFIX], ctx).final)[0]]
    for t in range(PREFIX, seq.shape[1]):
        ctx = runner.context(cache, torch.tensor([[t]]), trusted=True)
        with phase("decode"):
            out.append(runner.logits(runner.forward(seq[:, t : t + 1], ctx).final)[0])
    return torch.cat(out).float()


def _kl(logits: torch.Tensor, reference: torch.Tensor) -> float:
    return F.kl_div(logits.log_softmax(-1), reference.log_softmax(-1), log_target=True, reduction="batchmean").item()


def _quality(config: object, seq: torch.Tensor) -> tuple[float, float]:
    """``(max |logit difference|, mean KL)`` of teacher-forced decoding under ``config`` against the HF model."""
    base = make_tiny_model()
    reference = base(seq).logits[0].float()
    model = economize.apply(base, config)
    logits = _teacher_forced(model, config.technique, seq)
    model.detach()
    return (logits - reference).abs().max().item(), _kl(logits, reference)


def _identity_jump(base: torch.nn.Module, *configs: object, cycle_length: int = 4) -> economize.EfficientModel:
    """``base`` with a jump drafter from layer 0 to layer 0 whose predictor returns the next token's embedding,
    so drafts are good but not perfect, plus ``configs``."""
    jump = economize.JumpConfig(target_layer=0, source_layer=0, adaptation="frozen", lora_rank=2)
    model = economize.apply(base, jump, *configs, decoding=DecodeConfig(cycle_length=cycle_length))
    for name, parameter in model.techniques["jump"].predictor.named_parameters():
        if name != "out_gain":
            parameter.zero_()
    return model


# ---- numerics ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_fake_quantize_levels_and_error(bits: int) -> None:
    x = torch.randn(3, 5, 64, generator=torch.Generator().manual_seed(0))
    q = fake_quantize(x, bits, dim=-1)
    assert q.shape == x.shape and q.dtype == x.dtype
    for row in q.reshape(-1, 64):
        assert row.unique().numel() <= 2**bits
    step = (x.amax(-1, keepdim=True) - x.amin(-1, keepdim=True)) / (2**bits - 1)
    assert ((q - x).abs() <= step / 2 + 1e-3).all()
    assert torch.allclose(q.amin(-1), x.amin(-1), atol=1e-3)


def test_fake_quantize_of_a_constant_group_is_exact() -> None:
    x = torch.full((2, 8), 0.25)
    assert torch.equal(fake_quantize(x, 2, dim=-1), x)


def test_fake_quantize_more_bits_less_error() -> None:
    x = torch.randn(64, 32, generator=torch.Generator().manual_seed(1))
    errors = [(fake_quantize(x, bits, dim=0) - x).pow(2).mean().item() for bits in (2, 4, 8)]
    assert errors[0] > errors[1] > errors[2]


def test_kivi_bytes() -> None:
    config = KIVIConfig(bits=2, group_size=4, residual_length=8)
    heads, dim, element = 2, 16, 4
    assert kivi_bytes(10, 0, 0, heads, dim, element, config) == 2 * 10 * heads * dim * element
    # 8 keys and 2 values quantized: 2 bits per element, one fp16 scale and zero point per key group of 4 tokens
    # and per value group of 4 channels.
    keys = 8 * heads * dim / 4 + 2 * heads * dim * 4
    values = 2 * heads * dim / 4 + 2 * heads * (dim // 4) * 4
    full = (2 + 8) * heads * dim * element
    assert kivi_bytes(10, 8, 2, heads, dim, element, config) == keys + values + full


@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_fp8_round_trip(fmt: str) -> None:
    x = torch.randn(1, 2, 7, 16, generator=torch.Generator().manual_seed(0))
    for scale in ("none", "token"):
        y = fp8_round_trip(x, fmt, scale)
        assert y.dtype == x.dtype and not torch.equal(x, y)
        assert torch.allclose(x, y, rtol=0.13, atol=1e-2)
    assert (fp8_round_trip(x, "e4m3") - x).abs().mean() < (fp8_round_trip(x, "e5m2") - x).abs().mean()


# ---- configuration -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs", [dict(bits=3), dict(group_size=0), dict(residual_length=0), dict(group_size=32, residual_length=48)]
)
def test_kivi_config_validation(kwargs: dict) -> None:
    with pytest.raises(ConfigError):
        KIVIConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [dict(format="e3m4"), dict(scale="channel")])
def test_fp8_config_validation(kwargs: dict) -> None:
    with pytest.raises(ConfigError):
        FP8KVConfig(**kwargs)


def test_kivi_and_fp8_are_incompatible() -> None:
    with pytest.raises(IncompatibleTechniquesError):
        economize.apply(make_tiny_model(), KIVIConfig(), FP8KVConfig())


def test_layers_out_of_range() -> None:
    with pytest.raises(ConfigError):
        economize.apply(make_tiny_model(), KIVIConfig(layers=(7,)))


# ---- exactness of the trivial setting and the effect of the mechanisms ---------------------------------------------


def test_kivi_with_a_long_residual_is_exact() -> None:
    seq = _sequence()
    base = make_tiny_model()
    plain = economize.apply(base)
    expected = _teacher_forced_plain(plain, seq)
    plain.detach()
    model = economize.apply(base, KIVIConfig(bits=2, group_size=4, residual_length=1024))
    assert torch.equal(_teacher_forced(model, "kivi", seq), expected)
    assert torch.equal(model.decode(seq[0, :PREFIX], 16).tokens, reference_greedy(base, seq[0, :PREFIX], 16))


def _teacher_forced_plain(model: economize.EfficientModel, seq: torch.Tensor) -> torch.Tensor:
    runner = model.runner
    cache = runner.new_cache(64)
    ctx = runner.context(cache, torch.arange(PREFIX)[None], trusted=True)
    out = [runner.logits(runner.forward(seq[:, :PREFIX], ctx).final)[0]]
    for t in range(PREFIX, seq.shape[1]):
        ctx = runner.context(cache, torch.tensor([[t]]), trusted=True)
        out.append(runner.logits(runner.forward(seq[:, t : t + 1], ctx).final)[0])
    return torch.cat(out).float()


def test_quantization_error_orderings() -> None:
    seq = _sequence()
    _, kivi2 = _quality(KIVIConfig(bits=2, group_size=4, residual_length=8), seq)
    _, kivi4 = _quality(KIVIConfig(bits=4, group_size=4, residual_length=8), seq)
    _, kivi8 = _quality(KIVIConfig(bits=8, group_size=4, residual_length=8), seq)
    _, e4m3 = _quality(FP8KVConfig(), seq)
    _, token = _quality(FP8KVConfig(scale="token"), seq)
    _, e5m2 = _quality(FP8KVConfig(format="e5m2"), seq)
    assert kivi2 > kivi4 > kivi8 > 0
    assert e5m2 > e4m3 > token > 0
    assert kivi2 > 0.05 and e4m3 < 0.1


def test_kivi_layers_restrict_quantization() -> None:
    seq = _sequence()
    _, every = _quality(KIVIConfig(bits=2, group_size=4, residual_length=8), seq)
    _, some = _quality(KIVIConfig(bits=2, group_size=4, residual_length=8, layers=(6,)), seq)
    assert 0 < some < every


# ---- memory accounting ---------------------------------------------------------------------------------------------


def test_kivi_bytes_in_the_cost_summary() -> None:
    model = economize.apply(make_tiny_model(), KIVIConfig(bits=2, group_size=4, residual_length=8))
    result = model.decode(_sequence()[0, :PREFIX], 20)
    summary = model.techniques["kivi"].cost_summary(result)
    assert {"layer_tokens", "kv_bytes", "kv_bytes_full", "kv_bytes_read", "kv_compression"} <= summary.keys()
    # the last exact pass (the 20th) sees 12 + 18 = 30 committed tokens: 24 keys and 22 values are quantized
    # (6 layers, 2 KV heads of width 16, fp32)
    config = model.techniques["kivi"].config
    assert summary["kv_bytes"] == 6 * kivi_bytes(30, 24, 22, 2, 16, 4, config)
    assert summary["kv_bytes_full"] == 6 * 2 * 30 * 2 * 16 * 4
    assert summary["kv_compression"] > 1.5
    assert summary["kv_bytes_read"] < summary["kv_bytes_read_full"]
    assert summary["kv_passes"] == 20


def test_fp8_halves_the_bytes() -> None:
    for scale, ratio in (("none", 4.0), ("token", 4.0 * 16 / (16 + 4))):
        model = economize.apply(make_tiny_model(), FP8KVConfig(scale=scale))
        model.decode(_sequence()[0, :PREFIX], 8)
        stats = model.techniques["kv_fp8"].kv_stats()
        assert stats["kv_compression"] == pytest.approx(ratio)  # fp32 cache: 4 bytes per element, 1 in fp8
        model.detach()


def test_fp8_tensor_scales_are_calibrated_absmax(calib) -> None:
    from economize.core.calibration import token_batches
    from economize.techniques.kv_quant.fp8 import calibrate_kv_absmax

    model = economize.apply(make_tiny_model(), FP8KVConfig(scale="tensor"), calibration=calib)
    technique = model.techniques["kv_fp8"]
    assert technique.calibrated
    keys, values = calibrate_kv_absmax(model.adapter, token_batches(calib))
    assert torch.allclose(technique.key_scale, keys / torch.finfo(torch.float8_e4m3fn).max)
    assert torch.allclose(technique.value_scale, values / torch.finfo(torch.float8_e4m3fn).max)
    assert (technique.key_scale > 0).all() and (technique.value_scale > 0).all()
    model.detach()


def test_fp8_tensor_scales_need_calibration_data() -> None:
    with pytest.raises(ConfigError):
        economize.apply(make_tiny_model(), FP8KVConfig(scale="tensor"))


def test_fp8_tensor_scales_decode_close_to_the_model(calib) -> None:
    seq = _sequence()
    base = make_tiny_model()
    reference = base(seq).logits[0].float()
    model = economize.apply(base, FP8KVConfig(scale="tensor"), calibration=calib)
    tensor = _kl(_teacher_forced(model, "kv_fp8", seq), reference)
    _, none = _quality(FP8KVConfig(), seq)
    _, token = _quality(FP8KVConfig(scale="token"), seq)
    assert token < tensor < none < 0.1  # one calibrated scale per layer lies between none and per-token scales
    model.decode(seq[0, :PREFIX], 8)
    assert model.techniques["kv_fp8"].kv_stats()["kv_compression"] == pytest.approx(4.0)  # no per-token scales
    model.detach()


def test_fp8_tensor_scales_load_without_calibration(calib, tmp_path) -> None:
    model = economize.apply(make_tiny_model(), FP8KVConfig(scale="tensor"), calibration=calib)
    economize.save(model, tmp_path)
    expected = model.techniques["kv_fp8"].key_scale.clone()
    model.detach()
    loaded = economize.load(make_tiny_model(), tmp_path)  # no calibration data: the saved scales are used
    assert torch.equal(loaded.techniques["kv_fp8"].key_scale, expected)
    loaded.detach()


# ---- composition ---------------------------------------------------------------------------------------------------


def test_fp8_under_jump_spec_equals_its_own_greedy() -> None:
    prompts, lengths = _prompts(), [24, 17, 30]
    plain = economize.apply(make_tiny_model(), FP8KVConfig(), decoding=DecodeConfig(cycle_length=1))
    expected = [plain.decode(p, n).tokens for p, n in zip(prompts, lengths)]
    plain.detach()
    model = _identity_jump(make_tiny_model(), FP8KVConfig())
    results = model.decode_batch(prompts, lengths)
    for result, tokens in zip(results, expected):
        assert torch.equal(result.tokens, tokens)
        assert sum(drafted for drafted, _ in result.acceptance) > 0
    stats = [model.techniques["kv_fp8"].kv_stats(row) for row in range(3)]
    assert all(s["kv_compression"] == pytest.approx(4.0) for s in stats)
    assert stats[2]["kv_bytes"] > stats[0]["kv_bytes"] > stats[1]["kv_bytes"]  # rows keep their own lengths


def test_trivial_kivi_with_jump_and_batches_is_exact() -> None:
    base = make_tiny_model()
    prompts, lengths = _prompts(), [24, 17, 30]
    model = _identity_jump(base, KIVIConfig(bits=2, group_size=4, residual_length=1024))
    for prompt, length, result in zip(prompts, lengths, model.decode_batch(prompts, lengths)):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, length))


def test_kivi_quantizes_the_prompt_at_the_end_of_its_pass() -> None:
    """The prompt pass attends over the dense prompt and leaves it quantized, so the first decode pass finds only
    its own new entries to quantize (as KIVI quantizes the prompt at prefill)."""
    base = make_tiny_model()
    seq = _sequence()
    model = economize.apply(base, KIVIConfig(bits=2, group_size=4, residual_length=8))
    runner = model.runner
    cache = model.techniques["kivi"].wrap_cache(runner.new_cache(64))
    with phase("prefill"):
        ctx = runner.context(cache, torch.arange(PREFIX)[None], trusted=True)
        logits = runner.logits(runner.forward(seq[:, :PREFIX], ctx).final)[0].float()
    assert (cache.keys_quantized, cache.values_quantized) == ([8], [PREFIX - 8])
    torch.testing.assert_close(logits, base(seq[:, :PREFIX]).logits[0].float(), atol=1e-4, rtol=0)


def test_kivi_with_jump_quantizes_committed_entries_only() -> None:
    """Draft K/V stay full precision and are rolled back; the accounting counts committed entries per row."""
    base = make_tiny_model()
    prompts, lengths = _prompts(), [24, 17, 30]
    config = KIVIConfig(bits=2, group_size=4, residual_length=8)
    single = economize.apply(base, config, decoding=DecodeConfig(cycle_length=1))
    expected = [single.decode(p, n).tokens for p, n in zip(prompts, lengths)]
    single.detach()
    model = _identity_jump(base, config, cycle_length=1)
    assert all(torch.equal(r.tokens, e) for r, e in zip(model.decode_batch(prompts, lengths), expected))
    model.detach()
    model = _identity_jump(base, config)
    results = model.decode_batch(prompts, lengths)
    technique = model.techniques["kivi"]
    for row, (prompt, result) in enumerate(zip(prompts, results)):
        assert len(result.tokens) == lengths[row]
        stats = technique.kv_stats(row)
        assert 1.5 < stats["kv_compression"] < 16
        assert stats["kv_bytes_full"] <= 6 * 2 * (prompt.numel() + lengths[row]) * 2 * 16 * 4


def test_kv_quant_with_prune_and_lowrank(calib) -> None:
    from economize.techniques.lowrank import LowRankConfig
    from economize.techniques.prune import PruneConfig

    prompt = _prompts()[0]
    for weights in (PruneConfig(method="wanda", sparsity="2:4"), LowRankConfig(method="svd", compression=0.3)):
        compressed = economize.apply(make_tiny_model(), weights, calibration=calib)
        expected = compressed.decode(prompt, 20).tokens
        exact = economize.apply(
            make_tiny_model(), weights, KIVIConfig(group_size=4, residual_length=1024), calibration=calib
        )
        assert torch.equal(exact.decode(prompt, 20).tokens, expected)
        lossy = economize.apply(make_tiny_model(), weights, FP8KVConfig(format="e5m2"), calibration=calib)
        result = lossy.decode(prompt, 20)
        assert lossy.techniques["kv_fp8"].kv_stats()["kv_compression"] == pytest.approx(4.0)
        assert len(result.tokens) == 20


# ---- KIVI's fp16 scales, padding accounting and the cache wrapper ----
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fake_quantize_beyond_the_fp16_range_is_finite(dtype: torch.dtype) -> None:
    x = (torch.randn(4, 64, generator=torch.Generator().manual_seed(0)) * 3e5).to(dtype)
    x[0, 0] = 1e6
    q = fake_quantize(x, 4, dim=-1)
    assert torch.isfinite(q).all()
    step = (x.float().amax(-1, keepdim=True) - x.float().amin(-1, keepdim=True)) / 15
    assert ((q.float() - x.float()).abs() <= step / 2 * 1.02 + 1e-3 * x.float().abs()).all()


def test_fake_quantize_in_the_fp16_range_rounds_scales_to_fp16() -> None:
    """Unchanged numerics within the fp16 range: scale and zero point are the fp16-rounded ones."""
    x = torch.randn(3, 32, generator=torch.Generator().manual_seed(2)) * 7
    low, high = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    scale, zero = ((high - low) / 3).half().float(), low.half().float()
    expected = (((x - zero) / scale).round().clamp(0, 3) * scale + zero)
    assert torch.equal(fake_quantize(x, 2, dim=-1), expected)


def test_kivi_cache_with_huge_keys_stays_finite() -> None:
    base = make_tiny_model()
    model = economize.apply(base, KIVIConfig(bits=2, group_size=4, residual_length=4))
    runner = model.runner
    cache = model.techniques["kivi"].wrap_cache(runner.new_cache(64))
    seq = torch.randint(0, VOCAB, (1, PREFIX), generator=torch.Generator().manual_seed(0))
    with phase("prefill"):
        runner.forward(seq, runner.context(cache, torch.arange(PREFIX)[None], trusted=True))
    base_cache = cache.inner
    for layer_idx in base_cache.keys:
        base_cache.keys[layer_idx][..., :PREFIX, 0] = 2e5
    cache.pass_begin([PREFIX])
    for layer_idx in base_cache.keys:
        keys, values = base_cache.read(layer_idx)
        assert torch.isfinite(keys).all() and torch.isfinite(values).all()
    model.detach()


@pytest.mark.parametrize(
    "config",
    [KIVIConfig(bits=2, group_size=4, residual_length=4), FP8KVConfig(), KVEvictConfig(recent=4)],
    ids=["kivi", "fp8", "kv_evict"],
)
def test_padding_only_rows_are_not_accounted(config: object) -> None:
    model = economize.apply(make_tiny_model(), config)
    runner = model.runner
    technique = model.techniques[config.technique]  # type: ignore[attr-defined]
    cache = technique.wrap_cache(runner.new_cache(64, batch_size=2))
    seq = torch.randint(0, VOCAB, (2, PREFIX), generator=torch.Generator().manual_seed(1))
    with phase("prefill"):
        runner.forward(seq, runner.context(cache, torch.arange(PREFIX)[None].expand(2, -1), trusted=True))
    with phase("decode"):  # a pass over committed entries, so the rows hold accounted bytes
        runner.forward(seq[:, :1], runner.context(cache, torch.full((2, 1), PREFIX), trusted=True))
    before = [cache.stats.row(b).as_dict() for b in range(2)]
    assert all(row["kv_passes"] == 2 and row["kv_bytes"] > 0 for row in before)
    slots = torch.tensor([[PREFIX + 1], [cache.trash_slot]])  # row 1 only writes padding: its frontier is None
    with phase("decode"), cache.writing_at(slots):
        pass
    assert cache.stats.row(0).passes == 3
    assert cache.stats.row(1).as_dict() == before[1]
    model.detach()


@pytest.mark.parametrize(
    "config",
    [KIVIConfig(bits=2, group_size=4, residual_length=4), FP8KVConfig(), KVEvictConfig(recent=4)],
    ids=["kivi", "fp8", "kv_evict"],
)
def test_wrapper_exposes_the_buffers_and_length_of_the_wrapped_cache(config: object) -> None:
    """``keys``, ``values`` and ``get_seq_length`` are the wrapped cache's, so code written for a plain cache (the RL
    teacher's ``detached_cache``, the ``generate`` integration) works on a wrapped one."""
    model = economize.apply(make_tiny_model(), config)
    runner = model.runner
    inner = runner.new_cache(64)
    cache = model.techniques[config.technique].wrap_cache(inner)  # type: ignore[attr-defined]
    seq = torch.randint(0, VOCAB, (1, PREFIX), generator=torch.Generator().manual_seed(4))
    with phase("prefill"):
        runner.forward(seq, runner.context(cache, torch.arange(PREFIX)[None], trusted=True))
    assert cache.keys is inner.keys and cache.values is inner.values and len(cache.keys) > 0
    assert hasattr(cache, "get_seq_length") == hasattr(inner, "get_seq_length")
    inner.get_seq_length = lambda layer_idx=0: 7  # type: ignore[attr-defined]  # a cache that knows its length
    assert cache.get_seq_length() == 7
    model.detach()


def test_cache_wrapper_accounts_covered_layers_through_the_hook_and_the_rest_dense() -> None:
    """``CacheWrapper.account`` sums ``covered_bytes`` over the layers the wrapper covers and dense bytes over the
    others, and skips padding rows (frontier ``None``)."""
    from economize.techniques._kv import CacheWrapper

    model = economize.apply(make_tiny_model(), KIVIConfig())  # any technique: only its runner is used
    runner = model.runner
    inner = runner.new_cache(64, batch_size=2)
    seq = torch.randint(0, VOCAB, (2, PREFIX), generator=torch.Generator().manual_seed(5))
    with phase("prefill"):
        runner.forward(seq, runner.context(inner, torch.arange(PREFIX)[None].expand(2, -1), trusted=True))
    model.detach()

    class Half(CacheWrapper):
        layers = frozenset({0})

    wrapper = Half(inner)
    buffers = wrapper.positional_buffers()
    assert list(buffers) == sorted(buffers) and len(buffers) > 1
    assert wrapper.covers(0) and not wrapper.covers(1)
    wrapper.account([10, None], lambda i, keys, committed: (committed * 0 + 3.0, committed * 0 + 7.0))
    dense = sum(float(CacheWrapper.dense_bytes(k, torch.tensor([10.0], dtype=torch.float64))) for i, k in
                buffers.items() if i != 0)
    row = wrapper.stats.row(0)
    assert (row.kv_bytes, row.kv_bytes_full, row.passes) == (3.0 + dense, 7.0 + dense, 1)
    assert 1 not in wrapper.stats.rows
