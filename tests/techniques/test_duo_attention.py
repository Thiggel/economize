"""DuoAttention: retrieval heads keep the whole cache, streaming heads a StreamingLLM window (sinks and recent
entries), chosen per KV head from trained gate values in the reference's file format."""

from __future__ import annotations

import json

import pytest
import torch

import economize
from economize import DecodeConfig
from economize.errors import ConfigError, IncompatibleTechniquesError
from economize.runtime import phase
from economize.techniques.duo_attention import (
    PATTERN_FILE,
    DuoAttentionCache,
    DuoAttentionConfig,
    full_attention_heads,
    read_pattern,
    read_scores,
)
from economize.techniques.kv_evict import KVEvictConfig
from economize.techniques.prune import DepthPruneConfig
from helpers import VOCAB, make_tiny_model

PREFIX, TOTAL = 16, 48
SINKS, RECENT = 3, 8
#: Bytes of one token's K and V for one KV head of one layer of the tiny model (head width 16, fp32).
HEAD_BYTES = 2 * 16 * 4
#: Gate values of the tiny model (6 layers, 2 KV heads): with threshold 0.5, layer 3 keeps both heads, layer 4 none.
SCORES = ((0.9, 0.1), (0.2, 0.8), (0.6, 0.7), (0.3, 0.4), (0.95, 0.05), (0.15, 0.85))
STREAMING = {0: (False, True), 1: (True, False), 3: (True, True), 4: (False, True), 5: (True, False)}


def _sequence(seed: int = 0) -> torch.Tensor:
    return torch.randint(0, VOCAB, (1, TOTAL), generator=torch.Generator().manual_seed(seed))


def _prompts() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in (12, 7, 20)]


def _config(**fields: object) -> DuoAttentionConfig:
    settings: dict[str, object] = {"pattern": SCORES, "sparsity": None, "threshold": 0.5, "sinks": SINKS}
    return DuoAttentionConfig(**{**settings, "recent": RECENT, **fields})


def _teacher_forced(model: economize.EfficientModel, seq: torch.Tensor, name: str) -> tuple[torch.Tensor, object]:
    """Logits of ``seq`` from a prefill of :data:`PREFIX` tokens and single-token decode steps, and the cache."""
    runner = model.runner
    cache = model.techniques[name].wrap_cache(runner.new_cache(64))
    ctx = runner.context(cache, torch.arange(PREFIX)[None], trusted=True)
    with phase("prefill"):
        out = [runner.logits(runner.forward(seq[:, :PREFIX], ctx).final)[0]]
    for t in range(PREFIX, seq.shape[1]):
        ctx = runner.context(cache, torch.tensor([[t]]), trusted=True)
        with phase("decode"):
            out.append(runner.logits(runner.forward(seq[:, t : t + 1], ctx).final)[0])
    return torch.cat(out).float(), cache


def _identity_jump(base: torch.nn.Module, *configs: object, cycle_length: int = 4) -> economize.EfficientModel:
    jump = economize.JumpConfig(target_layer=0, source_layer=0, adaptation="frozen", lora_rank=2)
    model = economize.apply(base, jump, *configs, decoding=DecodeConfig(cycle_length=cycle_length))
    for name, parameter in model.techniques["jump"].predictor.named_parameters():
        if name != "out_gain":
            parameter.zero_()
    return model


# ---- the reference's head selection and files ----
def test_full_attention_heads_is_the_reference_quantile_threshold() -> None:
    """``sparsify_attention_heads`` of the reference: heads at or above the ``sparsity`` quantile (linear
    interpolation) of all gate values keep full attention."""
    generator = torch.Generator().manual_seed(0)
    scores = torch.rand(8, 4, generator=generator, dtype=torch.float64)
    for sparsity in (0.25, 0.5, 0.6, 0.9):
        sorted_values = scores.flatten().sort().values
        position = sparsity * (sorted_values.numel() - 1)  # numpy.quantile's default ("linear")
        low = int(position)
        cut = sorted_values[low] + (position - low) * (sorted_values[min(low + 1, 31)] - sorted_values[low])
        full = torch.tensor(full_attention_heads(scores.tolist(), sparsity=sparsity))
        assert torch.equal(full, scores >= cut)
    assert all(all(row) for row in full_attention_heads(scores.tolist(), sparsity=0.0))
    assert not any(any(row) for row in full_attention_heads(scores.tolist(), sparsity=1.0))
    assert full_attention_heads([[0.2, 0.5], [0.7]], threshold=0.5) == [[False, True], [True]]
    with pytest.raises(ConfigError, match="sparsity or a threshold"):
        full_attention_heads([[0.5]])


def test_reads_the_reference_pattern_directory(tmp_path) -> None:
    (tmp_path / PATTERN_FILE).write_text("0.5\t1.7\n\n-0.2\t0.25\t0.75\n")
    (tmp_path / "config.json").write_text(json.dumps({"sink_size": 32, "recent_size": 128, "sparsity": 0.5}))
    assert read_scores(tmp_path) == read_scores(tmp_path / PATTERN_FILE) == ((0.5, 1.0), (0.0, 0.25, 0.75))
    files = read_pattern(tmp_path)
    assert (files.sinks, files.recent) == (32, 128)
    config = DuoAttentionConfig.from_pattern(tmp_path, recent=64)
    assert (config.pattern, config.sinks, config.recent) == (str(tmp_path), 32, 64)
    assert config.scores() == files.scores
    assert read_pattern(tmp_path / PATTERN_FILE).sinks is None  # a bare file has no config.json next to it
    (tmp_path / "bad.tsv").write_text("0.5 x\n")
    with pytest.raises(ConfigError, match="numbers"):
        read_scores(tmp_path / "bad.tsv")
    with pytest.raises(ConfigError, match="neither"):
        read_scores(tmp_path / "missing")


def test_config_validation() -> None:
    with pytest.raises(ConfigError, match="streaming"):
        DuoAttentionConfig(pattern=SCORES, method="h2o")
    with pytest.raises(ConfigError, match="head pattern"):
        DuoAttentionConfig()
    with pytest.raises(ConfigError, match="either a sparsity or a threshold"):
        DuoAttentionConfig(pattern=SCORES, threshold=0.5)
    with pytest.raises(ConfigError, match="either a sparsity or a threshold"):
        DuoAttentionConfig(pattern=SCORES, sparsity=None)
    with pytest.raises(ConfigError, match="sparsity"):
        DuoAttentionConfig(pattern=SCORES, sparsity=1.5)
    with pytest.raises(ConfigError, match="sinks"):
        DuoAttentionConfig(pattern=SCORES, sinks=None)
    with pytest.raises(ConfigError, match="rows of gate values"):
        DuoAttentionConfig(pattern=((0.5, "x"),))  # type: ignore[arg-type]
    assert DuoAttentionConfig(pattern=SCORES).scores() == SCORES
    assert DuoAttentionConfig(pattern=((1.5, -1.0),)).scores() == ((1.0, 0.0),)


def test_pattern_shape_must_match_the_model() -> None:
    with pytest.raises(ConfigError, match="5 layers, the model 6"):
        economize.apply(make_tiny_model(), _config(pattern=SCORES[:5]))
    rows = (*SCORES[:2], (0.5, 0.5, 0.5), *SCORES[3:])
    with pytest.raises(ConfigError, match="3 values for layer 3, which has 2 KV heads"):
        economize.apply(make_tiny_model(), _config(pattern=rows))


def test_config_round_trips_through_dicts() -> None:
    config = _config()
    data = json.loads(json.dumps(config.to_dict()))
    assert DuoAttentionConfig.from_dict(data) == config
    assert economize.TECHNIQUES.get("duo_attention").config_class is DuoAttentionConfig


# ---- decoding ----
def test_streaming_heads_follow_the_threshold_and_layers() -> None:
    model = economize.apply(make_tiny_model(), _config())
    technique = model.techniques["duo_attention"]
    assert technique.streaming == STREAMING
    assert technique.describe()["streaming_heads"] == 6 and technique.describe()["kv_heads"] == 12
    restricted = economize.apply(make_tiny_model(), _config(layers=(2, 4)))
    assert restricted.techniques["duo_attention"].streaming == {1: STREAMING[1], 3: STREAMING[3]}
    # The sparsity counts only the covered layers' heads: half of layers 1 and 2's four heads stream.
    half = economize.apply(make_tiny_model(), _config(sparsity=0.5, threshold=None, layers=(1, 2)))
    assert half.techniques["duo_attention"].streaming == {0: (False, True), 1: (True, False)}


def _reference_logits(seq: torch.Tensor, monkeypatch: pytest.MonkeyPatch) -> torch.Tensor:
    """Logits of the reference's evaluation semantics, transcribed independently of the cache machinery: the prompt
    attends causally in full; at decode position ``t`` a streaming head sees the first :data:`SINKS` entries, the
    last :data:`RECENT` of the ``t`` committed entries and its own; a retrieval head sees everything. One forward
    over the whole sequence with this per-head mask added to eager attention."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.eager_attention_forward
    group = 2  # 4 query heads over 2 KV heads

    def duo_attention_forward(module, query, key, value, attention_mask, **kwargs):
        streaming = STREAMING.get(module.layer_idx, (False, False))
        heads = torch.tensor([s for s in streaming for _ in range(group)])
        length = query.shape[2]
        t = torch.arange(length)[:, None]
        j = torch.arange(key.shape[2])[None, :]
        evicted = (t >= PREFIX) & (j >= SINKS) & (j < t - RECENT)
        hidden = heads[:, None, None] & evicted[None]
        causal = j > t
        blocked = (hidden | causal[None])[None]
        bias = torch.zeros(blocked.shape, dtype=query.dtype).masked_fill(blocked, torch.finfo(query.dtype).min)
        return original(module, query, key, value, bias, **kwargs)

    base = make_tiny_model(attn="eager")
    monkeypatch.setattr(modeling_qwen2, "eager_attention_forward", duo_attention_forward)
    with torch.no_grad():
        logits = base(seq).logits[0].float()
    monkeypatch.setattr(modeling_qwen2, "eager_attention_forward", original)
    return logits


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_decodes_like_the_reference_semantics(attn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    seq = _sequence()
    reference = _reference_logits(seq, monkeypatch)
    base = make_tiny_model(attn=attn)
    plain = base(seq).logits[0].float()
    logits, cache = _teacher_forced(economize.apply(base, _config()), seq, "duo_attention")
    assert isinstance(cache, DuoAttentionCache)
    torch.testing.assert_close(logits, reference, atol=1e-4, rtol=1e-4)
    assert not torch.allclose(logits[PREFIX + RECENT + 1 :], plain[PREFIX + RECENT + 1 :], atol=1e-3)
    torch.testing.assert_close(logits[:PREFIX], plain[:PREFIX], atol=1e-4, rtol=1e-4)  # the prompt is exact


def test_kept_entries_and_bytes() -> None:
    model = economize.apply(make_tiny_model(), _config())
    _, cache = _teacher_forced(model, _sequence(), "duo_attention")
    committed = TOTAL - 1
    for layer in range(6):
        kept = cache.kept(layer)
        if layer not in STREAMING:
            assert kept is None
            continue
        for head, streaming in enumerate(STREAMING[layer]):
            row = kept[0, head, :committed]
            if streaming:
                assert row[:SINKS].all() and row[-RECENT:].all() and not row[SINKS:-RECENT].any()
            else:
                assert row.all()
    stats = model.techniques["duo_attention"].kv_stats()
    assert stats["kv_bytes_full"] == committed * 12 * HEAD_BYTES
    hidden = committed - SINKS - RECENT
    assert stats["kv_bytes"] == stats["kv_bytes_full"] - hidden * 6 * HEAD_BYTES


def test_all_streaming_is_streaming_llm_and_none_is_the_model() -> None:
    seq = _sequence()
    streaming = economize.apply(make_tiny_model(), KVEvictConfig(recent=RECENT, sinks=SINKS))
    expected, _ = _teacher_forced(streaming, seq, "kv_evict")
    model = economize.apply(make_tiny_model(), _config(sparsity=1.0, threshold=None))
    logits, _ = _teacher_forced(model, seq, "duo_attention")
    assert torch.equal(logits, expected)
    assert model.techniques["duo_attention"].kv_stats() == streaming.techniques["kv_evict"].kv_stats()
    base = make_tiny_model()
    exact = economize.apply(base, _config(sparsity=0.0, threshold=None))
    logits, _ = _teacher_forced(exact, seq, "duo_attention")
    torch.testing.assert_close(logits, base(seq).logits[0].float(), atol=1e-4, rtol=1e-4)
    assert exact.techniques["duo_attention"].streaming == {}


@pytest.mark.parametrize("cycle", [1, 4])
def test_batched_decoding_matches_single_rows(cycle: int) -> None:
    """Rows of different lengths in one batch decode like each row alone, with and without drafting."""
    base = make_tiny_model()
    model = economize.apply(base, _config()) if cycle == 1 else _identity_jump(base, _config(), cycle_length=cycle)
    lengths = [24, 17, 30]
    batched = [r.tokens.tolist() for r in model.decode_batch(_prompts(), lengths)]
    single = [model.decode(p, n).tokens.tolist() for p, n in zip(_prompts(), lengths, strict=True)]
    assert batched == single


def test_clashes_with_other_kv_eviction() -> None:
    with pytest.raises(IncompatibleTechniquesError):
        economize.apply(make_tiny_model(), _config(), KVEvictConfig(recent=8))
    info = economize.TECHNIQUES.get("duo_attention").info
    assert info.family == economize.TECHNIQUES.get("kv_evict").info.family
    assert info.simulation(_config()) is not None


def test_refuses_depth_pruning() -> None:
    """The pattern addresses the unpruned model's layers, so depth pruning in the same recipe is refused."""
    assert not economize.compatible("depth_prune", "duo_attention")
    with pytest.raises(IncompatibleTechniquesError, match="rows dropped"):
        economize.apply(make_tiny_model(), DepthPruneConfig(layers=(3,)), _config())


@pytest.mark.gpu
def test_gpu_kernel_decodes_like_sdpa() -> None:
    """Under ``attn_implementation="economize"`` the decode kernel skips the tiles the streaming heads do not see
    and gives the tokens of the masked SDPA path."""
    from economize.attention import install, kernel_usage

    install()
    tokens = {}
    for attn in ("economize", "sdpa"):
        model = economize.apply(make_tiny_model(attn=attn).cuda(), _config())
        kernel_usage(reset=True)
        tokens[attn] = [r.tokens.tolist() for r in model.decode_batch([p.cuda() for p in _prompts()], [24, 17, 30])]
        if attn == "economize":
            assert "kernel" in {row["path"] for row in kernel_usage(reset=True)}
    assert tokens["economize"] == tokens["sdpa"]
