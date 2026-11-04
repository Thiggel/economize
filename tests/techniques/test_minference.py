"""MInference: every head of a long prompt attends over its A-shape, vertical-slash or block-sparse pattern,
checked against transcriptions of the official implementation (``minference`` 0.1.6.0)."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import economize
from economize.arch import adapter_for
from economize.attention import install, kernel_usage
from economize.attention.minference import (
    HeadPattern,
    MInferencePrefill,
    a_shape_index,
    block_sparse_index,
    convert_vertical_slash,
    load_patterns,
    mask_from_index,
    minference_index,
    sparse_prefill_reference,
    vertical_slash_lines,
)
from economize.attention.prefill import prefill_of
from economize.errors import ConfigError, IncompatibleTechniquesError
from economize.techniques.minference import MInferenceConfig, minference_layers
from helpers import VOCAB, make_family_model, make_tiny_model

# ---- transcriptions of the official implementation ----
# The functions below follow minference 0.1.6.0 (microsoft/MInference, MIT License, Copyright (c) Microsoft
# Corporation): minference/modules/minference_forward.py (sum_all_diagonal_matrix, vertical_and_slash_kernel),
# csrc/vertical_slash_index.cu (convert_vertical_slash_indexes_kernel, one CUDA thread per block row, transcribed as
# a Python loop) and minference/ops/block_sparse_flash_attention.py (block_sparse_attention, _build_block_index).


def _sum_all_diagonal_matrix(mat: torch.Tensor) -> torch.Tensor:
    b, h, n, m = mat.shape
    zero_mat = torch.zeros((b, h, n, n)).to(mat.device)
    mat_padded = torch.cat((zero_mat, mat, zero_mat), -1)
    mat_strided = mat_padded.as_strided((1, 1, n, n + m), (1, n * (2 * n + m), 2 * n + m + 1, 1))
    sum_diags = torch.sum(mat_strided, 2)
    return sum_diags[:, :, 1:]


def _official_lines(q: torch.Tensor, k: torch.Tensor, vertical_size: int, slash_size: int) -> tuple[list, list]:
    """``vertical_and_slash_kernel`` up to the sparse attention, for one head (``q``, ``k`` ``[1, 1, Q, D]``):
    the vertical indices and the slash offsets, sorted as ``vertical_slash_sparse_attention`` sorts them."""
    q_len, head_dim = q.shape[2], q.shape[3]
    arange = torch.arange(64)
    last_q_mask = arange[None, None, :, None] >= arange[None, None, None, :]
    vertical_size, slash_size = min(q_len, max(vertical_size, 30)), min(q_len, max(slash_size, 50))
    last_q = min(64, q_len)
    qk = torch.einsum("bhmk, bhnk -> bhmn", q[:, :, -last_q:, :], k) / math.sqrt(head_dim)
    qk[:, :, :, -last_q:] = torch.where(last_q_mask[..., -last_q:, -last_q:], qk[:, :, :, -last_q:], -torch.inf)
    qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
    vertical = qk.sum(-2, keepdim=True)
    vertical[..., :30] = torch.inf
    vertical_topk = torch.topk(vertical, vertical_size, -1).indices
    slash = _sum_all_diagonal_matrix(qk)[..., : -last_q + 1]
    slash[..., -100:] = torch.inf
    slash = (q_len - 1) - torch.topk(slash, slash_size, -1).indices
    v_idx = vertical_topk.reshape(-1).sort(descending=False).values.tolist()
    s_idx = slash.reshape(-1).sort(descending=True).values.tolist()
    return v_idx, s_idx


def _official_convert(v_idx: list, s_idx: list, seqlen: int, block: int = 64) -> list[tuple[list, list]]:
    """``convert_vertical_slash_indexes_kernel``: per block row, the key block offsets and the columns."""
    rows = []
    for block_idx_m in range(math.ceil(seqlen / block)):
        end_m = block_idx_m * block + block
        columns: list[int] = []
        offsets: list[int] = []
        v = s = 0
        v_i = v_idx[v]
        v += 1
        s_i = s_idx[s]
        s += 1
        while s_i >= end_m:
            s_i = s_idx[s]
            s += 1
        s_i = max(end_m - s_i, block)
        range_start, range_end = s_i - block, s_i
        while True:
            if v_i < range_end:
                if v_i < range_start:
                    columns.append(v_i)
                if v < len(v_idx):
                    v_i = v_idx[v]
                    v += 1
                else:
                    v_i = end_m + block
            else:
                if s < len(s_idx):
                    s_i = max(end_m - s_idx[s], block)
                    s += 1
                else:
                    offsets.extend(range(range_start, range_end, block))
                    break
                if s_i > range_end + block:
                    offsets.extend(range(range_start, range_end, block))
                    range_start, range_end = s_i - block, s_i
                elif s_i > range_end:
                    range_end += block
        rows.append((offsets, columns))
    return rows


def _official_block_index(query: torch.Tensor, key: torch.Tensor, top_k: int) -> torch.Tensor:
    """``block_sparse_attention``'s padding and ``_build_block_index`` (``[B, H, P, k]`` key blocks)."""
    pad = 64 - (query.shape[2] & 63)
    query = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0])
    key = torch.nn.functional.pad(key, [0, 0, 0, pad, 0, 0, 0, 0])
    batch_size, num_heads, context_size, head_dim = query.shape
    query_pool = query.reshape((batch_size, num_heads, -1, 64, head_dim)).mean(dim=-2)
    key_pool = key.reshape((batch_size, num_heads, -1, 64, head_dim)).mean(dim=-2)
    arange_m = torch.arange(query_pool.shape[-2], dtype=torch.int32) * 64
    arange_n = torch.arange(key_pool.shape[-2], dtype=torch.int32) * 64
    p_pool = torch.einsum("bhmk, bhnk -> bhmn", query_pool, key_pool)
    p_pool = p_pool.where(arange_m[None, None, :, None] >= arange_n[None, None, None, :], -torch.inf)
    top_k = min(top_k, context_size // 64)
    return torch.topk(p_pool, top_k, dim=-1).indices.to(torch.int32).sort(dim=-1).values


def _official_mask(query: torch.Tensor, key: torch.Tensor, pattern: HeadPattern, top_k: int = 100) -> torch.Tensor:
    """The ``[Q, Q]`` visibility of one head (``query``, ``key`` ``[Q, D]``) under the official implementation:
    the converted vertical-slash index, the block index (the kernel reads ``min(m + 1, top_k)`` blocks of row
    ``m``) or the streaming kernel's window and sinks, always causal."""
    queries = query.shape[0]
    pos = torch.arange(queries)
    causal = pos[None, :] <= pos[:, None]
    seen = torch.zeros(queries, queries, dtype=torch.bool)
    if pattern.kind == "a_shape":
        seen = (pos[:, None] - pos[None, :] < pattern.slash) | (pos[None, :] < pattern.vertical)
    elif pattern.kind == "vertical_slash":
        v_idx, s_idx = _official_lines(query[None, None], key[None, None], pattern.vertical, pattern.slash)
        for m, (offsets, columns) in enumerate(_official_convert(v_idx, s_idx, queries)):
            rows = slice(64 * m, 64 * m + 64)
            for start in offsets:
                seen[rows, start : start + 64] = True
            seen[rows, columns] = True
    else:
        chosen = _official_block_index(query[None, None], key[None, None], top_k)[0, 0]
        for m in range(math.ceil(queries / 64)):
            for block in chosen[m, : min(m + 1, chosen.shape[-1])].tolist():
                seen[64 * m : 64 * m + 64, 64 * block : 64 * block + 64] = True
    return seen & causal


# ---- helpers ----
LONG = 200  # four block rows of 64 queries, the last one partial


def _qk(seed: int, heads: int = 4, kv_heads: int = 2, queries: int = LONG, dim: int = 16) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(1, heads, queries, dim, generator=generator)
    key = torch.randn(1, kv_heads, queries, dim, generator=generator)
    return query, key


def _sequence(seed: int = 0, length: int = LONG) -> torch.Tensor:
    return torch.randint(0, VOCAB, (1, length), generator=torch.Generator().manual_seed(seed))


def _model(*configs: object, kv_heads: int = 2) -> economize.EfficientModel:
    install()
    return economize.apply(make_tiny_model(attn="economize", kv_heads=kv_heads), *configs)


def _logits(model: economize.EfficientModel, seq: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.base_model(seq).logits[0].float()


_MIXED = (
    HeadPattern("vertical_slash", 40, 120),
    HeadPattern("block_sparse"),
    HeadPattern("a_shape", 8, 70),
    HeadPattern("vertical_slash", 30, 150),
)


def _pattern_file(tmp_path: Path, layers: int = 6) -> Path:
    official = {"vertical_slash": "vertical_and_slash", "block_sparse": "block_sparse", "a_shape": "stream_llm"}
    rows = [
        {str(h): [official[p.kind], p.vertical, p.slash, 0.9] for h, p in enumerate(_MIXED[n % 2 :] + _MIXED[: n % 2])}
        for n in range(layers)
    ]
    path = tmp_path / "patterns.json"
    path.write_text(json.dumps(rows))
    return path


# ---- the index against the official implementation ----
@pytest.mark.parametrize("queries", [200, 128, 97])
def test_vertical_and_slash_lines_match_the_official_estimate(queries: int) -> None:
    query, key = _qk(queries, heads=2, kv_heads=1, queries=queries)
    for vertical, slash in ((30, 100), (45, 130)):
        columns, diagonals = vertical_slash_lines(query, key, 0.25, vertical, slash)
        for h in range(2):
            v_idx, s_idx = _official_lines(query[:, h : h + 1], key, vertical, slash)
            assert columns[0, h].tolist() == v_idx
            assert diagonals[0, h].tolist() == s_idx


def test_per_head_line_counts_pad_the_shorter_heads() -> None:
    query, key = _qk(1, heads=2, kv_heads=2)
    columns, diagonals = vertical_slash_lines(query, key, 0.25, torch.tensor([30, 60]), torch.tensor([100, 140]))
    assert columns.shape == (1, 2, 60) and diagonals.shape == (1, 2, 140)
    assert (columns[0, 0, 30:] >= 2**30).all() and (columns[0, 0, :30] < LONG).all()
    assert (diagonals[0, 0, :40] >= 2**30).all() and (diagonals[0, 0, 40:] < LONG).all()
    alone, _ = vertical_slash_lines(query[:, :1], key[:, :1], 0.25, 30, 100)
    assert torch.equal(columns[0, 0, :30], alone[0, 0])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_conversion_matches_the_official_kernel(seed: int) -> None:
    """Ranges and columns per block row cover the keys of the official block offsets and columns, and the columns
    are the official ones in the same order."""
    query, key = _qk(seed, heads=1, kv_heads=1)
    columns, diagonals = vertical_slash_lines(query, key, 0.25, 40 + 20 * seed, 100 + 25 * seed)
    starts, ends, count, column_index, column_count = convert_vertical_slash(columns, diagonals, LONG, kernel=False)
    official = _official_convert(columns[0, 0].tolist(), diagonals[0, 0].tolist(), LONG)
    assert count.shape == (1, 1, 4)
    for m, (offsets, cols) in enumerate(official):
        n = int(count[0, 0, m])
        ours = {j for r in range(n) for j in range(int(starts[0, 0, m, r]), int(ends[0, 0, m, r]))}
        theirs = {j for start in offsets for j in range(start, start + 64)}
        assert ours == theirs, m
        assert column_index[0, 0, m, : int(column_count[0, 0, m])].tolist() == cols, m


@pytest.mark.parametrize("queries", [200, 128])
def test_block_index_matches_the_official_one(queries: int) -> None:
    query, key = _qk(5, heads=2, kv_heads=2, queries=queries)
    for top_k in (2, 100):
        index = block_sparse_index(query, key, top_k)
        chosen = _official_block_index(query, key, top_k)
        blocks = index.blocks
        for m in range(blocks):
            n = int(index.range_count[0, 0, m])
            assert n == min(m + 1, chosen.shape[-1])
            assert (index.range_start[0, :, m, :n] // 64).tolist() == chosen[0, :, m, :n].tolist()


@pytest.mark.parametrize("kv_heads", [4, 2, 1])
def test_layer_mask_matches_the_official_patterns(kv_heads: int) -> None:
    """A layer of mixed heads (and query heads of one KV group with different patterns) sees what the official
    implementation's heads see."""
    query, key = _qk(7, heads=4, kv_heads=kv_heads)
    index = minference_index(query, key, 16**-0.5, _MIXED, top_k=2)
    mask = mask_from_index(index)
    groups = 4 // kv_heads
    for h, pattern in enumerate(_MIXED):
        expected = _official_mask(query[0, h], key[0, h // groups], pattern, top_k=2)
        assert torch.equal(mask[0, h], expected), pattern
    assert 0.0 < index.density() < 1.0


def test_coverage_contains_the_lines_and_the_a_shape() -> None:
    query, key = _qk(3, heads=1, kv_heads=1)
    columns, diagonals = vertical_slash_lines(query, key, 0.25, 40, 110)
    mask = mask_from_index(minference_index(query, key, 0.25, (HeadPattern("vertical_slash", 40, 110),)))[0, 0]
    pos = torch.arange(LONG)
    causal = pos[None, :] <= pos[:, None]
    lines = torch.zeros(LONG, LONG, dtype=torch.bool)
    lines[:, columns[0, 0].long()] = True
    lines |= torch.isin(pos[:, None] - pos[None, :], diagonals[0, 0].long())
    assert not (lines & causal & ~mask).any()
    a_shape = mask_from_index(a_shape_index(1, LONG, 4, 70, 1, query.device))[0, 0]
    assert torch.equal(a_shape, causal & ((pos[:, None] - pos[None, :] < 70) | (pos[None, :] < 4)))


def test_large_budgets_keep_every_key() -> None:
    query, key = _qk(4)
    whole = (HeadPattern("vertical_slash", LONG, LONG), HeadPattern("block_sparse"), HeadPattern("a_shape", 0, LONG))
    for pattern in whole:
        mask = mask_from_index(minference_index(query, key, 0.25, (pattern,) * 4))
        assert mask.sum(-1).tolist() == [[list(range(1, LONG + 1))] * 4], pattern


def test_reference_attention_hides_the_unindexed_keys() -> None:
    generator = torch.Generator().manual_seed(0)
    query, key = _qk(9)
    value = torch.randn(1, 2, LONG + 6, 16, generator=generator)
    key = torch.cat([key, torch.randn(1, 2, 6, 16, generator=generator)], dim=2)  # slots past the prompt
    index = minference_index(query, key[:, :, :LONG], 0.25, _MIXED, top_k=2)
    got = sparse_prefill_reference(query, key, value, index, 0.25)
    mask = mask_from_index(index)
    k, v = key[:, :, :LONG].repeat_interleave(2, 1), value[:, :, :LONG].repeat_interleave(2, 1)
    weights = (query @ k.transpose(-1, -2) * 0.25).masked_fill(~mask, float("-inf")).softmax(-1)
    torch.testing.assert_close(got, (weights @ v).transpose(1, 2), atol=1e-5, rtol=1e-5)


def test_pattern_files_read_the_official_format(tmp_path: Path) -> None:
    rows = load_patterns(_pattern_file(tmp_path, layers=2), ratio=0.5)
    assert len(rows) == 2 and rows[0][0] == HeadPattern("vertical_slash", 20, 60)
    assert rows[1][0] == HeadPattern("block_sparse", 500, 3048)
    assert rows[0][2] == HeadPattern("a_shape", 4, 35)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"0": ["dense", 1, 1, 0.0]}]))
    with pytest.raises(ConfigError, match="bad entry"):
        load_patterns(bad)
    with pytest.raises(ConfigError, match="cannot read"):
        load_patterns(tmp_path / "missing.json")


# ---- the model ----
def _reference_logits(
    seq: torch.Tensor, heads: object, dense_layers: int, monkeypatch: pytest.MonkeyPatch
) -> torch.Tensor:
    """Logits of one eager forward whose layers from ``dense_layers`` on attend over the official masks;
    ``heads(layer)`` gives the layer's head patterns."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.eager_attention_forward

    def official_attention_forward(module, query, key, value, attention_mask, scaling, **kwargs):
        if module.layer_idx < dense_layers:
            return original(module, query, key, value, attention_mask, scaling=scaling, **kwargs)
        groups = query.shape[1] // key.shape[1]
        patterns = heads(module.layer_idx)  # type: ignore[operator]
        mask = torch.stack([_official_mask(query[0, h], key[0, h // groups], p) for h, p in enumerate(patterns)])
        k, v = key.repeat_interleave(groups, 1), value.repeat_interleave(groups, 1)
        weights = (query @ k.transpose(-1, -2) * scaling).masked_fill(~mask, float("-inf")).softmax(-1)
        return (weights @ v).transpose(1, 2).contiguous(), None

    base = make_tiny_model(attn="eager")
    monkeypatch.setattr(modeling_qwen2, "eager_attention_forward", official_attention_forward)
    with torch.no_grad():
        logits = base(seq).logits[0].float()
    monkeypatch.setattr(modeling_qwen2, "eager_attention_forward", original)
    return logits


def test_prefill_attends_like_the_official_patterns(monkeypatch: pytest.MonkeyPatch) -> None:
    seq = _sequence()
    config = MInferenceConfig(vertical=40, slash=120, dense_layers=2)
    model = _model(config)
    kernel_usage(reset=True)
    got = _logits(model, seq)
    usage = kernel_usage(reset=True)
    assert {row["reason"] for row in usage if row["layer"] in (2, 3, 4, 5)} == {"minference_cpu"}
    pattern = HeadPattern("vertical_slash", 40, 120)
    expected = _reference_logits(seq, lambda layer: (pattern,) * 4, 2, monkeypatch)
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)
    plain = make_tiny_model()(seq).logits[0].float()
    assert not torch.allclose(got, plain, atol=1e-3)
    torch.testing.assert_close(got[:64], plain[:64], atol=1e-4, rtol=1e-4)  # the first block row sees every key


def test_pattern_file_mixes_patterns_per_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seq = _sequence(1)
    path = _pattern_file(tmp_path)
    rows = load_patterns(path)
    model = _model(MInferenceConfig(pattern_file=str(path), top_k=100))
    expected = _reference_logits(seq, lambda layer: rows[layer], 0, monkeypatch)
    torch.testing.assert_close(_logits(model, seq), expected, atol=1e-4, rtol=1e-4)


def test_short_prompts_large_budgets_and_detach_are_dense() -> None:
    seq = _sequence()
    plain = make_tiny_model()(seq).logits[0].float()
    model = _model(MInferenceConfig(vertical=LONG, slash=LONG))
    torch.testing.assert_close(_logits(model, seq), plain, atol=1e-4, rtol=1e-4)
    sparse = _model(MInferenceConfig(vertical=30, slash=100, min_queries=LONG + 1))
    kernel_usage(reset=True)
    torch.testing.assert_close(_logits(sparse, seq), plain, atol=1e-4, rtol=1e-4)
    assert not any(row["reason"].startswith("minference") for row in kernel_usage(reset=True))
    attached = _model(MInferenceConfig(vertical=30, slash=100))
    modules = [attached.adapter.attention_module(i) for i in range(1, 7)]
    assert all(isinstance(prefill_of(m), MInferencePrefill) for m in modules)
    attached.techniques["minference"].detach()
    assert all(prefill_of(m) is None for m in modules)
    torch.testing.assert_close(_logits(attached, seq), plain, atol=1e-4, rtol=1e-4)


def test_decoding_after_a_sparse_prefill() -> None:
    """Decoding steps attend densely over the cache of the sparse prefill; the tokens are the model's own when the
    prompt is short and the batch matches single rows."""
    model = _model(MInferenceConfig(vertical=30, slash=100))
    prompt = _sequence(2, length=150)[0]
    out = model.decode(prompt, 20)
    assert out.tokens.numel() == 20
    short = _sequence(3, length=40)[0]
    plain = economize.apply(make_tiny_model()).decode(short, 10).tokens.tolist()
    assert model.decode(short, 10).tokens.tolist() == plain
    assert [r.tokens.tolist() for r in model.decode_batch([prompt, prompt], [12, 12])] == [out.tokens[:12].tolist()] * 2


# ---- configuration and composition ----
def test_config_validation(tmp_path: Path) -> None:
    for fields, match in (
        ({"pattern": "dense"}, "pattern must be one of"),
        ({"slash": 0}, "slash >= 1"),
        ({"top_k": 0}, "top_k"),
        ({"dense_layers": -1}, "non-negative"),
        ({"min_queries": 1}, "at least 2"),
        ({"ratio": 0.0}, "ratio"),
    ):
        with pytest.raises(ConfigError, match=match):
            MInferenceConfig(**fields)  # type: ignore[arg-type]
    config = MInferenceConfig(pattern="a_shape", sinks=4, local=64, layers=(2, 3))
    assert config.head_pattern() == HeadPattern("a_shape", 4, 64)
    assert MInferenceConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    with pytest.raises(ConfigError, match="no layer"):
        _model(MInferenceConfig(dense_layers=6))
    with pytest.raises(ConfigError, match="2 layers"):
        _model(MInferenceConfig(pattern_file=str(_pattern_file(tmp_path, layers=2))))


def test_needs_an_attention_implementation_with_sparse_prefill() -> None:
    with pytest.raises(IncompatibleTechniquesError, match="economize"):
        economize.apply(make_tiny_model(attn="sdpa"), MInferenceConfig())


def test_covers_full_attention_layers() -> None:
    adapter = adapter_for(make_tiny_model())
    assert minference_layers(MInferenceConfig(dense_layers=2), adapter) == (3, 4, 5, 6)
    assert minference_layers(MInferenceConfig(layers=(1, 3, 5)), adapter) == (1, 3, 5)
    gemma = adapter_for(make_family_model("gemma4_text"))
    assert minference_layers(MInferenceConfig(), gemma) == (3, 6)  # layers 1, 2, 4 and 5 slide


def test_combines_with_kv_and_weight_techniques() -> None:
    for name in ("kv_evict", "kivi", "kv_fp8", "quest", "duo_attention", "kv_evict_fused"):
        assert economize.compatible("minference", name), name
    from economize.techniques.kv_evict import KVEvictConfig

    model = _model(MInferenceConfig(vertical=30, slash=100), KVEvictConfig(recent=32, sinks=4))
    assert model.decode(_sequence(4, length=150)[0], 12).tokens.numel() == 12


# ---- the kernels ----
_INTERPRETER = """
import torch
from economize.attention.minference import (
    HeadPattern, convert_vertical_slash, minference_index, sparse_prefill_attention, sparse_prefill_reference,
    vertical_slash_lines,
)

torch.manual_seed(0)
worst = 0.0
for queries in (200, 64):
    q, k = torch.randn(1, 2, queries, 16), torch.randn(1, 2, queries, 16)
    columns, diagonals = vertical_slash_lines(q, k, 0.25, torch.tensor([30, 45]), torch.tensor([100, 120]))
    ours = convert_vertical_slash(columns, diagonals, queries, kernel=True)
    theirs = convert_vertical_slash(columns, diagonals, queries, kernel=False)
    for a, b, n in ((ours[0], theirs[0], ours[2]), (ours[1], theirs[1], ours[2]), (ours[3], theirs[3], ours[4])):
        valid = torch.arange(a.shape[-1]) < n.unsqueeze(-1)
        worst = max(worst, float((a.where(valid, 0) != b.where(valid, 0)).any()))
    worst = max(worst, float(not torch.equal(ours[2], theirs[2])), float(not torch.equal(ours[4], theirs[4])))
patterns = (HeadPattern("vertical_slash", 40, 120), HeadPattern("block_sparse"), HeadPattern("a_shape", 8, 70),
            HeadPattern("vertical_slash", 30, 150))
for dtype in (torch.float32, torch.float8_e4m3fn):
    q = torch.randn(2, 4, 150, 32)
    k, v = torch.randn(2, 2, 160, 32), torch.randn(2, 2, 160, 32)
    index = minference_index(q, k[:, :, :150], 0.2, patterns, top_k=1, kernel=False)
    if dtype != torch.float32:
        k, v = k.to(dtype), v.to(dtype)
    expected = sparse_prefill_reference(q, k.float(), v.float(), index, 0.2)
    for tiles in (None, (16, 32), (32, 16)):
        got = sparse_prefill_attention(q, k, v, index, 0.2, tiles=tiles)
        worst = max(worst, (got.float() - expected.float()).abs().max().item())
    got = sparse_prefill_attention(q, k, v, index, 0.2, softcap=5.0)
    expected = sparse_prefill_reference(q, k.float(), v.float(), index, 0.2, softcap=5.0)
    worst = max(worst, (got.float() - expected.float()).abs().max().item())
print(worst)
"""


@pytest.mark.slow
def test_kernels_match_the_reference_in_the_triton_interpreter() -> None:
    """The conversion kernel gives the ranges and columns of the torch path; the attention kernel gives the
    reference attention for mixed patterns, GQA, fp32 and fp8 keys, three tile shapes and a soft cap (Triton's
    CPU interpreter; it must be enabled before triton loads, hence a process)."""
    pytest.importorskip("triton")
    src = str(Path(economize.__file__).resolve().parents[1])
    env = {**os.environ, "TRITON_INTERPRET": "1", "PYTHONPATH": src}
    result = subprocess.run(
        [sys.executable, "-c", _INTERPRETER], env=env, capture_output=True, text=True, timeout=900, check=True
    )
    assert float(result.stdout.strip().splitlines()[-1]) < 1e-5


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
def test_gpu_kernel_matches_the_reference(dtype: torch.dtype, head_dim: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(1, 8, 1000, head_dim, device="cuda", dtype=dtype, generator=generator)
    k = torch.randn(1, 2, 1000, head_dim, device="cuda", dtype=dtype, generator=generator)
    v = torch.randn(1, 2, 1000, head_dim, device="cuda", dtype=dtype, generator=generator)
    from economize.attention import sparse_prefill_attention

    index = minference_index(q, k, head_dim**-0.5, _MIXED * 2, top_k=4)
    cpu = minference_index(q.cpu().float(), k.cpu().float(), head_dim**-0.5, _MIXED * 2, top_k=4, kernel=False)
    assert torch.equal(mask_from_index(index).cpu(), mask_from_index(cpu))  # the conversion kernel on CUDA
    expected = sparse_prefill_reference(q, k, v, index, head_dim**-0.5)
    for kv in (dtype, torch.float8_e4m3fn):
        got = sparse_prefill_attention(q, k.to(kv), v.to(kv), index, head_dim**-0.5)
        reference = expected
        if kv != dtype:
            k8, v8 = k.to(kv).to(dtype), v.to(kv).to(dtype)
            reference = sparse_prefill_reference(q, k8, v8, index, head_dim**-0.5)
        torch.testing.assert_close(got.float(), reference.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.gpu
def test_gpu_prefill_uses_the_kernel() -> None:
    seq = _sequence()
    expected = _logits(_model(MInferenceConfig(vertical=40, slash=120)), seq)
    install()
    model = economize.apply(make_tiny_model(attn="economize").cuda(), MInferenceConfig(vertical=40, slash=120))
    kernel_usage(reset=True)
    got = _logits(model, seq.cuda()).cpu()
    assert "kernel_minference" in {row["path"] for row in kernel_usage(reset=True)}
    torch.testing.assert_close(got, expected, atol=1e-3, rtol=1e-3)
