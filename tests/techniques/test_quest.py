"""Quest: every decoding query reads the ``token_budget / page_size - 1`` committed pages with the largest score
bounds plus the page of its own token, checked against the official implementation's reference attention."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import economize
from economize import DecodeConfig
from economize.arch import adapter_for
from economize.attention import install
from economize.attention.quest import QuestPages, page_bounds, quest_reference, select_pages
from economize.errors import ConfigError, IncompatibleTechniquesError
from economize.runtime import phase
from economize.techniques.duo_attention import DuoAttentionConfig
from economize.techniques.kv_evict import KVEvictConfig
from economize.techniques.kv_quant import KIVIConfig
from economize.techniques.quest import QuestCache, QuestConfig, quest_layers
from helpers import VOCAB, make_family_model, make_tiny_model

PAGE, BUDGET = 16, 48  # two committed pages and the running page
PREFIX, TOTAL = 20, 90


def _ref_self_approx_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, page_size: int, page_budget: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``_ref_self_approx_attention`` of the official Quest tests (mit-han-lab/Quest, quest/tests/
    test_approx_attention.py; MIT License, Copyright (c) 2024 MIT HAN Lab), unchanged up to formatting: one
    multi-head decoding step over NHD tensors, returning the output and the selected pages."""
    head_dim = q.size(2)
    qo_len = q.size(0)
    kv_len = k.size(0)
    q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
    attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)
    attn_mask = torch.ones_like(attn_weights, dtype=torch.bool).tril(diagonal=kv_len - qo_len)
    cur_num_pages = (kv_len + page_size - 1) // page_size
    if cur_num_pages > page_budget:
        sign = (q > 0) + (~(q > 0)) * -1
        max_key = k * sign
        postive_query = q * sign
        padding_length = page_size - ((kv_len - 1) % page_size + 1)
        max_key = torch.cat(
            [
                max_key,
                torch.ones((max_key.shape[0], padding_length, max_key.shape[2]), device=max_key.device)
                * torch.tensor(torch.finfo(max_key.dtype).min),
            ],
            dim=-2,
        )
        page_max_key = max_key.reshape(
            max_key.shape[0], max_key.shape[1] // page_size, page_size, max_key.shape[2]
        ).amax(dim=-2, keepdim=False)
        approx_attn_paged = torch.matmul(postive_query.float(), page_max_key.transpose(1, 2))
        approx_attn_paged = approx_attn_paged[:, :, : cur_num_pages - 1]  # remove the last page
        _, topk = approx_attn_paged.topk(page_budget - 1, dim=-1)
        saved_topk_indices = topk.clone().squeeze().to(torch.int32)
        topk = torch.cat(
            [topk, torch.tensor([cur_num_pages - 1], device=topk.device).repeat(topk.shape[0], topk.shape[1], 1)],
            dim=-1,
        )
        topk = topk.unsqueeze(-1).repeat(1, 1, 1, page_size) * page_size + torch.arange(page_size, device=topk.device)
        topk = topk.reshape(topk.shape[0], topk.shape[1], -1)
        topk = topk[:, :, : page_budget * page_size - padding_length]
        mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
        mask_bottom.scatter_(-1, topk, True)
        attn_weights[~mask_bottom] = torch.tensor(torch.finfo(attn_weights.dtype).min)
    else:
        saved_topk_indices = None
    attn_weights[~attn_mask] = torch.tensor(torch.finfo(attn_weights.dtype).min)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn_weights, v).transpose(0, 1), saved_topk_indices


def _pages(key: torch.Tensor, done: list[int], pages: int) -> QuestPages:
    most = max(done)
    kmin, kmax = page_bounds(key, 0, most, PAGE)
    return QuestPages(kmin, kmax, torch.tensor(done, dtype=torch.int32), min(done), most, PAGE, pages)


def _sequence(seed: int = 0, length: int = TOTAL) -> torch.Tensor:
    return torch.randint(0, VOCAB, (1, length), generator=torch.Generator().manual_seed(seed))


def _prompts() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in (40, 23, 57)]


def _model(*configs: object, kv_heads: int = 2, **decoding: int) -> economize.EfficientModel:
    install()
    base = make_tiny_model(attn="economize", kv_heads=kv_heads)
    return economize.apply(base, *configs, decoding=DecodeConfig(**decoding) if decoding else None)


def _config(**fields: object) -> QuestConfig:
    return QuestConfig(**{"page_size": PAGE, "token_budget": BUDGET, **fields})


def _teacher_forced(model: economize.EfficientModel, seq: torch.Tensor) -> tuple[torch.Tensor, QuestCache]:
    """Logits of ``seq`` from a prefill of :data:`PREFIX` tokens and single-token decode steps, and the cache."""
    runner = model.runner
    cache = model.techniques["quest"].wrap_cache(runner.new_cache(128))
    ctx = runner.context(cache, torch.arange(PREFIX, device=seq.device)[None], trusted=True)
    with phase("prefill"):
        out = [runner.logits(runner.forward(seq[:, :PREFIX], ctx).final)[0]]
    for t in range(PREFIX, seq.shape[1]):
        ctx = runner.context(cache, torch.tensor([[t]], device=seq.device), trusted=True)
        with phase("decode"):
            out.append(runner.logits(runner.forward(seq[:, t : t + 1], ctx).final)[0])
    assert isinstance(cache, QuestCache)
    return torch.cat(out).float().cpu(), cache


def _gqa_reference(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float) -> torch.Tensor:
    """One decoding step of grouped-query Quest, transcribed from the paper: the query heads of a KV head add their
    bounds ``sum_d max(q_d min_d, q_d max_d)`` and share the top pages. ``query`` ``[Hq, D]``, ``key``/``value``
    ``[Hkv, S, D]`` (the running token last); returns ``[Hq, D]``."""
    kv_heads, slots, dim = key.shape
    groups = query.shape[0] // kv_heads
    done, pages = (slots - 1) // PAGE, BUDGET // PAGE - 1
    keep = torch.ones(kv_heads, slots, dtype=torch.bool)
    if done > pages:
        blocks = key[:, : done * PAGE].unflatten(1, (done, PAGE))
        low, high = blocks.amin(dim=2), blocks.amax(dim=2)  # [Hkv, P, D]
        q = query.view(kv_heads, groups, 1, dim)
        bound = torch.maximum(q * low[:, None], q * high[:, None]).sum(dim=-1).sum(dim=1)  # [Hkv, P]
        chosen = torch.zeros(kv_heads, done, dtype=torch.bool)
        chosen.scatter_(1, bound.topk(pages, dim=-1).indices, True)
        keep[:, : done * PAGE] = chosen.repeat_interleave(PAGE, dim=1)
    keep = keep.repeat_interleave(groups, dim=0)
    scores = torch.einsum("hd,hsd->hs", query, key.repeat_interleave(groups, dim=0)) * scale
    weights = scores.masked_fill(~keep, float("-inf")).softmax(dim=-1)
    return torch.einsum("hs,hsd->hd", weights, value.repeat_interleave(groups, dim=0))


def _reference_logits(seq: torch.Tensor, kv_heads: int, monkeypatch: pytest.MonkeyPatch) -> torch.Tensor:
    """Logits of ``seq`` from one eager forward whose decode positions (from :data:`PREFIX` on) in layers 3 to 6
    attend like the reference: the official function for multi-head attention, the transcription for GQA."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.eager_attention_forward

    def quest_attention_forward(module, query, key, value, attention_mask, scaling, **kwargs):
        out, weights = original(module, query, key, value, attention_mask, scaling=scaling, **kwargs)
        if module.layer_idx < 2:
            return out, weights
        out = out.clone()
        for t in range(PREFIX, query.shape[2]):
            q, k, v = query[0, :, t], key[0, :, : t + 1], value[0, :, : t + 1]
            if kv_heads == query.shape[1]:
                step, _ = _ref_self_approx_attention(
                    q[None], k.transpose(0, 1), v.transpose(0, 1), PAGE, BUDGET // PAGE
                )
                out[0, t] = step[0]
            else:
                out[0, t] = _gqa_reference(q, k, v, scaling)
        return out, weights

    base = make_tiny_model(attn="eager", kv_heads=kv_heads)
    monkeypatch.setattr(modeling_qwen2, "eager_attention_forward", quest_attention_forward)
    with torch.no_grad():
        logits = base(seq).logits[0].float()
    monkeypatch.setattr(modeling_qwen2, "eager_attention_forward", original)
    return logits


# ---- the official reference ----
@pytest.mark.parametrize("kv_len", [40, 49, 64, 65, 113, 160])
def test_selection_and_output_match_the_official_reference(kv_len: int) -> None:
    """For multi-head attention the selected pages and the output equal the official reference's, including its
    dense regime (at most ``page_budget`` pages in all) and every position of the running token in its page."""
    generator = torch.Generator().manual_seed(kv_len)
    heads, dim, budget = 4, 32, 3
    q, k, v = (torch.randn(n, heads, dim, generator=generator) for n in (1, kv_len, kv_len))
    expected, indices = _ref_self_approx_attention(q, k, v, PAGE, budget)
    query, key, value = q.permute(1, 0, 2)[None], k.permute(1, 0, 2)[None], v.permute(1, 0, 2)[None]
    done = (kv_len - 1) // PAGE
    if done == 0:
        assert indices is None
        return
    pages = _pages(key, [done], budget - 1)
    got = quest_reference(query, key, value, None, dim**-0.5, pages)
    torch.testing.assert_close(got[0], expected, atol=1e-5, rtol=1e-5)
    selected = pages.select(query)[0]
    if indices is None:  # dense: every committed page
        assert done <= budget - 1
        assert selected[:, :done].tolist() == [list(range(done))] * heads and (selected[:, done:] == -1).all()
    else:
        assert torch.equal(selected, indices.sort(dim=-1).values)


def test_gqa_selection_adds_the_bounds_of_the_group() -> None:
    generator = torch.Generator().manual_seed(0)
    query = torch.randn(2, 8, 1, 16, generator=generator)
    key = torch.randn(2, 2, 100, 16, generator=generator)
    value = torch.randn(2, 2, 100, 16, generator=generator)
    pages = _pages(key, [6, 4], 2)
    expected = torch.stack([_gqa_reference(query[0, :, 0], key[0, :, :97], value[0, :, :97], 0.25)])
    got = quest_reference(query[:1], key[:1, :, :97], value[:1, :, :97], None, 0.25, _pages(key[:1, :, :97], [6], 2))
    torch.testing.assert_close(got[:, 0], expected, atol=1e-5, rtol=1e-5)
    selected = select_pages(query, pages.kmin, pages.kmax, pages.done, 6, 2)
    assert selected.shape == (2, 2, 2) and (selected[1] < 4).all() and (selected >= 0).all()
    assert (selected.diff(dim=-1) > 0).all()  # increasing slot order
    few = select_pages(query, pages.kmin, pages.kmax, torch.tensor([6, 1], dtype=torch.int32), 6, 3)
    assert few[1].tolist() == [[0, -1, -1]] * 2  # a short row reads its committed pages, then nothing


def test_restrict_mask_keeps_the_selected_pages_and_the_tail() -> None:
    generator = torch.Generator().manual_seed(1)
    query = torch.randn(1, 4, 1, 16, generator=generator)
    key = torch.randn(1, 2, 80, 16, generator=generator)
    pages = _pages(key, [4], 2)
    mask = torch.ones(1, 1, 1, 80, dtype=torch.bool)
    mask[..., 75:] = False
    restricted = pages.restrict_mask(query, key, mask)
    selected = pages.select(query)[0]
    for head in range(4):
        visible = restricted[0, head, 0].nonzero().flatten().tolist()
        chosen = [p * PAGE + i for p in selected[head // 2].tolist() for i in range(PAGE)]
        assert visible == chosen + list(range(64, 75))
    additive = pages.restrict_mask(query, key, torch.zeros(1, 1, 1, 80))
    assert torch.equal(additive == 0, pages.restrict_mask(query, key, None))
    assert pages.restrict_mask(torch.randn(1, 4, 2, 16), key, mask) is mask  # longer queries see every key


# ---- decoding ----
@pytest.mark.parametrize("kv_heads", [4, 2])
def test_decodes_like_the_reference(kv_heads: int, monkeypatch: pytest.MonkeyPatch) -> None:
    seq = _sequence()
    reference = _reference_logits(seq, kv_heads, monkeypatch)
    model = _model(_config(), kv_heads=kv_heads)
    logits, cache = _teacher_forced(model, seq)
    plain = make_tiny_model(kv_heads=kv_heads)(seq).logits[0].float()
    torch.testing.assert_close(logits, reference, atol=1e-4, rtol=1e-4)
    sparse = BUDGET  # the first position with more committed pages than the budget holds
    assert not torch.allclose(logits[sparse:], plain[sparse:], atol=1e-3)
    torch.testing.assert_close(logits[:sparse], plain[:sparse], atol=1e-4, rtol=1e-4)
    assert cache.done == [(TOTAL - 1) // PAGE]
    assert cache.bounds(0) is None and cache.bounds(2) is not None  # the first two layers stay dense


def test_a_budget_above_the_context_is_the_model() -> None:
    seq = _sequence(length=60)
    logits, cache = _teacher_forced(_model(_config(token_budget=64)), seq)
    torch.testing.assert_close(logits, make_tiny_model()(seq).logits[0].float(), atol=1e-4, rtol=1e-4)
    assert all(cache.bounds(layer) is None for layer in range(6))


@pytest.mark.parametrize("cycle", [1, 4])
def test_batched_decoding_matches_single_rows(cycle: int) -> None:
    """Rows of different lengths in one batch decode like each row alone, with and without drafting (whose
    verification passes attend densely)."""
    if cycle == 1:
        model = _model(_config(token_budget=32))
    else:
        install()
        jump = economize.JumpConfig(target_layer=0, source_layer=0, adaptation="frozen", lora_rank=2)
        model = economize.apply(
            make_tiny_model(attn="economize"), jump, _config(token_budget=32),
            decoding=DecodeConfig(cycle_length=cycle),
        )
        for name, parameter in model.techniques["jump"].predictor.named_parameters():
            if name != "out_gain":
                parameter.zero_()
    lengths = [30, 41, 25]
    batched = [r.tokens.tolist() for r in model.decode_batch(_prompts(), lengths)]
    single = [model.decode(p, n).tokens.tolist() for p, n in zip(_prompts(), lengths, strict=True)]
    assert batched == single


def test_read_bytes_count_the_selected_pages() -> None:
    model = _model(_config())
    _teacher_forced(model, _sequence())
    entry = 2 * 2 * 16  # K and V of one slot of one layer, in elements
    pages = BUDGET // PAGE - 1
    read = full = 0
    for t in range(PREFIX, TOTAL):
        done = t // PAGE
        dense = entry * t * 4
        full += 6 * dense
        sparse = entry * (pages * PAGE + t - done * PAGE) * 4 + entry * done * 4
        read += 2 * dense + 4 * (sparse if done > pages else dense)
    stats = model.techniques["quest"].kv_stats()
    assert stats["kv_bytes_read_full"] == full and stats["kv_bytes_read"] == read
    committed = TOTAL - 1
    assert stats["kv_bytes_full"] == 6 * entry * committed * 4
    assert stats["kv_bytes"] == stats["kv_bytes_full"] + 4 * entry * (committed // PAGE) * 4  # the bounds


# ---- configuration and composition ----
def test_config_validation() -> None:
    for fields, match in (
        ({"page_size": 8}, "power of two"),
        ({"page_size": 24}, "power of two"),
        ({"token_budget": 16}, "two pages"),
        ({"dense_layers": -1}, "non-negative"),
        ({"max_queries": 0}, "positive"),
    ):
        with pytest.raises(ConfigError, match=match):
            QuestConfig(**fields)  # type: ignore[arg-type]
    assert QuestConfig().pages == 127 and _config().pages == 2
    data = json.loads(json.dumps(_config(layers=(3, 4)).to_dict()))
    assert QuestConfig.from_dict(data) == _config(layers=(3, 4))
    with pytest.raises(ConfigError, match="no layer"):
        _model(_config(dense_layers=6))


def test_needs_an_attention_implementation_that_reads_the_pages() -> None:
    with pytest.raises(IncompatibleTechniquesError, match="economize"):
        economize.apply(make_tiny_model(attn="sdpa"), _config())


def test_a_dropped_attachment_is_an_error() -> None:
    """When the pages reach no attention function that reads them, the pass fails instead of decoding densely."""
    install()
    base = make_tiny_model(attn="economize")
    model = economize.apply(base, _config())
    base.config._attn_implementation = "sdpa"
    with pytest.raises(IncompatibleTechniquesError, match="no attention function read"):
        model.decode(_prompts()[0], 20)


def test_covers_full_attention_layers_that_write_their_cache() -> None:
    adapter = adapter_for(make_tiny_model())
    assert quest_layers(_config(), adapter) == frozenset(range(2, 6))
    assert quest_layers(_config(layers=(1, 3, 5)), adapter) == frozenset({2, 4})
    gemma = adapter_for(make_family_model("gemma4_text"))
    # Layers 1, 2, 4 and 5 slide; 5 and 6 read layer 3 and 4's cache (K/V sharing): only layer 3 writes a full one.
    assert quest_layers(_config(dense_layers=0), gemma) == frozenset({2})
    assert quest_layers(_config(dense_layers=0, include_sliding=True), gemma) == frozenset({0, 1, 2, 3})


def test_combines_with_eviction_and_simulated_quantization() -> None:
    for name in ("kv_evict", "duo_attention", "kivi", "kv_fp8"):
        assert economize.compatible("quest", name), name
    for name in ("kivi_fused", "kv_evict_fused"):
        assert not economize.compatible("quest", name), name
    others = (
        KVEvictConfig(recent=8, sinks=2),
        KVEvictConfig(method="pyramidkv", heavy=8, observation_window=4, beta=4),
        DuoAttentionConfig(pattern=((0.9, 0.1),) * 6, sparsity=0.5),
        KIVIConfig(bits=4, group_size=16, residual_length=16),
    )
    for other in others:
        model = _model(_config(token_budget=32), other)
        assert model.decode(_prompts()[0], 30).tokens.numel() == 30


# ---- the kernel ----
_INTERPRETER = """
import sys
import torch
from economize.attention.decode import DecodePlan
from economize.attention.quest import QuestPages, page_bounds, quest_decode_attention, quest_reference

torch.manual_seed(0)
worst = 0.0
for dtype in (torch.float32, torch.float8_e4m3fn):
    q = torch.randn(2, 4, 1, 32)
    k, v = torch.randn(2, 2, 150, 32), torch.randn(2, 2, 150, 32)
    if dtype != torch.float32:
        k, v = k.to(dtype), v.to(dtype)
    kmin, kmax = page_bounds(k, 0, 7, 16)
    pages = QuestPages(kmin, kmax, torch.tensor([7, 4], dtype=torch.int32), 4, 7, 16, 3)
    mask = torch.zeros(2, 1, 1, 150, dtype=torch.bool)
    mask[0, ..., :130] = True
    mask[1, ..., :90] = True
    expected = quest_reference(q, k, v, mask, 0.2, pages)
    for launch in (DecodePlan(16, 16, 32, 1, 32, 6), DecodePlan(16, 16, 32, 1, 64, 3)):
        got = quest_decode_attention(q, k, v, mask, 0.2, pages, launch=launch)
        worst = max(worst, (got.float() - expected.float()).abs().max().item())
print(worst)
"""


@pytest.mark.slow
def test_kernel_matches_the_reference_in_the_triton_interpreter() -> None:
    """The page kernel reads the selected pages and the tail of rows of different lengths, fp32 and fp8 keys,
    under two split shapes (Triton's CPU interpreter; it must be enabled before triton loads, hence a process)."""
    pytest.importorskip("triton")
    src = str(Path(economize.__file__).resolve().parents[1])
    env = {**os.environ, "TRITON_INTERPRET": "1", "PYTHONPATH": src}
    result = subprocess.run(
        [sys.executable, "-c", _INTERPRETER], env=env, capture_output=True, text=True, timeout=600, check=True
    )
    assert float(result.stdout.strip().splitlines()[-1]) < 1e-5


@pytest.mark.gpu
def test_gpu_kernel_decodes_like_the_masked_fallback() -> None:
    """On CUDA the ``"economize"`` decode kernel reads only the selected pages and gives the logits of the CPU
    fallback, which hides the other pages in the mask."""
    from economize.attention import kernel_usage

    seq = _sequence()
    expected, _ = _teacher_forced(_model(_config()), seq)
    install()
    model = economize.apply(make_tiny_model(attn="economize").cuda(), _config())
    kernel_usage(reset=True)
    got, _ = _teacher_forced(model, seq.cuda())
    assert "kernel" in {row["path"] for row in kernel_usage(reset=True)}
    torch.testing.assert_close(got, expected, atol=1e-3, rtol=1e-3)

