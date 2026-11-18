"""StreamingLLM, H2O and SnapKV against independent sequential references.

Every reference runs the model without a cache, through :func:`_masked_forward`: a plain re-implementation of the
Qwen2 forward from its modules in which each layer and KV head gets its own boolean mask of the keys every query
may read. Evicting an entry before query ``t`` is then the same as hiding it from row ``t`` (and from every later
row) of that mask, because a query's hidden states depend only on its own mask row and on the keys' hidden states,
which their own rows fixed when they were queries. The rules follow the reference code:

* StreamingLLM: ``StartRecentKVCache`` of https://github.com/mit-han-lab/streaming-llm with ``start_size = sinks``
  and ``recent_size = recent`` (after each step the cache holds the first ``start_size`` and the last
  ``recent_size`` entries). The reference re-assigns positions inside the cache; economize keeps the original
  positions (documented on :class:`~economize.techniques.kv_evict.StreamingEviction`), and so does this reference.
* H2O: the paper's real-drop decoding (https://github.com/FMInference/H2O): the attention every key received is
  accumulated over all queries (prompt and generated), and before each generated token the cache keeps the
  ``recent`` last entries plus the ``heavy`` older entries with the highest accumulated score. Under GQA the scores
  of a query group are summed and the group shares one selection.
* SnapKV: ``SnapKVCluster.update_kv`` of https://github.com/Zefan-Cai/KVCache-Factory, with the scores of a query
  group summed (as for H2O).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

import economize
from economize.runtime import phase
from economize.techniques.kv_evict import KVEvictConfig
from helpers import VOCAB, make_tiny_model

PREFIX, TOTAL = 16, 40
LAYERS, HEADS, KV_HEADS = 6, 4, 2


def _sequence(seed: int = 0) -> torch.Tensor:
    return torch.randint(0, VOCAB, (1, TOTAL), generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def _masked_forward(model: nn.Module, ids: torch.Tensor, allowed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits ``[T, V]`` and attention probabilities ``[L, H, T, T]`` of ``ids`` ``[1, T]`` when query ``i`` of
    layer ``l`` and KV head ``h`` reads key ``j`` iff ``allowed[l, h, i, j]``."""
    inner = model.model
    T = ids.shape[1]
    hidden = inner.embed_tokens(ids)
    cos, sin = inner.rotary_emb(hidden, torch.arange(T)[None])
    probs = []
    for layer, block in enumerate(inner.layers):
        attn = block.self_attn
        x = block.input_layernorm(hidden)
        q = attn.q_proj(x).view(1, T, -1, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(1, T, -1, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(1, T, -1, attn.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        groups = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        scores = q @ k.transpose(2, 3) * attn.scaling
        mask = allowed[layer].repeat_interleave(groups, 0)[None]
        p = scores.masked_fill(~mask, float("-inf")).softmax(-1)
        probs.append(p[0])
        out = (p @ v).transpose(1, 2).reshape(1, T, -1)
        hidden = hidden + attn.o_proj(out)
        hidden = hidden + block.mlp(block.post_attention_layernorm(hidden))
    return model.lm_head(inner.norm(hidden))[0], torch.stack(probs)


def _group_sum(probs: torch.Tensor) -> torch.Tensor:
    """``[L, H, ...]`` summed over the query heads of each KV head: ``[L, H_kv, ...]``."""
    return probs.view(probs.shape[0], KV_HEADS, -1, *probs.shape[2:]).sum(2)


def _causal() -> torch.Tensor:
    causal = torch.ones(TOTAL, TOTAL, dtype=torch.bool).tril()
    return causal.expand(LAYERS, KV_HEADS, TOTAL, TOTAL).clone()


def _teacher_forced(model: economize.EfficientModel, seq: torch.Tensor) -> tuple[torch.Tensor, object]:
    """Logits of ``seq`` from a prefill of :data:`PREFIX` tokens and single-token decode steps, and the cache."""
    runner = model.runner
    cache = model.techniques["kv_evict"].wrap_cache(runner.new_cache(64))
    ctx = runner.context(cache, torch.arange(PREFIX)[None], trusted=True)
    with phase("prefill"):
        out = [runner.logits(runner.forward(seq[:, :PREFIX], ctx).final)[0]]
    for t in range(PREFIX, seq.shape[1]):
        ctx = runner.context(cache, torch.tensor([[t]]), trusted=True)
        with phase("decode"):
            out.append(runner.logits(runner.forward(seq[:, t : t + 1], ctx).final)[0])
    return torch.cat(out).float(), cache


@pytest.mark.parametrize(("sinks", "recent"), [(4, 8), (0, 6), (2, 3)])
def test_streaming_matches_the_start_recent_cache(sinks: int, recent: int) -> None:
    seq = _sequence()
    allowed = _causal()
    for t in range(PREFIX, TOTAL):
        # after step t - 1 the reference cache holds the first ``sinks`` and the last ``recent`` entries
        allowed[..., t, sinks : max(sinks, t - recent)] = False
    expected, _ = _masked_forward(make_tiny_model(attn="eager"), seq, allowed)
    model = economize.apply(make_tiny_model(), KVEvictConfig(recent=recent, sinks=sinks))
    logits, _ = _teacher_forced(model, seq)
    torch.testing.assert_close(logits, expected, rtol=0, atol=2e-4)


@pytest.mark.parametrize(("recent", "heavy"), [(4, 8), (6, 3), (2, 0)])
def test_h2o_matches_the_sequential_heavy_hitter_reference(recent: int, heavy: int) -> None:
    seq = _sequence()
    base = make_tiny_model(attn="eager")
    allowed = _causal()
    keep = torch.ones(LAYERS, KV_HEADS, TOTAL, dtype=torch.bool)
    for t in range(PREFIX, TOTAL):
        # the scores so far: every query before t, each over the keys it was allowed to read
        _, probs = _masked_forward(base, seq[:, :t], allowed[..., :t, :t])
        received = _group_sum(probs).sum(2)  # [L, H_kv, t]
        old = keep[..., :t].clone()
        old[..., max(t - recent, 0) :] = False
        ranked = received.masked_fill(~old, float("-inf"))
        top = ranked.sort(dim=-1, descending=True, stable=True).indices[..., :heavy]
        chosen = torch.zeros_like(old).scatter_(-1, top, True) & old
        keep[..., :t] &= ~old | chosen
        allowed[..., t, :t] = keep[..., :t]
    expected, _ = _masked_forward(base, seq, allowed)
    model = economize.apply(make_tiny_model(), KVEvictConfig(method="h2o", recent=recent, heavy=heavy))
    logits, cache = _teacher_forced(model, seq)
    torch.testing.assert_close(logits, expected, rtol=0, atol=2e-4)
    for layer in range(LAYERS):
        assert torch.equal(cache.keep[layer][0, :, : TOTAL - 1], keep[layer, :, : TOTAL - 1])


def _snapkv_reference(base: nn.Module, prompt: torch.Tensor, heavy: int, window: int, kernel: int, pooling: str
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """``SnapKVCluster.update_kv``: the prompt entries kept per layer and KV head ``[L, H_kv, PREFIX]``, and the
    pooled scores ``[L, H_kv, PREFIX - window]`` the choice is made on."""
    attentions = torch.stack(base(prompt, output_attentions=True).attentions)[:, 0].float()  # [L, H, Q, K]
    attn_weights_sum = _group_sum(attentions[:, :, -window:, :-window].sum(-2))
    flat = attn_weights_sum.reshape(-1, 1, attn_weights_sum.shape[-1])
    pool = F.max_pool1d if pooling == "max" else F.avg_pool1d
    attn_cache = pool(flat, kernel_size=kernel, padding=kernel // 2, stride=1).view(attn_weights_sum.shape)
    keep = torch.zeros(LAYERS, KV_HEADS, PREFIX, dtype=torch.bool)
    keep[..., -window:] = True
    if heavy + window > PREFIX:
        keep[:] = True
    else:
        keep[..., :-window].scatter_(-1, attn_cache.topk(heavy, dim=-1).indices, True)
    return keep, attn_cache


@pytest.mark.parametrize(("heavy", "window", "kernel"), [(4, 4, 3), (6, 2, 5), (20, 4, 3)])
def test_snapkv_avg_pooling_matches_kvcache_factory(heavy: int, window: int, kernel: int) -> None:
    """With average pooling the scores have no ties, so the kept entries, and the logits of the tokens generated
    after the compression, match the reference exactly."""
    seq = _sequence()
    base = make_tiny_model(attn="eager")
    keep, _ = _snapkv_reference(base, seq[:, :PREFIX], heavy, window, kernel, "avg")
    config = KVEvictConfig(method="snapkv", heavy=heavy, observation_window=window, kernel_size=kernel, pooling="avg")
    logits, cache = _teacher_forced(economize.apply(make_tiny_model(), config), seq)
    for layer in range(LAYERS):
        assert torch.equal(cache.keep[layer][0, :, :PREFIX], keep[layer])
    allowed = _causal()
    allowed[..., PREFIX:, :PREFIX] = keep[:, :, None, :]
    expected, _ = _masked_forward(base, seq, allowed)
    torch.testing.assert_close(logits, expected, rtol=0, atol=2e-4)


def test_snapkv_max_pooling_keeps_the_reference_scores() -> None:
    """Max pooling makes plateaus of equal scores, where ``topk`` breaks ties in an unspecified order and economize
    takes the earlier slot. The pooled scores of the kept entries therefore match as a multiset."""
    heavy, window, kernel = 4, 4, 5
    seq = _sequence()
    keep, pooled = _snapkv_reference(make_tiny_model(attn="eager"), seq[:, :PREFIX], heavy, window, kernel, "max")
    config = KVEvictConfig(method="snapkv", heavy=heavy, observation_window=window, kernel_size=kernel)
    _, cache = _teacher_forced(economize.apply(make_tiny_model(), config), seq)
    for layer in range(LAYERS):
        ours = cache.keep[layer][0, :, :PREFIX]
        assert torch.equal(ours[:, -window:], keep[layer, :, -window:])
        for head in range(KV_HEADS):
            mine = pooled[layer, head][ours[head, :-window]].sort().values
            theirs = pooled[layer, head][keep[layer, head, :-window]].sort().values
            torch.testing.assert_close(mine, theirs, rtol=0, atol=1e-6)
