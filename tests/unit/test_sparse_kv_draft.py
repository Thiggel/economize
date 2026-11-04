"""Long-context self-speculation on a sparse view of the cache (MagicDec, TriForce).

Both drafters run the base model with its own weights; their full-attention layers read a StreamingLLM view
(MagicDec) or the retrieved prompt chunks (TriForce) of the shared cache, and the exact pass reads all of it. So in
float32 greedy decoding equals plain greedy decoding for every view, every draft policy and every family. The views
are checked against pure-Python references of the two rules. The CUDA graph and kernel tests need ``-m gpu``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import economize
from economize import ConfigError, DecodeConfig
from economize.core.metadata import Saving
from economize.runtime.cache import UNSEEN_POSITION
from economize.techniques.sparse_kv_draft import (
    MagicDecConfig,
    MagicDecDrafter,
    TriForceConfig,
    TriForceDrafter,
    retrieval_keep,
    sparse_layers,
    streaming_keep,
)
from helpers import FAMILY_MODELS, VOCAB, family_skip_reason, make_family_model, make_tiny_model, reference_greedy

CUDA = torch.cuda.is_available()
MAX_NEW = 24
LENGTHS = (40, 25, 33)
CONFIGS = {
    "magicdec": lambda: MagicDecConfig(budget=8, sinks=2),
    "triforce": lambda: TriForceConfig(budget=8, chunk_size=2),
}
DRAFTER_CLASS = {"magicdec": MagicDecDrafter, "triforce": TriForceDrafter}
POLICIES = {
    "chain": {},
    "dynamic_tree": {"draft_policy": "dynamic_tree"},
    "sequoia": {"draft_policy": "sequoia",
                "draft_policy_options": {"acceptance": (0.6, 0.2, 0.1), "total_tokens": 10, "depth": 4}},
    "specexec": {"draft_policy": "specexec",
                 "draft_policy_options": {"depth": 4, "width": 2, "topk": 3, "total_tokens": 10}},
}


def _prompts(device: str = "cpu", lengths: tuple[int, ...] = LENGTHS) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    return [torch.randint(3, VOCAB, (n,), generator=generator).to(device) for n in lengths]


def _decoding(policy: str, **extra: Any) -> DecodeConfig:
    return DecodeConfig(cycle_length=5, **POLICIES[policy], **extra)


def _accepted(results: list[Any]) -> float:
    cycles = [a for r in results for _, a in r.acceptance]
    return sum(cycles) / len(cycles)


# ---- losslessness ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("policy", sorted(POLICIES))
@pytest.mark.parametrize("name", sorted(CONFIGS))
@torch.no_grad()
def test_the_sparse_drafters_are_lossless_in_float32(name: str, policy: str) -> None:
    base = make_tiny_model()
    prompts = _prompts()
    greedy = [reference_greedy(base, p, MAX_NEW).tolist() for p in prompts]
    model = economize.apply(base, CONFIGS[name]())
    results = model.decode_batch(prompts, MAX_NEW, decoding=_decoding(policy))
    assert [r.tokens.tolist() for r in results] == greedy
    assert model.decode(prompts[1], MAX_NEW, decoding=_decoding(policy)).tokens.tolist() == greedy[1]
    drafter = model.decoder(_decoding("chain")).active_drafter
    assert isinstance(drafter, DRAFTER_CLASS[name])


@pytest.mark.parametrize("name", sorted(CONFIGS))
@torch.no_grad()
def test_rows_that_finish_early_keep_the_others_lossless(name: str) -> None:
    """Rows of different lengths drop out of the batch; TriForce keeps the selection of the surviving rows."""
    base = make_tiny_model()
    prompts = _prompts()
    lengths = [7, 24, 15]
    greedy = [reference_greedy(base, p, n).tolist() for p, n in zip(prompts, lengths, strict=True)]
    model = economize.apply(base, CONFIGS[name]())
    for policy in ("chain", "sequoia"):
        results = model.decode_batch(prompts, lengths, decoding=_decoding(policy))
        assert [r.tokens.tolist() for r in results] == greedy, policy


@torch.no_grad()
def test_magicdec_is_lossless_on_eager_attention() -> None:
    base = make_tiny_model(attn="eager")
    prompts = _prompts()
    greedy = [reference_greedy(base, p, MAX_NEW).tolist() for p in prompts]
    model = economize.apply(base, CONFIGS["magicdec"]())
    for policy in ("chain", "specexec"):
        results = model.decode_batch(prompts, MAX_NEW, decoding=_decoding(policy))
        assert [r.tokens.tolist() for r in results] == greedy, policy


@pytest.mark.parametrize("family", sorted(FAMILY_MODELS))
@pytest.mark.parametrize("name", sorted(CONFIGS))
@torch.no_grad()
def test_the_sparse_drafters_are_lossless_on_every_family(name: str, family: str) -> None:
    """Families without a full-attention layer, or on an attention path the drafter cannot mask, are refused."""
    reason = family_skip_reason(family)
    if reason is not None:
        pytest.skip(reason)
    base = make_family_model(family)
    adapter = economize.arch.adapter_for(base)
    allowed = ("sdpa", "eager") if name == "magicdec" else ("sdpa",)
    if not sparse_layers(adapter) or adapter.attn_implementation not in allowed:
        with pytest.raises(ConfigError):
            economize.apply(base, CONFIGS[name]())
        return
    prompts = _prompts(lengths=(21, 13, 17))
    greedy = [reference_greedy(base, p, 16).tolist() for p in prompts]
    model = economize.apply(base, CONFIGS[name]())
    try:
        results = model.decode_batch(prompts, 16, decoding=_decoding("chain"))
        assert [r.tokens.tolist() for r in results] == greedy
    finally:
        model.detach()


# ---- what the views keep --------------------------------------------------------------------------------------------
class _Cache:
    def __init__(self, num_slots: int) -> None:
        self.num_slots = num_slots

    def key_positions(self, device: torch.device) -> torch.Tensor:
        keys = torch.arange(self.num_slots, device=device)
        keys[-1] = UNSEEN_POSITION
        return keys


@pytest.mark.parametrize(("sinks", "window"), [(0, 1), (2, 3), (4, 12), (1, 40)])
def test_the_streaming_view_matches_streaming_llm(sinks: int, window: int) -> None:
    positions = torch.tensor([[17, 18, 19], [5, 6, 7]])
    ctx = SimpleNamespace(cache=_Cache(31), positions=positions, tree=None)
    keep = streaming_keep(ctx, 31, sinks, window, torch.device("cpu"))  # type: ignore[arg-type]
    assert keep.shape == (2, 1, 3, 31)
    for b, row in enumerate(positions.tolist()):
        for q, p in enumerate(row):
            # the causal mask hides the keys after the query (and the trash slot), whatever the view keeps
            expected = [s < sinks or s > p - window for s in range(p + 1)]
            assert keep[b, 0, q, : p + 1].tolist() == expected


def test_the_streaming_view_keeps_the_tree() -> None:
    """In a draft tree the query reads the sinks, the window before the tree's base and every tree slot (which the
    tree mask restricts to its ancestors)."""
    positions = torch.tensor([[20, 21, 21, 22]])
    tree = SimpleNamespace(base=torch.tensor([20]))
    ctx = SimpleNamespace(cache=_Cache(41), positions=positions, tree=tree)
    keep = streaming_keep(ctx, 41, 2, 4, torch.device("cpu"))  # type: ignore[arg-type]
    for q, p in enumerate(positions[0].tolist()):
        got = keep[0, 0, q].tolist()
        assert [s for s in range(20) if got[s]] == [0, 1, *range(p - 3, 20)]
        assert all(got[20:])


def _reference_retrieval(query: torch.Tensor, key: torch.Tensor, position: int, chunk: int, sets: int,
                         groups: int) -> list[list[bool]]:
    """TriForce's rule written out per KV head: score whole prompt chunks by the summed group query against the
    chunk's mean key, keep chunk 0 and the best ``sets - 1`` others, and every slot after the last whole chunk."""
    heads_kv = query.shape[0] // groups
    num_keys = key.shape[1]
    whole = min(position // chunk, num_keys // chunk)
    result = []
    for h in range(heads_kv):
        kv = key[h] if key.shape[0] == heads_kv else key[h * groups]
        scores = []
        for c in range(1, whole):
            mean = kv[c * chunk:(c + 1) * chunk].mean(0)
            scores.append((sum(float(query[h * groups + g] @ mean) for g in range(groups)), c))
        best = {c for _, c in sorted(scores, key=lambda t: (-t[0], t[1]))[: max(sets - 1, 0)]}
        chosen = ({0} | best) if whole else set()
        result.append([(s // chunk in chosen and s < whole * chunk) or s >= whole * chunk for s in range(num_keys)])
    return result


@pytest.mark.parametrize(("position", "chunk", "sets", "groups", "repeated"),
                         [(30, 4, 3, 2, False), (30, 4, 3, 2, True), (17, 3, 2, 1, False), (8, 4, 9, 4, False),
                          (3, 4, 2, 2, False), (40, 5, 1, 2, False)])
def test_the_retrieval_view_matches_the_rule(position: int, chunk: int, sets: int, groups: int,
                                             repeated: bool) -> None:
    generator = torch.Generator().manual_seed(position)
    heads_kv, num_keys = 2, 32
    query = torch.randn(1, heads_kv * groups, 1, 8, generator=generator)
    key = torch.randn(1, heads_kv, num_keys, 8, generator=generator)
    if repeated:  # eager-style keys, already repeated to the query heads
        key = key.repeat_interleave(groups, dim=1)
    keep = retrieval_keep(query, key, torch.tensor([position]), None, chunk_size=chunk, select_sets=sets,
                          groups=groups, width=num_keys + 3)
    assert keep.shape == (1, heads_kv * groups, 1, num_keys + 3)
    expected = _reference_retrieval(query[0, :, 0], key[0], position, chunk, sets, groups)
    for h in range(heads_kv * groups):
        assert keep[0, h, 0, :num_keys].tolist() == expected[h // groups], h
        assert keep[0, h, 0, num_keys:].all()


def _official_triforce_chunks(query: torch.Tensor, prefill_keys: torch.Tensor, chunk_size: int,
                              select_sets: int) -> torch.Tensor:
    """The chunk selection of the official TriForce code, transcribed with its tensor layout.

    Infini-AI-Lab/TriForce @ 164c8c0, ``models/cache.py``, ``RetrievalCache.init_graph_cache`` (lines 146-175):
    ``query_states`` is ``[1, 1, H, D]`` and the prefill keys are ``[1, prefill, H, D]`` (multi-head attention, one
    query head per key head). Returns the selected chunk indices, ``[1, H, select_sets]``.
    """
    num_heads, head_dim = prefill_keys.shape[2], prefill_keys.shape[3]
    chunks = prefill_keys.shape[1] // chunk_size
    chunk_k = prefill_keys.view(1, chunks, chunk_size, num_heads, head_dim).mean(dim=-3)
    chunk_attn = torch.matmul(query.permute(0, 2, 1, 3), chunk_k.permute(0, 2, 3, 1)).squeeze(2)
    _, topk_idx_rest = torch.topk(chunk_attn[:, :, 1:], k=select_sets - 1, dim=-1)
    topk_idx_rest += 1
    topk_idx_first = torch.zeros((topk_idx_rest.shape[0], topk_idx_rest.shape[1], 1), dtype=topk_idx_rest.dtype)
    return torch.cat([topk_idx_first, topk_idx_rest], dim=-1)


@pytest.mark.parametrize(("prefill", "chunk", "sets"), [(64, 8, 4), (96, 16, 2), (40, 4, 10), (128, 8, 2)])
def test_the_retrieval_view_matches_the_official_triforce_selection(prefill: int, chunk: int, sets: int) -> None:
    """With one query head per KV head and a prompt of whole chunks, the kept prompt slots are exactly the chunks of
    the official ``init_graph_cache``; the generated slots after the prompt are kept (TriForce appends them too)."""
    generator = torch.Generator().manual_seed(prefill + chunk)
    heads, dim, generated = 4, 16, 5
    query = torch.randn(1, 1, heads, dim, generator=generator)  # official layout [bsz, 1, H, D]
    keys = torch.randn(1, prefill + generated, heads, dim, generator=generator)  # official layout [bsz, seq, H, D]
    official = _official_triforce_chunks(query, keys[:, :prefill], chunk, sets)
    keep = retrieval_keep(query.permute(0, 2, 1, 3), keys.permute(0, 2, 1, 3), torch.tensor([prefill]), None,
                          chunk_size=chunk, select_sets=sets, groups=1, width=prefill + generated)
    for h in range(heads):
        kept = keep[0, h, 0]
        chosen = sorted({s // chunk for s in range(prefill) if kept[s]})
        assert chosen == sorted(official[0, h].tolist()), h
        assert all(kept[s] == (s // chunk in chosen) for s in range(prefill))
        assert kept[prefill:].all()


def test_the_retrieval_view_keeps_every_slot_of_an_invalid_row() -> None:
    query, key = torch.randn(2, 2, 1, 4), torch.randn(2, 2, 16, 4)
    keep = retrieval_keep(query, key, torch.tensor([14, 14]), torch.tensor([True, False]), chunk_size=2,
                          select_sets=2, groups=1, width=16)
    assert not keep[0].all() and keep[1].all()


# ---- acceptance -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(CONFIGS))
@torch.no_grad()
def test_a_budget_covering_the_context_drafts_the_exact_model(name: str) -> None:
    """With a budget beyond the context the view is the whole cache, so every cycle accepts all four drafts; a
    small budget accepts fewer."""
    base = make_tiny_model()
    base.lm_head.weight.data *= 30
    prompts = _prompts()
    greedy = [reference_greedy(base, p, MAX_NEW).tolist() for p in prompts]
    full = MagicDecConfig(budget=128, sinks=4) if name == "magicdec" else TriForceConfig(budget=128, chunk_size=4)
    model = economize.apply(base, full)
    results = model.decode_batch(prompts, MAX_NEW, decoding=_decoding("chain"))
    assert [r.tokens.tolist() for r in results] == greedy
    for result in results:
        accepted = [a for _, a in result.acceptance]
        assert accepted[:-1] == [4] * (len(accepted) - 1) and accepted
    model.detach()
    small = economize.apply(base, CONFIGS[name]())
    results = small.decode_batch(prompts, MAX_NEW, decoding=_decoding("chain"))
    assert [r.tokens.tolist() for r in results] == greedy
    assert _accepted(results) < 4


@torch.no_grad()
def test_triforce_selects_once_per_generation() -> None:
    model = economize.apply(make_tiny_model(), CONFIGS["triforce"]())
    prompts = _prompts()
    decoder = model.decoder(_decoding("chain"))
    drafter = decoder.active_drafter
    assert isinstance(drafter, TriForceDrafter)
    decoder.decode_batch(prompts, MAX_NEW)
    assert drafter.path.selections == 1
    decoder.decode_batch(prompts[:1], MAX_NEW)
    assert drafter.path.selections == 2
    buffers = drafter.path.buffer(1, next(iter(drafter.path._buffers))[1])
    assert buffers is not None and set(buffers) == set(drafter.path.layers)


# ---- configuration and metadata -------------------------------------------------------------------------------------
@pytest.mark.parametrize("build", [
    lambda: MagicDecConfig(budget=4, sinks=4),
    lambda: MagicDecConfig(budget=0),
    lambda: MagicDecConfig(sinks=-1),
    lambda: TriForceConfig(budget=32, chunk_size=64),
    lambda: TriForceConfig(chunk_size=0),
])
def test_invalid_configs_are_rejected(build: Any) -> None:
    with pytest.raises(ConfigError):
        build()


def test_triforce_refuses_eager_attention() -> None:
    with pytest.raises(ConfigError, match="sdpa"):
        economize.apply(make_tiny_model(attn="eager"), CONFIGS["triforce"]())


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_the_metadata_describes_a_lossless_latency_drafter(name: str) -> None:
    model = economize.apply(make_tiny_model(), CONFIGS[name]())
    (technique,) = model.techniques.values()
    assert technique.info.lossless and technique.info.saves is Saving.LATENCY
    simulation = technique.simulation()
    assert simulation is not None and "reads" in simulation.reason  # sdpa masks the dropped entries


# ---- CUDA graphs and the economize kernel ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy", ["chain", "sequoia", "specexec"])
@pytest.mark.parametrize("name", sorted(CONFIGS))
@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
@torch.no_grad()
def test_the_sparse_drafters_under_cuda_graphs_are_lossless_and_equal_eager(name: str, policy: str) -> None:
    base = make_tiny_model().to("cuda")
    prompts = _prompts("cuda")
    greedy = [reference_greedy(base, p, MAX_NEW).tolist() for p in prompts]
    model = economize.apply(base, CONFIGS[name]())
    eager = model.decode_batch(prompts, MAX_NEW, decoding=_decoding(policy))
    summary = [(r.tokens.tolist(), r.acceptance) for r in eager]
    for _ in range(3):  # eager warm-up, capture, replay
        graphed = model.decode_batch(prompts, MAX_NEW, decoding=_decoding(policy, cuda_graphs=True))
        assert [(r.tokens.tolist(), r.acceptance) for r in graphed] == summary
        assert [r.tokens.tolist() for r in graphed] == greedy


@pytest.mark.parametrize("policy", ["chain", "sequoia"])
@pytest.mark.parametrize("name", sorted(CONFIGS))
@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
@torch.no_grad()
def test_the_sparse_drafters_run_on_the_economize_attention_kernel(name: str, policy: str) -> None:
    from economize.attention import install, kernel_usage, usage_summary

    install()
    base = make_tiny_model().to("cuda").eval()
    prompts = _prompts("cuda", lengths=(300, 200, 260))
    greedy = [reference_greedy(base, p, MAX_NEW).tolist() for p in prompts]
    base.set_attn_implementation("economize")
    config = MagicDecConfig(budget=64, sinks=4) if name == "magicdec" else TriForceConfig(budget=128, chunk_size=64)
    model = economize.apply(base, config)
    (technique,) = model.techniques.values()
    assert technique.simulation() is None
    kernel_usage(reset=True)
    for graphs in (False, True, True):
        decoding = _decoding(policy, cuda_graphs=graphs)
        assert [r.tokens.tolist() for r in model.decode_batch(prompts, MAX_NEW, decoding=decoding)] == greedy
    summary = usage_summary()
    assert summary["calls"].get("kernel", 0) > 0, summary
