"""Long-context self-speculation on a sparse view of the cache (MagicDec, TriForce).

Both drafters run the base model with its own weights; their full-attention layers read a StreamingLLM view
(MagicDec) or the retrieved prompt chunks (TriForce) of the shared cache, and the exact pass reads all of it. So in
float32 greedy decoding equals plain greedy decoding for every view, every draft policy and every family. The views
are checked against pure-Python references of the two rules. The default gathered view hands each draft layer only
its budget's entries; it drafts the same tokens as the masked reference view, which hides the other entries in the
mask. The CUDA graph and kernel tests need ``-m gpu``.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import economize
from economize import ConfigError, DecodeConfig
from economize.core.compatibility import SupportStatus
from economize.core.metadata import Saving
from economize.runtime.cache import UNSEEN_POSITION
from economize.techniques.sparse_kv_draft import (
    DRAFT_VIEWS,
    MagicDecConfig,
    MagicDecDrafter,
    TriForceConfig,
    TriForceDrafter,
    chunk_slots,
    gather_entries,
    retrieval_keep,
    sparse_layers,
    streaming_budget,
    streaming_keep,
    tail_width,
)
from economize.techniques.sparse_kv_draft import views as views_module
from economize.techniques.sparse_kv_draft.retrieval import _choose
from helpers import FAMILY_MODELS, VOCAB, family_skip_reason, make_family_model, make_tiny_model, reference_greedy

CUDA = torch.cuda.is_available()
MAX_NEW = 24
LENGTHS = (40, 25, 33)
#: Prompts long enough that TriForce's chunks and tail are fewer entries than the cache, so its view gathers.
LONG_LENGTHS = (120, 97, 110)
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


class _GatherCount:
    """Counts the gathers of the gathered view (each draft layer that reads a budget gathers its keys and values)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls = 0
        original = views_module.gather_entries

        def counted(buffer: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return original(buffer, slots)

        monkeypatch.setattr(views_module, "gather_entries", counted)


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
    prompts = _prompts(lengths=(21, 13, 17) if name == "magicdec" else (90, 70, 80))  # long enough to gather
    greedy = [reference_greedy(base, p, 16).tolist() for p in prompts]
    model = economize.apply(base, CONFIGS[name]())
    try:
        results = model.decode_batch(prompts, 16, decoding=_decoding("chain"))
        assert [r.tokens.tolist() for r in results] == greedy
    finally:
        model.detach()


@torch.no_grad()
def test_triforce_masks_on_a_latent_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """DeepSeek's latent cache shares every entry between the heads, so the per-head chunks are masked (and the
    technique says that the saving is simulated) while MagicDec's head-independent window still gathers."""
    base = make_family_model("deepseek_v3")
    prompts = _prompts(lengths=(90, 70, 80))
    greedy = [reference_greedy(base, p, 16).tolist() for p in prompts]
    for name, gathers in (("triforce", False), ("magicdec", True)):
        model = economize.apply(base, CONFIGS[name]())
        (technique,) = model.techniques.values()
        assert (technique.simulation() is None) is gathers
        counter = _GatherCount(monkeypatch)
        try:
            results = model.decode_batch(prompts, 16, decoding=_decoding("chain"))
            assert [r.tokens.tolist() for r in results] == greedy
        finally:
            model.detach()
        assert (counter.calls > 0) is gathers, name


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


# ---- the gathered view ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("sinks", "window"), [(0, 1), (2, 3), (4, 12), (1, 40), (3, 64)])
def test_the_streaming_budget_reads_the_streaming_view(sinks: int, window: int) -> None:
    """The gathered chain slots, less those the causal mask hides, are the slots :func:`streaming_keep` keeps."""
    positions = torch.tensor([[17], [5], [29]])
    cache = _Cache(31)
    ctx = SimpleNamespace(cache=cache, positions=positions, tree=None)
    budget = streaming_budget(ctx, sinks, window, torch.device("cpu"))  # type: ignore[arg-type]
    assert budget is not None and budget.keep is not None
    assert budget.slots.shape == (3, 1, sinks + window) and budget.width == sinks + window
    keys = cache.key_positions(torch.device("cpu"))
    for b, (p,) in enumerate(positions.tolist()):
        pairs = zip(budget.slots[b, 0].tolist(), budget.keep[b, 0, 0].tolist(), strict=True)
        read = {s for s, kept in pairs if kept and keys[s] <= p}
        assert read == {s for s in range(p + 1) if s < sinks or s > p - window}


def test_the_streaming_budget_reads_the_tree_view() -> None:
    """A tree step gathers the sinks, every node's window and the tree slots; its ``keep`` cuts each node's window."""
    positions = torch.tensor([[20, 21, 21, 22]])
    tree = SimpleNamespace(base=torch.tensor([20]), slots=torch.arange(20, 24)[None])
    ctx = SimpleNamespace(cache=_Cache(41), positions=positions, tree=tree)
    budget = streaming_budget(ctx, 2, 4, torch.device("cpu"))  # type: ignore[arg-type]
    reference = streaming_keep(ctx, 41, 2, 4, torch.device("cpu"))  # type: ignore[arg-type]
    assert budget is not None and budget.keep is not None and budget.width == 2 + 4 - 1 + 4
    for q in range(4):
        pairs = zip(budget.slots[0, 0].tolist(), budget.keep[0, 0, q].tolist(), strict=True)
        assert {s for s, kept in pairs if kept} - {40} == {s for s in range(24) if reference[0, 0, q, s]}, q


def test_a_multi_token_chain_step_has_no_streaming_budget() -> None:
    ctx = SimpleNamespace(cache=_Cache(31), positions=torch.tensor([[5, 6]]), tree=None)
    assert streaming_budget(ctx, 2, 4, torch.device("cpu")) is None  # type: ignore[arg-type]


@pytest.mark.parametrize(("position", "chunk", "sets"), [(30, 4, 3), (17, 3, 2), (8, 4, 9), (40, 5, 1), (3, 4, 2)])
def test_the_chunk_slots_are_the_retrieval_views_chunks(position: int, chunk: int, sets: int) -> None:
    """The gathered chunk slots plus the tail from the last whole chunk on are the slots :func:`retrieval_keep`
    keeps, per KV head."""
    generator = torch.Generator().manual_seed(position + chunk)
    heads_kv, groups, num_keys = 2, 2, 48
    query = torch.randn(1, heads_kv * groups, 1, 8, generator=generator)
    key = torch.randn(1, heads_kv, num_keys, 8, generator=generator)
    at = torch.tensor([position])
    keep = retrieval_keep(query, key, at, None, chunk_size=chunk, select_sets=sets, groups=groups, width=num_keys)
    chosen, region = _choose(query, key, at, chunk_size=chunk, select_sets=sets, groups=groups)
    trash = num_keys - 1
    if chosen is None:  # no whole chunk: every slot is kept
        assert keep.all()
        return
    slots = chunk_slots(chosen, chunk, sets, trash)
    assert slots.shape == (1, heads_kv, sets * chunk)
    tail = set(range(int(region[0]) * chunk, trash))
    for h in range(heads_kv):
        gathered = (set(slots[0, h].tolist()) - {trash}) | tail
        assert gathered == {s for s in range(trash) if keep[0, h * groups, 0, s]}, h


@pytest.mark.parametrize("need", [1, 63, 64, 65, 300, 1000, 5000, 40000])
def test_the_tail_width_buckets_within_a_quarter(need: int) -> None:
    width = tail_width(need)
    assert width % 64 == 0 and need <= width < need + max(64, need // 4)
    assert tail_width(0) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("heads", [1, 2, 4])
def test_gather_entries_picks_the_slots(dtype: torch.dtype, heads: int) -> None:
    """Slots shared by all heads, per KV head of a repeated buffer, or per head; fp8 is gathered bit for bit."""
    buffer = torch.randn(2, 4, 10, 3).to(dtype)
    slots = torch.randint(0, 10, (2, heads, 3), generator=torch.Generator().manual_seed(heads))
    got = gather_entries(buffer, slots)
    assert got.shape == (2, 4, 3, 3) and got.dtype == dtype
    for b in range(2):
        for h in range(4):
            row = slots[b, h * heads // 4]
            assert torch.equal(got[b, h].view(torch.uint8), buffer[b, h, row].view(torch.uint8)), (b, h)


@pytest.mark.parametrize("policy", sorted(POLICIES))
@pytest.mark.parametrize("name", sorted(CONFIGS))
@torch.no_grad()
def test_the_gathered_view_drafts_as_the_masked_view(name: str, policy: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both views read the same entries, so they draft and accept the same tokens, with rows finishing early."""
    base = make_tiny_model()
    prompts = _prompts(lengths=LONG_LENGTHS)
    lengths = [MAX_NEW, 9, 17]
    greedy = [reference_greedy(base, p, n).tolist() for p, n in zip(prompts, lengths, strict=True)]
    count = _GatherCount(monkeypatch)
    runs = {}
    for view in ("masked", "gathered"):
        model = economize.apply(base, replace(CONFIGS[name](), view=view))
        try:
            results = model.decode_batch(prompts, lengths, decoding=_decoding(policy))
        finally:
            model.detach()
        runs[view] = [(r.tokens.tolist(), r.acceptance) for r in results]
        assert [tokens for tokens, _ in runs[view]] == greedy, view
        assert (count.calls > 0) == (view == "gathered"), view
    assert runs["gathered"] == runs["masked"]


@pytest.mark.parametrize("name", sorted(CONFIGS))
@torch.no_grad()
def test_the_gathered_draft_logits_match_the_masked_ones(name: str) -> None:
    """Draft logits of the two views agree step by step to float32 summation noise."""
    base = make_tiny_model()
    prompts = _prompts(lengths=LONG_LENGTHS)
    logits: dict[str, list[torch.Tensor]] = {}
    for view in ("masked", "gathered"):
        model = economize.apply(base, replace(CONFIGS[name](), view=view))
        decoder = model.decoder(_decoding("chain"))
        drafter = decoder.active_drafter
        recorded: list[torch.Tensor] = []
        step = drafter.step

        def recording(*args: Any, _step: Any = step, _out: list[torch.Tensor] = recorded) -> torch.Tensor:
            out = _step(*args)
            _out.append(out.detach().clone())
            return out

        drafter.step = recording  # type: ignore[method-assign]
        try:
            decoder.decode_batch(prompts, MAX_NEW)
        finally:
            del drafter.step
            model.detach()
        logits[view] = recorded
    assert len(logits["masked"]) == len(logits["gathered"]) > 0
    for masked, gathered in zip(logits["masked"], logits["gathered"], strict=True):
        torch.testing.assert_close(gathered, masked, rtol=1e-4, atol=1e-4)


@torch.no_grad()
def test_triforce_gathers_its_chunks_and_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    """After the selecting step, TriForce's draft layers read the chosen chunks and a tail up to the cache's end;
    rows that finish leave the chunks and the tail of the survivors."""
    model = economize.apply(make_tiny_model(), CONFIGS["triforce"]())
    decoder = model.decoder(_decoding("chain"))
    drafter = decoder.active_drafter
    assert isinstance(drafter, TriForceDrafter)
    count = _GatherCount(monkeypatch)
    decoder.decode_batch(_prompts(lengths=LONG_LENGTHS), [MAX_NEW, 5, 12])
    assert count.calls > 0
    path = drafter.path
    tail = path.tail
    assert tail is not None and tail.shape[0] == 1 and tail.shape[1] > 0 and tail.shape[1] % 64 == 0
    chunks = path.chunks(1)
    assert chunks is not None and set(chunks) == set(path.layers)
    assert all(c.shape[-1] == path.select_sets * path.chunk_size for c in chunks.values())


@torch.no_grad()
def test_triforce_masks_when_gathering_would_read_the_whole_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """With short prompts the chunks and the tail are as many entries as the cache, so the draft layers mask."""
    base = make_tiny_model()
    prompts = _prompts()
    model = economize.apply(base, CONFIGS["triforce"]())
    decoder = model.decoder(_decoding("chain"))
    drafter = decoder.active_drafter
    assert isinstance(drafter, TriForceDrafter)
    count = _GatherCount(monkeypatch)
    results = decoder.decode_batch(prompts, MAX_NEW)
    assert [r.tokens.tolist() for r in results] == [reference_greedy(base, p, MAX_NEW).tolist() for p in prompts]
    assert count.calls == 0
    assert drafter.path.tail is not None and drafter.path.tail.shape[-1] == 0


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
    """The default gathered view realizes the saving; the masked reference view simulates it on ``sdpa``."""
    model = economize.apply(make_tiny_model(), CONFIGS[name]())
    (technique,) = model.techniques.values()
    assert technique.info.lossless and technique.info.saves is Saving.LATENCY
    assert technique.simulation() is None
    masked = economize.apply(make_tiny_model(), replace(CONFIGS[name](), view="masked"))
    (technique,) = masked.techniques.values()
    simulation = technique.simulation()
    assert simulation is not None and "reads" in simulation.reason


@pytest.mark.parametrize("build", [MagicDecConfig, TriForceConfig])
def test_the_default_configs_report_no_simulated_backend(build: Any) -> None:
    """The default configurations realize their saving on transformers and every other backend refuses them with
    its reason; only the explicit masked view reports a simulation."""
    support = economize.backend_support(build())
    assert [name for name, entry in support.items() if entry.status is SupportStatus.SIMULATED] == []
    assert support["transformers"].status is SupportStatus.REAL
    for name, entry in support.items():
        assert entry.status is SupportStatus.REAL or entry.detail, name
    for name in ("gguf", "mlx"):  # the family default names hidden states, which these drafters do not read
        assert "hidden states" not in support[name].detail, name
    masked = economize.backend_support(build(view="masked"))
    assert masked["transformers"].status is SupportStatus.SIMULATED


def test_the_views_are_registered() -> None:
    assert {"gathered", "masked"} <= set(DRAFT_VIEWS.names())
    with pytest.raises(ConfigError):
        MagicDecConfig(view="nope")


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
