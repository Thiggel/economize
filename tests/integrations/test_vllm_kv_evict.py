"""StreamingLLM on vLLM (:mod:`economize.backends.vllm.kv_evict`): plans, masks and freed pages, checked on CPU.

The GPU tests in ``test_vllm_serving_gpu.py`` decode with the backend and compare it with the economize HF runtime.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytest.importorskip("vllm")

import economize
from economize.backends import UnsupportedBackendError
from economize.backends.vllm import plugin, serving
from economize.backends.vllm.kv_evict import BACKEND_PATH
from economize.backends.vllm.kv_evict.backend import (
    CONFIG_KEY,
    SINGLE_PASS,
    UNBOUNDED,
    EconomizeStreamingMetadataBuilder,
    StreamingMetadata,
    StreamingSettings,
    compact_block_table,
    dropped_pages,
    prefix_lens,
    settings_of,
    streaming_spec,
)
from economize.backends.vllm.kv_evict.spec import StreamingManager, StreamingSpec, freed_block_range
from economize.backends.vllm.serving import Role, check_servable, support_table, to_vllm
from economize.errors import ConfigError, IncompatibleTechniquesError
from economize.techniques.draft_model.config import DraftModelConfig
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.kv_evict.policies import PassInfo, StreamingEviction
from economize.techniques.kv_evict.technique import KVEvictConfig
from economize.techniques.kv_quant.fp8 import FP8KVConfig
from economize.techniques.lookup.config import LookupConfig
from economize.techniques.rest.config import RestConfig
from helpers import make_tiny_model

STREAMING = KVEvictConfig(method="streaming", sinks=4, recent=16)


@pytest.fixture
def tiny_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "tiny"
    make_tiny_model(seed=0).save_pretrained(folder)
    return folder


def _base(folder: Path, **config: Any) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32, **config).eval()


# ---- the mask is the HF runtime's keep set ----------------------------------------------------------------------


def _hf_readable(sinks: int, recent: int, query: int) -> set[int]:
    """Slots a decoding query at ``query`` reads in economize's runtime (``StreamingEviction`` before its pass)."""
    method = StreamingEviction(KVEvictConfig(method="streaming", sinks=sinks, recent=recent))
    keep = torch.ones(1, 1, query, dtype=torch.bool)
    info = PassInfo("decode", torch.tensor([query]), torch.tensor([query + 1]))
    method.before_pass(keep, None, info)
    return {int(slot) for slot in keep[0, 0].nonzero()[:, 0]} | {query}


def _vllm_readable(settings: StreamingSettings, query: int, prefilling: bool) -> set[int]:
    """Slots vLLM's Triton R-SWA mask lets ``query`` read: ``j < prefix_len`` or ``query - j < window``."""
    prefix = int(prefix_lens(torch.tensor([prefilling]), 1, settings.sinks)[0])
    return {j for j in range(query + 1) if j < prefix or query - j < settings.window}


@pytest.mark.parametrize(("sinks", "recent"), [(4, 16), (0, 8), (3, 0), (16, 4)])
def test_decoding_mask_is_the_runtime_keep_set(sinks: int, recent: int) -> None:
    settings = StreamingSettings(sinks=sinks, recent=recent)
    for query in range(1, 60):
        assert _vllm_readable(settings, query, prefilling=False) == _hf_readable(sinks, recent, query), query


def test_prefill_reads_the_whole_prompt() -> None:
    settings = StreamingSettings(sinks=4, recent=8)
    assert _vllm_readable(settings, 40, prefilling=True) == set(range(41))
    lens = prefix_lens(torch.tensor([True, False, True]), 5, 4)  # rows beyond the flags are padding
    assert lens.tolist() == [UNBOUNDED, 4, UNBOUNDED, 4, 4]
    with pytest.raises(RuntimeError, match=r"V2 model runner"):
        prefix_lens(None, 2, 4)


# ---- compacted block tables -------------------------------------------------------------------------------------


def _visible_slots(table: torch.Tensor, seq: int, queries: int, prefix: int, window: int, page: int) -> list[set]:
    """The physical slots ``(page id, offset)`` each query of a request reads under vLLM's R-SWA mask, from its
    block table row and sequence length (the Triton kernel's ``context = seq - queries``)."""
    seen = []
    for index in range(queries):
        query = seq - queries + index
        keys = [j for j in range(min(query + 1, seq)) if j < prefix or query - j < window]
        seen.append({(int(table[j // page]), j % page) for j in keys})
    return seen


@pytest.mark.parametrize("page", [1, 4, 16])
@pytest.mark.parametrize(("sinks", "recent"), [(1, 0), (4, 16), (5, 3), (17, 40), (4, 200)])
def test_compaction_reads_the_same_keys(page: int, sinks: int, recent: int) -> None:
    generator = torch.Generator().manual_seed(page * 1000 + sinks * 10 + recent)
    rows, width = 12, 64
    window = recent + 1
    queries = torch.randint(1, 6, (rows,), generator=generator)
    queries[:3] = 1
    seq = torch.minimum(torch.randint(1, width * page, (rows,), generator=generator), torch.tensor(width * page))
    seq = torch.maximum(seq, queries)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), queries.cumsum(0)])
    decoding = torch.rand(rows, generator=generator) < 0.75
    table = torch.stack([torch.randperm(10_000, generator=generator)[:width] for _ in range(rows)]).to(torch.int32)
    drop = dropped_pages(starts, seq, decoding, sinks, recent, page)
    compact = torch.empty_like(table)
    compact_block_table(table, drop, sinks, page, compact)
    short = seq - drop * page
    sink_pages = -(-sinks // page)
    for row in range(rows):
        prefix = sinks if decoding[row] else UNBOUNDED
        n = int(queries[row])
        assert _visible_slots(compact[row], int(short[row]), n, prefix, window, page) == _visible_slots(
            table[row], int(seq[row]), n, prefix, window, page), row
        assert int(compact[row, 0]) == int(table[row, 0])  # key 0 (a visible sink) opens the first tile
        if decoding[row]:  # the kernel visits the sink pages and the window, plus at most one page in between
            assert int(short[row]) <= sink_pages * page + recent + n + page - 1
        else:
            assert int(drop[row]) == 0


def test_compaction_keeps_short_requests() -> None:
    starts = torch.tensor([0, 1, 2])
    drop = dropped_pages(starts, torch.tensor([20, 21]), torch.tensor([True, True]), sinks=4, recent=16, page=4)
    assert drop.tolist() == [0, 0] and drop.dtype == torch.int32  # the window reaches into the sink page
    drop = dropped_pages(starts, torch.tensor([40, 0]), torch.tensor([True, True]), sinks=4, recent=16, page=4)
    assert drop.tolist() == [4, 0]  # first query 39 reads keys 23..39 (page 5 on); pages 1..4 go, padding stays


def _triton_metadata(seq: torch.Tensor, starts: torch.Tensor, table: torch.Tensor) -> Any:
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata

    return TritonAttentionMetadata(
        num_actual_tokens=int(starts[-1]), max_query_len=1, query_start_loc=starts, max_seq_len=int(seq.max()),
        seq_lens=seq, block_table=table, slot_mapping=torch.zeros(1), seq_threshold_3D=8, num_par_softmax_segments=16,
        softmax_segm_output=torch.zeros(1), softmax_segm_max=torch.zeros(1), softmax_segm_expsum=torch.zeros(1),
        causal=True, use_cascade=False, common_prefix_len=0, cu_prefix_query_lens=None, prefix_kv_lens=None,
        suffix_kv_lens=None)


def _streaming_builder(settings: StreamingSettings) -> EconomizeStreamingMetadataBuilder:
    builder = object.__new__(EconomizeStreamingMetadataBuilder)
    builder.streaming = settings
    builder._max_rows = 4
    builder._table = builder._seq_lens = builder._dropped = None
    builder.block_size, builder.kernel_block_size = 64, 4
    return builder


def test_builder_compacts_decoding_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder

    seq = torch.tensor([40, 40, 3], dtype=torch.int32)
    starts = torch.tensor([0, 1, 4, 5], dtype=torch.int32)
    table = torch.arange(3 * 12, dtype=torch.int32).view(3, 12) + 100

    def build(self: Any, common_prefix_len: int, common: Any, fast_build: bool = False) -> Any:
        metadata = _triton_metadata(common.seq_lens, common.query_start_loc, common.block_table_tensor)
        metadata.rswa_prefix_lens = common.rswa_prefix_lens
        metadata.mm_prefix_range = common.mm_ranges
        return metadata

    monkeypatch.setattr(TritonAttentionMetadataBuilder, "build", build)
    monkeypatch.setattr("economize.backends.vllm.kv_evict.backend.replace",
                        lambda obj, **kw: SimpleNamespace(**{**vars(obj), **kw}))
    common = SimpleNamespace(is_prefilling=torch.tensor([False, False, True]), num_reqs=3, rswa_prefix_lens=None,
                             seq_lens=seq.clone(), query_start_loc=starts, block_table_tensor=table, mm_ranges=None)
    builder = _streaming_builder(StreamingSettings(sinks=4, recent=16))
    metadata = builder.build(0, common)
    assert isinstance(metadata, StreamingMetadata) and metadata.seq_threshold_3D == SINGLE_PASS
    # request 0 decodes (query 39, keys 23..39: drop pages 1..4), request 1 verifies 3 queries (first 37, keys
    # 21..: drop pages 1..4), request 2 prefills and keeps its table
    assert metadata.dropped_tokens.tolist() == [16, 16, 0]
    assert metadata.seq_lens.tolist() == [24, 24, 3]
    assert metadata.block_table[0, :4].tolist() == [100, 105, 106, 107]
    assert metadata.block_table[2].tolist() == table[2].tolist()
    assert metadata.full_seq_lens is common.seq_lens
    pointers = (metadata.seq_lens.data_ptr(), metadata.block_table.data_ptr())
    common.seq_lens += 2  # the draft loop advances vLLM's lengths in place
    builder.update_draft_decode_metadata(metadata)
    assert metadata.seq_lens.tolist() == [26, 26, 5]
    again = builder.build(0, common)
    assert (again.seq_lens.data_ptr(), again.block_table.data_ptr()) == pointers  # graphs keep reading the buffers
    common.mm_ranges = {0: [(0, 3)]}
    assert builder.build(0, common).dropped_tokens.tolist() == [0, 0, 0]  # multimodal batches keep the table
    common.mm_ranges = None
    assert builder.build(7, common).dropped_tokens.tolist() == [0, 0, 0]  # so does a common prefix
    captured = builder.build_for_cudagraph_capture(common)
    assert captured.seq_lens.tolist() == [1, 1, 1] and captured.dropped_tokens.tolist() == [0, 0, 0]
    assert common.seq_lens.tolist() == [1, 1, 1]  # vLLM's capture lengths, as for its own builder


def test_layer_subsets_keep_the_table() -> None:
    builder = _streaming_builder(StreamingSettings(sinks=4, recent=16, layers=frozenset({2})))
    assert not builder.compacts
    assert _streaming_builder(StreamingSettings(sinks=4, recent=16)).compacts


# ---- freed pages ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("sinks", "recent", "retain"), [(4, 16, 0), (4, 16, 2), (0, 5, 0), (17, 3, 1)])
def test_freed_blocks_are_never_read_again(sinks: int, recent: int, retain: int) -> None:
    block = 4
    prompt = 10
    for processed in range(80):
        first, last = freed_block_range(sinks, recent, retain, block, processed, prompt)
        assert first <= last
        if processed < prompt:
            assert first == last == 0  # prefilling queries read every prompt token
            continue
        freed = set(range(first * block, last * block))
        # every later query (position >= processed) and every draft position down to processed - retain
        for query in range(processed - retain, processed + 40):
            assert not freed & _hf_readable(sinks, recent, query), (processed, query)
        if last > first:  # the range is the largest one: the next block holds a token some query still reads
            assert (last * block) + block > processed - recent - retain
    assert freed_block_range(4, 16, 0, 4, 100, None) == (0, 0)


def test_streaming_spec_replaces_full_attention_only() -> None:
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    full = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float32)
    settings = StreamingSettings(sinks=4, recent=16, retain=2)
    spec = streaming_spec(full, settings)
    assert isinstance(spec, StreamingSpec) and (spec.sinks, spec.recent, spec.retain) == (4, 16, 2)
    assert spec.page_size_bytes == full.page_size_bytes and spec.block_size == 16
    assert StreamingSpec.merge([spec, spec]) == spec
    assert streaming_spec(full, None) is full
    assert streaming_spec(full, StreamingSettings(sinks=4, recent=16, layers=frozenset({1}))) is full
    sliding = SlidingWindowSpec(block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float32, sliding_window=32)
    assert streaming_spec(sliding, settings) is sliding
    chunked = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float32,
                                attention_chunk_size=64)
    assert streaming_spec(chunked, settings) is chunked
    other = StreamingSpec(block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float32, sinks=4, recent=8)
    with pytest.raises(ConfigError, match="one window"):
        StreamingSpec.merge([spec, other])


def test_settings_follow_the_engine_config() -> None:
    block = {"method": "streaming", "sinks": 4, "recent": 16, "layers": [2, 3]}
    assert settings_of(SimpleNamespace(additional_config={})) is None
    settings = settings_of(SimpleNamespace(additional_config={CONFIG_KEY: block}, speculative_config=None))
    assert settings == StreamingSettings(sinks=4, recent=16, layers=frozenset({2, 3}), retain=0)
    assert settings.window == 17
    speculative = SimpleNamespace(num_speculative_tokens=5)
    config = SimpleNamespace(additional_config={CONFIG_KEY: block}, speculative_config=speculative)
    assert settings_of(config).retain == 4


# ---- plans ------------------------------------------------------------------------------------------------------


def test_plan_serves_streaming_on_the_custom_backend(tiny_dir: Path, tmp_path: Path) -> None:
    plan = to_vllm(economize.apply(_base(tiny_dir), STREAMING), tmp_path / "out")
    assert plan.techniques == {"kv_evict": "plugin"}
    assert plan.engine["attention_backend"] == "CUSTOM"
    assert plan.engine["enable_prefix_caching"] is False
    assert plan.engine["additional_config"][CONFIG_KEY] == {"method": "streaming", "sinks": 4, "recent": 16,
                                                            "layers": []}
    assert plan.env["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert any("freed" in note for note in plan.notes["kv_evict"])
    kwargs = plan.llm_kwargs()
    assert "attention_config" not in kwargs  # attention_backend and attention_config.backend exclude each other


def test_default_sinks_and_layer_subset(tiny_dir: Path, tmp_path: Path) -> None:
    plan = to_vllm(economize.apply(_base(tiny_dir), KVEvictConfig(recent=32, layers=(3, 1))), tmp_path / "out")
    assert plan.engine["additional_config"][CONFIG_KEY] == {"method": "streaming", "sinks": 4, "recent": 32,
                                                            "layers": [1, 3]}
    assert any("masked only" in note for note in plan.notes["kv_evict"])


def test_streaming_with_a_drafter_and_fp8_kv(tiny_dir: Path, tmp_path: Path) -> None:
    model = economize.apply(_base(tiny_dir), STREAMING, FP8KVConfig(scale="token"), Eagle3Config())
    plan = to_vllm(model, tmp_path / "out", num_speculative_tokens=3)
    assert plan.engine["attention_backend"] == "CUSTOM"  # the one backend both accept
    assert plan.engine["kv_cache_dtype"] == "fp8_per_token_head"
    assert plan.env["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert plan.speculative is not None and plan.speculative["method"] == "eagle3"
    assert any("draft model" in note for note in plan.notes["kv_evict"])


@pytest.mark.parametrize("drafter", ["draft_model", "lookup", "rest"])
def test_drafters_the_v2_runner_lacks_are_refused(tiny_dir: Path, tmp_path: Path, drafter: str) -> None:
    """vLLM 0.30's V2 model runner (which the streaming backend needs) runs no draft_model, ngram or custom_class."""
    configs = {"draft_model": lambda: DraftModelConfig(model=str(tiny_dir)), "lookup": LookupConfig,
               "rest": lambda: RestConfig(datastore=str(tmp_path))}
    model = economize.apply(_base(tiny_dir), STREAMING, configs[drafter]())
    with pytest.raises(UnsupportedBackendError, match=r"V2 model runner.*speculative method"):
        check_servable(model)


def test_fp8_token_scale_alone_runs_on_triton(tiny_dir: Path, tmp_path: Path) -> None:
    plan = to_vllm(economize.apply(_base(tiny_dir), FP8KVConfig(scale="token")), tmp_path / "out")
    assert plan.engine["attention_backend"] == "TRITON_ATTN"


def test_attention_requirements_must_meet(tiny_dir: Path) -> None:
    builder = serving.PlanBuilder(_base(tiny_dir), None)
    assert builder.resolve_attention() is None
    builder.require_attention("TRITON_ATTN", "CUSTOM", owner="kv_fp8")
    builder.require_attention("CUSTOM", owner="kv_evict")
    assert builder.resolve_attention() == "CUSTOM"
    builder.require_attention("FLASH_ATTN", owner="other")
    with pytest.raises(IncompatibleTechniquesError, match=r"kv_evict.*CUSTOM"):
        builder.resolve_attention()


# ---- refusals ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["h2o", "h2o_prefill", "snapkv", "pyramidkv"])
def test_score_based_methods_are_refused(tiny_dir: Path, method: str) -> None:
    model = economize.apply(_base(tiny_dir), KVEvictConfig(method=method))
    with pytest.raises(UnsupportedBackendError, match="per KV head"):
        check_servable(model)


def test_wide_sliding_window_is_refused(tmp_path: Path) -> None:
    folder = tmp_path / "sliding"
    model = make_tiny_model(seed=0)
    model.config.layer_types = ["full_attention", "sliding_attention"] * 3
    model.config.sliding_window = 24
    model.config.use_sliding_window = True
    model.save_pretrained(folder)
    with pytest.raises(UnsupportedBackendError, match=r"layer 2 .*sliding window of 24.*recent to at least 23"):
        check_servable(economize.apply(_base(folder), KVEvictConfig(recent=16)))
    check_servable(economize.apply(_base(folder), KVEvictConfig(recent=23)))  # the window fits inside
    check_servable(economize.apply(_base(folder), KVEvictConfig(recent=16, layers=(1, 3, 5))))


def test_streaming_groups_run_the_single_pass_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder

    seen = []

    def build(self: Any, common_prefix_len: int, common: Any, fast_build: bool = False) -> Any:
        seen.append(common.rswa_prefix_lens)
        return SimpleNamespace(seq_threshold_3D=8)

    monkeypatch.setattr(TritonAttentionMetadataBuilder, "build", build)
    builder = object.__new__(EconomizeStreamingMetadataBuilder)
    common = SimpleNamespace(is_prefilling=torch.tensor([True, False]), num_reqs=3, rswa_prefix_lens=None)
    monkeypatch.setattr("economize.backends.vllm.kv_evict.backend.replace",
                        lambda obj, **kw: SimpleNamespace(**{**vars(obj), **kw}))
    builder.streaming = None
    assert builder.build(0, common).seq_threshold_3D == 8 and seen[-1] is None  # other groups keep vLLM's choice
    builder.streaming = StreamingSettings(sinks=4, recent=16, layers=frozenset({1}))  # no compaction
    assert builder.build(0, common).seq_threshold_3D == SINGLE_PASS == 0
    assert seen[-1].tolist() == [UNBOUNDED, 4, 4]


def test_streaming_without_sinks_is_refused(tiny_dir: Path) -> None:
    with pytest.raises(UnsupportedBackendError, match="sinks >= 1"):
        check_servable(economize.apply(_base(tiny_dir), KVEvictConfig(method="streaming", sinks=0, recent=16)))


def test_v1_runner_is_refused(tiny_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    with pytest.raises(UnsupportedBackendError, match=r"V2 model runner"):
        check_servable(economize.apply(_base(tiny_dir), STREAMING))


def test_two_eviction_rules_are_refused(tiny_dir: Path) -> None:
    builder = serving.PlanBuilder(_base(tiny_dir), None)
    builder.claim(Role.KV_EVICTION, "kv_evict")
    with pytest.raises(IncompatibleTechniquesError):
        builder.claim(Role.KV_EVICTION, "kv_evict_fused")


def test_support_table_and_plugin_step() -> None:
    (row,) = support_table(["kv_evict"])
    assert row[1:3] == ("kv_eviction", "plugin")
    assert support_table(["kv_evict_fused"])[0][2] == "unsupported"
    assert "kv_evict extension" in [name for name, _ in plugin._steps()]


def test_install_registers_the_custom_backend() -> None:
    from vllm.utils.import_utils import resolve_obj_by_qualname
    from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    from economize.backends.vllm import custom_attention
    from economize.backends.vllm.kv_evict import install
    from economize.backends.vllm.kv_evict.backend import EconomizeStreamingBackend

    previous = _ATTN_OVERRIDES.get(AttentionBackendEnum.CUSTOM)
    try:
        install()
        assert _ATTN_OVERRIDES[AttentionBackendEnum.CUSTOM] == custom_attention.DISPATCH_PATH
        assert AttentionBackendEnum.CUSTOM.get_class().get_name() == "CUSTOM"
        selected = custom_attention.resolve({CONFIG_KEY: {"sinks": 4, "recent": 8}})
        assert selected is not None and selected.path == BACKEND_PATH
        assert resolve_obj_by_qualname(selected.path) is EconomizeStreamingBackend
        spec = StreamingSpec(block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float32, sinks=4, recent=8)
        assert KVCacheSpecRegistry.get_manager_class(spec) is StreamingManager
    finally:
        if previous is None:
            _ATTN_OVERRIDES.pop(AttentionBackendEnum.CUSTOM, None)
        else:
            _ATTN_OVERRIDES[AttentionBackendEnum.CUSTOM] = previous
