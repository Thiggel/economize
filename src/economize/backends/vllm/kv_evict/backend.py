"""StreamingLLM as a vLLM attention backend: vLLM's Triton attention with a sink-plus-window mask.

vLLM's Triton kernel has a mask for reference sliding-window attention (R-SWA): a query at position ``q`` of a
sequence reads key ``j`` when ``j < prefix_len`` or ``q - j < window``. :class:`EconomizeStreamingBackend` is the
Triton backend whose metadata builder sets ``prefix_len = sinks`` for every decoding request and
``window = recent + 1``, so a decoding query reads the first ``sinks`` tokens and the last ``recent`` tokens before
it, as :class:`economize.techniques.kv_evict.StreamingEviction` keeps them. A request that is still prefilling gets an
unbounded prefix, so its queries read the whole prompt (StreamingLLM evicts at decoding steps only). The KV cache
spec of every streaming layer becomes :class:`~economize.backends.vllm.kv_evict.spec.StreamingSpec`, whose manager
frees the blocks no later query reads.

The engine finds the window in ``additional_config["economize_kv_evict"]`` (written by
:func:`economize.backends.vllm.serving.to_vllm`) and selects this class as vLLM's ``CUSTOM`` attention backend. The
per-request prefill flag comes from vLLM's V2 model runner; the V1 runner is refused.

Streaming groups always run vLLM's single-pass (2D) Triton kernel. vLLM 0.30's split-KV (3D) decode kernel is wrong
under a sink-plus-window mask: a KV segment whose keys are all masked stores the running maximum 0 (``softmax_step``
replaces a ``-inf`` maximum by 0), and ``reduce_segments`` then takes the maximum over segments including that 0. When
the true row maximum of a head is far below 0 (about -480 on one head of Qwen2.5-0.5B), ``exp(m - 0)`` underflows and
the output of that head is garbage. The 2D kernel visits the sink tiles first, so its running maximum is finite before
any masked tile; this needs ``sinks >= 1``, which :class:`economize.backends.vllm.serving.kv_evict.KVEvictVllm` checks.
The price is slower long-context decoding at small batch sizes, where vLLM would otherwise split the keys.

vLLM's kernel skips no tile under an R-SWA mask, so a decoding query would still visit every key of its sequence.
The builder therefore also compacts the block table of each decoding request
(:func:`dropped_pages`, :func:`compact_block_table`): it keeps the pages that hold the sinks, removes the ``d`` whole
pages between them and the first page of the window, and shortens ``seq_lens`` by ``d`` pages. The kernel then visits
the sink pages and the window only, and reads the same keys under the same mask: a kept key and its query move by the
same distance, the non-sink keys of the sink pages end up at least ``window`` positions before every query, and key 0
stays the first key of the first tile. Rotary embeddings are applied before the cache write, which goes through
``slot_mapping`` and never through this table. Prefilling requests, multimodal batches and layer subsets (whose
other layers share the table) keep the full table.

Importing this module imports vLLM.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar

import torch
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

from ....errors import ConfigError, UnsupportedModelError
from ..engine_settings import EngineSettings
from .spec import StreamingSpec

__all__ = ["CONFIG_KEY", "SETTINGS", "SINGLE_PASS", "EconomizeStreamingBackend", "StreamingMetadata",
           "StreamingSettings", "compact_block_table", "dropped_pages", "prefix_lens", "settings_of"]

#: The ``additional_config`` key of the engine that holds the window.
CONFIG_KEY = "economize_kv_evict"
#: The prefix length of a prefilling request: every key is visible.
UNBOUNDED = torch.iinfo(torch.int32).max
#: ``seq_threshold_3D`` of a streaming group: vLLM launches its 3D (split-KV) kernel only for batches of at most this
#: many sequences, so 0 keeps every launch on the single-pass 2D kernel (see the module docstring).
SINGLE_PASS = 0
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class StreamingSettings:
    """The StreamingLLM window of an engine.

    Attributes:
        sinks: First tokens every query reads.
        recent: Tokens before each query it reads in addition to itself.
        layers: 1-indexed decoder layers that stream; empty means every full-attention layer (and only then are
            blocks freed).
        retain: Tokens kept below the window for draft models that lag behind the committed frontier.
    """

    sinks: int
    recent: int
    layers: frozenset[int] = frozenset()
    retain: int = 0

    @property
    def window(self) -> int:
        """vLLM's R-SWA window: the query and the ``recent`` tokens before it."""
        return self.recent + 1


def _parse(entry: Mapping[str, Any], vllm_config: Any) -> StreamingSettings:
    speculative = getattr(vllm_config, "speculative_config", None)
    retain = max(0, int(getattr(speculative, "num_speculative_tokens", 0) or 0) - 1) if speculative else 0
    return StreamingSettings(sinks=int(entry["sinks"]), recent=int(entry["recent"]),
                             layers=frozenset(int(layer) for layer in entry.get("layers") or ()), retain=retain)


#: The streaming settings of the engine (see :class:`~economize.backends.vllm.engine_settings.EngineSettings`).
SETTINGS: EngineSettings[StreamingSettings] = EngineSettings(CONFIG_KEY, _parse)
#: The streaming settings ``vllm_config`` asks for, or ``None``.
settings_of = SETTINGS.of


def prefix_lens(is_prefilling: torch.Tensor | None, num_reqs: int, sinks: int) -> torch.Tensor:
    """The R-SWA prefix length of each of ``num_reqs`` requests (CPU, int32): ``sinks`` for a decoding request,
    :data:`UNBOUNDED` for a prefilling one (rows beyond ``is_prefilling`` are padding and decode)."""
    if is_prefilling is None:
        raise RuntimeError("economize's StreamingLLM backend needs the per-request prefill flag of vLLM's V2 model "
                           "runner (common_attn_metadata.is_prefilling); this attention metadata has none")
    prefix = torch.full((num_reqs,), sinks, dtype=torch.int32)
    count = min(num_reqs, is_prefilling.shape[0])
    prefix[:count].masked_fill_(is_prefilling[:count].to("cpu", torch.bool), UNBOUNDED)
    return prefix


def dropped_pages(query_start_loc: torch.Tensor, seq_lens: torch.Tensor, decoding: torch.Tensor, sinks: int,
                  recent: int, page: int) -> torch.Tensor:
    """The number of whole pages between the sink pages and the window of each request (int32, zero unless
    ``decoding``).

    A request with ``n`` queries ending at ``seq_lens`` has its first query at ``q0 = seq_lens - n``, which reads
    keys from ``q0 - recent`` on; later queries read later keys. The page of key ``q0 - recent`` is the first page of
    the window, and the ``ceil(sinks / page)`` first pages hold the sinks.
    """
    rows = seq_lens.shape[0]
    queries = query_start_loc[1:rows + 1] - query_start_loc[:rows]
    first = torch.div((seq_lens - queries - recent).clamp(min=0), page, rounding_mode="floor")
    drop = (first - -(-sinks // page)).clamp(min=0)
    return torch.where(decoding, drop, torch.zeros_like(drop)).to(torch.int32)


def compact_block_table(table: torch.Tensor, drop: torch.Tensor, sinks: int, page: int, out: torch.Tensor) -> None:
    """Write into ``out`` the block table of each request without its ``drop`` pages after the sink pages.

    Entry ``i`` of a row is the page ``i`` of ``table`` for the sink pages and the page ``i + drop`` after them; the
    entries past the end of the original row repeat its last page and are never read (``seq_lens`` ends earlier).
    """
    width = table.shape[1]
    positions = torch.arange(width, device=table.device)
    shift = (positions >= -(-sinks // page)).to(drop.dtype) * drop[:, None]
    index = (positions + shift).clamp(max=width - 1).to(torch.int64)
    out.copy_(torch.gather(table, 1, index))


@dataclass
class StreamingMetadata(TritonAttentionMetadata):
    """Triton metadata of a streaming group with compacted block tables.

    Attributes:
        full_seq_lens: The sequence lengths before compaction (vLLM's buffer, which the draft loop advances in place).
        dropped_tokens: The tokens removed from each request's table (``page`` times its dropped pages).
    """

    full_seq_lens: torch.Tensor | None = None
    dropped_tokens: torch.Tensor | None = None


def _streams(spec: Any) -> bool:
    return getattr(spec, "sliding_window", None) is None and not getattr(spec, "non_causal", False) \
        and getattr(spec, "attention_chunk_size", None) is None


def streaming_spec(spec: Any, settings: StreamingSettings | None) -> Any:
    """``spec`` as a :class:`StreamingSpec` when its layer streams and frees blocks, else ``spec``."""
    if settings is None or settings.layers or type(spec) is not FullAttentionSpec or not _streams(spec):
        return spec
    kwargs = {f.name: getattr(spec, f.name) for f in fields(FullAttentionSpec) if f.init}
    return StreamingSpec(**kwargs, sinks=settings.sinks, recent=settings.recent, retain=settings.retain)


class EconomizeStreamingMetadataBuilder(TritonAttentionMetadataBuilder):
    """Triton metadata with the sink-plus-window mask on every decoding request of a full-attention group."""

    def __init__(self, kv_cache_spec: Any, layer_names: list[str], vllm_config: Any, device: torch.device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        settings = SETTINGS.bind(vllm_config)
        self.streaming = settings if settings is not None and _streams(kv_cache_spec) else None
        if self.streaming is None:
            return
        if self.rswa_window is not None:
            raise UnsupportedModelError("economize's StreamingLLM backend cannot serve a model that uses vLLM's "
                                        "R-SWA already")
        if not getattr(vllm_config, "use_v2_model_runner", True):
            raise ConfigError("economize's StreamingLLM backend needs vLLM's V2 model runner: unset "
                             "VLLM_USE_V2_MODEL_RUNNER=0")
        self.rswa_window = self.streaming.window
        self._max_rows = int(vllm_config.scheduler_config.max_num_seqs)
        self.persistent_rswa_prefix_lens = torch.empty(self._max_rows, dtype=torch.int32, device=device)
        # Persistent buffers of the compacted metadata (CUDA graphs keep their addresses), allocated at the first build.
        self._table: torch.Tensor | None = None
        self._seq_lens: torch.Tensor | None = None
        self._dropped: torch.Tensor | None = None

    @property
    def compacts(self) -> bool:
        """Whether this group's block tables are compacted (every layer of the group streams)."""
        return self.streaming is not None and not self.streaming.layers

    def _page(self) -> int:
        return int(getattr(self, "kernel_block_size", None) or self.block_size)

    def _buffers(self, table: torch.Tensor, rows: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        need_rows = max(self._max_rows, table.shape[0], rows)
        if self._table is None or self._table.shape[0] < need_rows or self._table.shape[1] < table.shape[1]:
            self._table = torch.zeros(need_rows, table.shape[1], dtype=table.dtype, device=table.device)
            self._seq_lens = torch.zeros(need_rows, dtype=torch.int32, device=table.device)
            self._dropped = torch.zeros(need_rows, dtype=torch.int32, device=table.device)
        assert self._seq_lens is not None and self._dropped is not None
        return self._table, self._seq_lens, self._dropped

    def build(self, common_prefix_len: int, common_attn_metadata: Any, fast_build: bool = False) -> Any:
        if self.streaming is None:
            return super().build(common_prefix_len, common_attn_metadata, fast_build)
        prefix = prefix_lens(common_attn_metadata.is_prefilling, common_attn_metadata.num_reqs, self.streaming.sinks)
        metadata = super().build(common_prefix_len, replace(common_attn_metadata, rswa_prefix_lens=prefix), fast_build)
        metadata.seq_threshold_3D = SINGLE_PASS
        if not self.compacts:
            return metadata
        return self._compact(metadata, prefix, common_prefix_len)

    def _compact(self, metadata: Any, prefix: torch.Tensor, common_prefix_len: int) -> StreamingMetadata:
        """``metadata`` with each decoding request's block table compacted (through persistent buffers always, so
        that captured CUDA graphs read the current step's tables)."""
        assert self.streaming is not None
        table, seq_lens = metadata.block_table, metadata.seq_lens
        rows, table_rows = seq_lens.shape[0], table.shape[0]
        table_out, seq_out, dropped_out = self._buffers(table, rows)
        known = min(rows, table_rows)  # rows with a length and a table (vLLM passes as many of each)
        decoding = torch.zeros(table_rows, dtype=torch.bool)
        if common_prefix_len == 0 and metadata.mm_prefix_range is None:
            flagged = min(known, prefix.shape[0])
            decoding[:flagged] = prefix[:flagged] != UNBOUNDED
        decoding = decoding.to(table.device, non_blocking=True)
        lens = torch.zeros(table_rows, dtype=seq_lens.dtype, device=table.device)
        lens[:known] = seq_lens[:known]
        starts = torch.zeros(table_rows + 1, dtype=metadata.query_start_loc.dtype, device=table.device)
        starts[:known + 1] = metadata.query_start_loc[:known + 1]
        starts[known + 1:] = starts[known]
        page = self._page()
        drop = dropped_pages(starts, lens, decoding, self.streaming.sinks, self.streaming.recent, page)
        compact_block_table(table, drop, self.streaming.sinks, page, table_out[:table_rows, :table.shape[1]])
        dropped = dropped_out[:rows]
        dropped.zero_()
        dropped[:known] = drop[:known] * page
        torch.sub(seq_lens, dropped, out=seq_out[:rows])
        values = {f.name: getattr(metadata, f.name) for f in fields(TritonAttentionMetadata)}
        values.update(seq_lens=seq_out[:rows], block_table=table_out[:table_rows, :table.shape[1]])
        return StreamingMetadata(**values, full_seq_lens=seq_lens, dropped_tokens=dropped)

    def update_draft_decode_metadata(self, metadata: Any) -> None:
        """Follow vLLM's in-place advance of the draft positions: the pages dropped at the first draft step stay
        before the window of every later step, so only the shortened lengths change."""
        if isinstance(metadata, StreamingMetadata) and metadata.full_seq_lens is not None:
            assert metadata.dropped_tokens is not None
            torch.sub(metadata.full_seq_lens, metadata.dropped_tokens, out=metadata.seq_lens)

    def build_for_cudagraph_capture(self, common_attn_metadata: Any) -> Any:
        metadata = super().build_for_cudagraph_capture(common_attn_metadata)
        if isinstance(metadata, StreamingMetadata) and metadata.full_seq_lens is not None:
            assert metadata.dropped_tokens is not None
            metadata.full_seq_lens.fill_(1)  # vLLM captures with unit lengths (TritonAttentionMetadataBuilder)
            metadata.dropped_tokens.zero_()
        return metadata


class EconomizeStreamingImpl(TritonAttentionImpl):
    """Triton attention that drops the streaming mask on the layers outside a layer subset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        settings = SETTINGS.current()
        self._subset = settings.layers if settings is not None else frozenset()
        self._layer_streams: bool | None = None
        if settings is not None and not settings.layers and getattr(self, "alibi_slopes", None) is not None:
            raise UnsupportedModelError(
                "economize's StreamingLLM backend cannot serve ALiBi attention on every layer: it compacts the block "
                "table of each decoding request, which moves the window towards the sinks, and ALiBi biases a score "
                "by the distance between query and key. List the streaming layers in layers=(...) to keep the full "
                "table, or serve the model with economize's runtime")

    def _streams_at(self, layer: Any) -> bool:
        if not self._subset:
            return True
        match = _LAYER.search(str(getattr(layer, "layer_name", "")))
        return match is not None and int(match.group(1)) + 1 in self._subset

    def forward(self, layer: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, kv_cache: Any,
                attn_metadata: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self._subset and attn_metadata is not None and attn_metadata.rswa_prefix_lens is not None:
            if self._layer_streams is None:
                self._layer_streams = self._streams_at(layer)
            if not self._layer_streams:
                attn_metadata = replace(attn_metadata, rswa_prefix_lens=None, rswa_window=None)
        return super().forward(layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs)


class EconomizeStreamingBackend(TritonAttentionBackend):
    """vLLM's Triton attention backend with StreamingLLM's sink-plus-window mask (registered as ``CUSTOM``)."""

    accept_output_buffer: ClassVar[bool] = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[EconomizeStreamingImpl]:
        return EconomizeStreamingImpl

    @staticmethod
    def get_builder_cls() -> type[EconomizeStreamingMetadataBuilder]:
        return EconomizeStreamingMetadataBuilder

    @classmethod
    def customize_spec(cls, spec: Any) -> Any:
        return streaming_spec(super().customize_spec(spec), SETTINGS.current())
