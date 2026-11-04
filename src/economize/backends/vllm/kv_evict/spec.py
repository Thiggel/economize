"""The KV cache spec and manager of StreamingLLM on vLLM: full-attention pages whose middle blocks are freed.

A layer that streams keeps its first ``sinks`` tokens and the last ``recent`` tokens before each query (plus the
query). Once a request has finished its prefill, :class:`StreamingManager` frees every block that lies entirely
between the sink blocks and that window, so a long generation holds ``O(sinks + recent)`` pages per request. The
freed blocks become vLLM's null block, which the attention mask of
:class:`economize.backends.vllm.kv_evict.backend.EconomizeStreamingBackend` never reads.

Importing this module imports vLLM and registers the spec with vLLM's ``KVCacheSpecRegistry`` (grouped with full
attention, so a model whose full-attention layers all stream keeps one KV cache group).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from vllm.utils.math_utils import cdiv
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from ....errors import ConfigError

__all__ = ["StreamingManager", "StreamingSpec", "freed_block_range"]


@dataclass(frozen=True, kw_only=True)
class StreamingSpec(FullAttentionSpec):
    """A full-attention spec whose layers stream.

    Attributes:
        sinks: Tokens at the start of the sequence every query keeps (attention sinks).
        recent: Tokens before each query it keeps in addition (the query itself is always kept).
        retain: Tokens below the window that are kept although no query reads them, for draft models that write
            the KV of positions behind the committed frontier (``num_speculative_tokens - 1`` under speculative
            decoding, else 0).
    """

    sinks: int
    recent: int
    retain: int = 0

    @classmethod
    def merge(cls, specs: list[Any]) -> StreamingSpec:
        if not all(isinstance(spec, StreamingSpec) for spec in specs):
            raise ConfigError("all attention layers of a KV cache group must be StreamingSpec when one is")
        windows = {(spec.sinks, spec.recent, spec.retain) for spec in specs}
        if len(windows) != 1:
            raise ConfigError(f"all streaming layers of a KV cache group must share one window, got {windows}")
        sinks, recent, retain = windows.pop()
        base = FullAttentionSpec.merge(specs)
        kwargs = {f.name: getattr(base, f.name) for f in fields(FullAttentionSpec) if f.init}
        return cls(**kwargs, sinks=sinks, recent=recent, retain=retain)


def freed_block_range(sinks: int, recent: int, retain: int, block_size: int, processed_tokens: int,
                      num_prompt_tokens: int | None) -> tuple[int, int]:
    """The blocks ``[first, last)`` that no future query of a streaming request reads.

    ``processed_tokens`` is the committed computed-token count (vLLM's freeing frontier): every query of a later
    step sits at a position ``>= processed_tokens`` and reads ``[0, sinks)`` and ``[position - recent, position]``.
    Nothing is freed while the prompt is still being prefilled (its queries read the whole prompt) or when the
    prompt length is unknown.
    """
    if num_prompt_tokens is None or processed_tokens < num_prompt_tokens:
        return 0, 0
    first = cdiv(sinks, block_size)
    last = max(0, processed_tokens - recent - retain) // block_size
    return first, max(first, last)


class StreamingManager(FullAttentionManager):
    """Full-attention manager that frees the blocks between the sinks and the window of a decoding request."""

    def __init__(self, kv_cache_spec: StreamingSpec, **kwargs: Any) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.sinks = kv_cache_spec.sinks
        self.recent = kv_cache_spec.recent
        self.retain = kv_cache_spec.retain

    def remove_skipped_blocks(self, request_id: str, processed_computed_tokens: int,
                              num_prompt_tokens: int | None = None) -> None:
        first, last = freed_block_range(self.sinks, self.recent, self.retain, self.block_size,
                                        processed_computed_tokens, num_prompt_tokens)
        if last > first:
            self._remove_blocks_in_range(request_id, first, last)


def _register() -> None:
    from vllm.config import get_current_vllm_config_or_none

    register_all_kvcache_specs(get_current_vllm_config_or_none())  # the built-in specs first: the registry fills
    # itself lazily only while it is empty
    KVCacheSpecRegistry.register(StreamingSpec, StreamingManager, uniform_type_base_spec=FullAttentionSpec)


_register()
