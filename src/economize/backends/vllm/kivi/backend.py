"""KIVI as a vLLM attention backend: vLLM's Triton attention over a paged cache that KIVI quantizes in place.

Every cell of the cache (one per page, KV head and token) is widened to :class:`~.layout.CellLayout`: the dense K/V
vLLM writes, followed by the codes, scales, zero points and two flags of KIVI's packed form. vLLM's own cache write
fills the dense part and :func:`~.kernels.clear_flags` marks the new cells as full precision. Before the first covered
layer of a target pass attends, :func:`~.kernels.quantize_committed` quantizes, for every request that decodes, the
keys below ``(c // R) * R`` and the values below ``c - R`` (``c`` = tokens before the pass), exactly where
:class:`economize.techniques.kv_quant.KIVICache` moves its frontier; each layer quantizes its own cache. The dense
entry of a quantized cell is overwritten with the dequantized value, so the cache holds KIVI's numbers whichever
kernel reads it:

* ``kivi`` attends with vLLM's Triton kernel over the dense entries (the fake-quantized cache of HF ``kivi``).
* ``kivi_fused`` attends, on batches where every request decodes and at most 64 queries per request, with
  :func:`~.kernels.paged_kivi_attention`, which reads the packed codes of quantized cells
  (:func:`economize.attention.kivi.kivi_decode_attention` over pages); other batches use the Triton kernel.

Speculative decoding needs no rollback: only committed tokens (below ``c``) are quantized, and a rejected draft's cell
is rewritten (and its flags cleared) by the pass that writes the next token there. The passes of a drafter leave the
cache alone: the builders of economize's jump drafter are marked by
:func:`economize.backends.vllm.custom_attention.mark_draft_builders`, and EAGLE-3's layers lie beyond the target's.
A request that is still prefilling is not quantized (HF quantizes at its first decoding pass too). Under CUDA graphs
the kernels read only persistent buffers (vLLM's block table, query offsets and sequence lengths, and the builder's
prefill flags), and capture runs with sequence lengths 1, where nothing is quantized.

Prefix caching must be off (a cached block would be shared between requests whose quantization frontiers differ);
:class:`economize.backends.vllm.serving.kivi.KIVIVllm` turns it off.

Importing this module imports vLLM.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Any, ClassVar

import torch
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadataBuilder,
)

from ....errors import ConfigError
from ..custom_attention import is_draft_builder
from .layout import SETTINGS, CellLayout, KIVISettings, cell_layout

__all__ = ["MAX_FUSED_QUERIES", "EconomizeKIVIBackend", "EconomizeKIVIImpl", "EconomizeKIVIMetadataBuilder"]

logger = logging.getLogger(__name__)

#: Queries per request up to which ``kivi_fused`` attends with the packed kernel (spec verification fits).
MAX_FUSED_QUERIES = 64
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def layer_index(name: str) -> int | None:
    """The 0-indexed decoder layer of a vLLM attention layer name, or ``None``."""
    match = _LAYER.search(name)
    return int(match.group(1)) if match else None


def _covers(settings: KIVISettings | None, name: str) -> bool:
    index = layer_index(name)
    return settings is not None and index is not None and settings.covers(index)


class EconomizeKIVIMetadataBuilder(TritonAttentionMetadataBuilder):
    """Triton metadata plus what KIVI's kernels read: whether this pass quantizes, and the per-request prefill flags."""

    #: Whether this builder warned already about a pass without per-request prefill flags.
    kivi_warned: bool = False

    def __init__(self, kv_cache_spec: Any, layer_names: list[str], vllm_config: Any, device: torch.device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.kivi = SETTINGS.bind(vllm_config)
        self.kivi_covers = any(_covers(self.kivi, name) for name in layer_names)
        self.kivi_keys = int(vllm_config.model_config.max_model_len)
        self.kivi_prefill = torch.zeros(vllm_config.scheduler_config.max_num_seqs, dtype=torch.int32, device=device)

    def build(self, common_prefix_len: int, common_attn_metadata: Any, fast_build: bool = False) -> Any:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        rows = metadata.seq_lens.shape[0]
        flags = common_attn_metadata.is_prefilling
        quantize = self.kivi_covers and not is_draft_builder(self)
        host = torch.ones(rows, dtype=torch.int32)
        if flags is None:
            if quantize and not self.kivi_warned:
                self.kivi_warned = True
                logger.warning("economize KIVI: this vLLM pass has no per-request prefill flags "
                               "(common_attn_metadata.is_prefilling); its cache is left in full precision")
        else:
            count = min(rows, flags.shape[0])
            host[:count] = flags[:count].to("cpu", torch.int32)
            host[count:] = 0
        self.kivi_prefill[:rows].copy_(host, non_blocking=True)
        metadata.kivi_quantize = quantize
        metadata.kivi_prefill = self.kivi_prefill[:rows]
        metadata.kivi_any_prefill = bool(host.any())
        metadata.kivi_keys = self.kivi_keys if self.decode_cudagraph_enabled else int(metadata.max_seq_len)
        return metadata


class EconomizeKIVIImpl(TritonAttentionImpl):
    """Triton attention over the dense part of KIVI's cells; quantization and the packed kernel on covered layers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.kivi = SETTINGS.current()
        self._dense = 2 * self.head_size
        self._layout: CellLayout | None = None
        self._covered: bool | None = None

    def fused_rope_kvcache_supported(self) -> bool:
        return False  # the cache write must go through do_kv_cache_update (it clears KIVI's flags)

    def _layer_layout(self, layer: Any, kv_cache: torch.Tensor) -> CellLayout | None:
        if self._covered is None:
            self._covered = _covers(self.kivi, str(getattr(layer, "layer_name", "")))
            if self._covered:
                assert self.kivi is not None
                self._layout = cell_layout(self.head_size, kv_cache.dtype, self.kivi)
        return self._layout if self._covered else None

    def do_kv_cache_update(self, layer: Any, key: torch.Tensor, value: torch.Tensor, kv_cache: torch.Tensor,
                           slot_mapping: torch.Tensor) -> None:
        super().do_kv_cache_update(layer, key, value, kv_cache[..., :self._dense], slot_mapping)
        layout = self._layer_layout(layer, kv_cache)
        if layout is not None and self.attn_type not in ("encoder", "encoder_only"):
            from .kernels import clear_flags

            clear_flags(kv_cache, slot_mapping[:key.shape[0]], layout)

    def _packed_kernel_fits(self, metadata: Any, output_scale: Any) -> bool:
        return (self.kivi is not None and self.kivi.fused and not metadata.kivi_any_prefill
                and metadata.max_query_len <= MAX_FUSED_QUERIES and output_scale is None
                and self.alibi_slopes is None and tuple(self.sliding_window) == (-1, -1)
                and metadata.mm_prefix_range_tensor is None and metadata.rswa_window is None
                and self.chunk_lookback in (None, -1) and metadata.causal is True)

    def forward(self, layer: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, kv_cache: torch.Tensor,
                attn_metadata: Any, output: torch.Tensor, output_scale: torch.Tensor | None = None,
                output_block_scale: torch.Tensor | None = None) -> torch.Tensor:
        dense = kv_cache[..., :self._dense]
        if attn_metadata is None or not hasattr(attn_metadata, "kivi_quantize"):
            return super().forward(layer, query, key, value, dense, attn_metadata, output, output_scale,
                                   output_block_scale)
        layout = self._layer_layout(layer, kv_cache)
        if layout is None:
            return super().forward(layer, query, key, value, dense, attn_metadata, output, output_scale,
                                   output_block_scale)
        from .kernels import paged_kivi_attention, quantize_committed

        assert self.kivi is not None
        if attn_metadata.kivi_quantize:
            quantize_committed(kv_cache, layout, self.kivi.residual_length, attn_metadata.block_table,
                               attn_metadata.query_start_loc, attn_metadata.seq_lens, attn_metadata.kivi_prefill)
        if not self._packed_kernel_fits(attn_metadata, output_scale):
            return super().forward(layer, query, key, value, dense, attn_metadata, output, output_scale,
                                   output_block_scale)
        tokens = attn_metadata.num_actual_tokens
        out = output.view(output.shape[0], self.num_heads, self.head_size)
        paged_kivi_attention(
            query[:tokens], kv_cache, layout, attn_metadata.block_table, attn_metadata.query_start_loc,
            attn_metadata.seq_lens, attn_metadata.max_query_len, attn_metadata.kivi_keys, self.scale, out,
            softcap=self.logits_soft_cap or None, sinks=self.sinks,
        )
        return output


class EconomizeKIVIBackend(TritonAttentionBackend):
    """vLLM's Triton attention backend over KIVI's quantized paged cache (registered as ``CUSTOM``)."""

    accept_output_buffer: ClassVar[bool] = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[EconomizeKIVIImpl]:
        return EconomizeKIVIImpl

    @staticmethod
    def get_builder_cls() -> type[EconomizeKIVIMetadataBuilder]:
        return EconomizeKIVIMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[Any]:
        from vllm.v1.attention.backend import MultipleOf

        settings = SETTINGS.current()
        return [MultipleOf(settings.block_multiple if settings is not None else 16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        settings = SETTINGS.current()
        return block_size is None or block_size % (settings.block_multiple if settings is not None else 16) == 0

    @classmethod
    def customize_spec(cls, spec: Any) -> Any:
        spec = super().customize_spec(spec)
        settings = SETTINGS.current()
        if settings is None or getattr(spec, "state_content_bytes", None) is not None or "MLA" in type(spec).__name__:
            return spec
        if spec.head_size_v != spec.head_size:
            raise ConfigError(f"economize's vLLM KIVI backend needs equal key and value head sizes, not "
                              f"{spec.head_size} / {spec.head_size_v}")
        return replace(spec, state_content_bytes=cell_layout(spec.head_size, spec.dtype, settings).size)
