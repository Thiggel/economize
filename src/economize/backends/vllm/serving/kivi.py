"""``kivi`` and ``kivi_fused``: KIVI's quantized KV cache on vLLM, through economize's ``CUSTOM`` attention backend.

=================================  ==============================================================================
economize                          vLLM 0.30 with economize's plugin
=================================  ==============================================================================
``kivi``                           attention backend ``CUSTOM`` (:mod:`economize.backends.vllm.kivi`): vLLM's Triton
                                   attention over pages whose committed tokens KIVI quantizes in place (keys per
                                   channel over groups of tokens, values per token over groups of channels, the most
                                   recent ``residual_length`` tokens in full precision); it reads the dequantized
                                   entries, the numbers of economize's ``kivi``
``kivi_fused``                     the same cache; decoding batches attend with economize's paged KIVI kernel, which
                                   reads the packed codes (economize's ``kivi_fused`` kernel over vLLM's pages)
``bits`` / ``group_size`` /        ``additional_config["economize_kivi"]``; the page size is a multiple of
``residual_length``                ``lcm(16, group_size)``, so a key group never spans two pages
``layers=(...)``                   the quantization on those layers only
=================================  ==============================================================================

Each cell of the paged cache keeps a token's dense K/V beside its packed codes, scales and zero points
(:class:`economize.backends.vllm.kivi.layout.CellLayout`), so the cache takes more memory than vLLM's dense cache.
KIVI on vLLM reproduces KIVI's numbers and, with ``kivi_fused``, reads fewer bytes per decoding step; it does not let
more requests fit in the KV cache.
"""

from __future__ import annotations

import math
from typing import Any

from torch import nn

from ....core.metadata import Saving, Simulation
from ...exporters import UnsupportedBackendError
from .base import PlanBuilder, Role, Support, VllmTechnique, register_vllm_technique

__all__ = ["CONFIG_KEY", "KIVI_ATTENTION", "FusedKIVIVllm", "KIVIVllm", "kivi_settings"]

#: The ``additional_config`` key the backend reads (:data:`economize.backends.vllm.kivi.CONFIG_KEY`).
CONFIG_KEY = "economize_kivi"
#: vLLM's name of the attention backend economize registers for KIVI.
KIVI_ATTENTION = "CUSTOM"
#: vLLM's Triton kernels need pages of a multiple of 16 tokens and head sizes of at least 32.
_TRITON_BLOCK = 16
_MIN_HEAD = 32
#: Techniques that cannot share an engine with KIVI's backend, with the reason.
_REFUSED = {
    "kv_evict": "StreamingLLM runs its own CUSTOM attention backend, and vLLM runs one attention backend per engine",
    "dflash": "DFlash's drafter attends bidirectionally over its block, which the KIVI backend has not been "
              "verified with",
}


def kivi_settings(technique: nn.Module, num_layers: int) -> dict[str, Any]:
    """The ``additional_config`` entry of a ``kivi`` / ``kivi_fused`` technique on a target of ``num_layers``."""
    config: Any = technique.config
    return {"bits": int(config.bits), "group_size": int(config.group_size),
            "residual_length": int(config.residual_length), "layers": sorted(int(layer) for layer in config.layers),
            "num_layers": int(num_layers), "fused": config.technique == "kivi_fused"}


@register_vllm_technique
class KIVIVllm(VllmTechnique):
    """``kivi``: KIVI's fake-quantized cache, attended by vLLM's Triton kernel."""

    technique = "kivi"
    role = Role.KV_CACHE
    support = Support.PLUGIN
    reason = ("economize's CUSTOM attention backend: vLLM's Triton attention over pages whose committed tokens KIVI "
              "quantizes in place (per-channel keys, per-token values, a full-precision residual); the cache keeps "
              "the dense entries beside the codes, so it saves no KV memory")
    simulation = Simulation(Saving.KV_MEMORY, "the vLLM cache keeps the dense entries beside the codes")

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        for name, why in _REFUSED.items():
            if name in builder.techniques:
                raise UnsupportedBackendError(f"{self.technique} and {name} are not served together on vLLM: {why}")
        config: Any = technique.config
        group = int(config.group_size)
        if group & (group - 1):
            raise UnsupportedBackendError(
                f"{self.technique} on vLLM needs a power-of-two group_size (the page size is a multiple of "
                f"lcm(16, group_size), and vLLM's page sizes are powers of two), got {group}")
        adapter = builder.model.adapter
        covered = config.layers or range(1, int(adapter.num_layers) + 1)
        for index in covered:
            spec = adapter.attention_spec(index)
            if spec.kind != "full":
                raise UnsupportedBackendError(
                    f"layer {index} is a {spec.kind!r} layer; KIVI on vLLM quantizes full-attention layers only "
                    "(vLLM frees the pages a sliding window has left, and the quantization walks back over them). "
                    "List the full-attention layers in layers=(...)")
            head = int(adapter.head_geometry(index).head_dim)
            if head < _MIN_HEAD or head & (head - 1):
                raise UnsupportedBackendError(
                    f"layer {index} has head size {head}; KIVI on vLLM needs a power of two of at least {_MIN_HEAD} "
                    "(vLLM's Triton attention and economize's packed kernel)")
            if head % (8 // int(config.bits)):
                raise UnsupportedBackendError(f"layer {index}: head size {head} is not a multiple of the "
                                              f"{8 // int(config.bits)} codes a byte holds")

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        settings = kivi_settings(technique, int(builder.model.adapter.num_layers))
        builder.set_additional(CONFIG_KEY, settings, owner=self.technique)
        builder.require_attention(KIVI_ATTENTION, owner=self.technique)
        builder.set_engine("enable_prefix_caching", False, owner=self.technique)
        block = math.lcm(_TRITON_BLOCK, settings["group_size"])
        if block > _TRITON_BLOCK:
            builder.set_engine("block_size", block, owner=self.technique)
        builder.note(self.technique, f"KIVI on economize's {KIVI_ATTENTION} attention backend: before a target pass "
                                     "attends, the keys of complete blocks of residual_length "
                                     f"({settings['residual_length']}) committed tokens and the values older than "
                                     f"residual_length are quantized to {settings['bits']} bits in place; prefix "
                                     "caching is off")
        builder.note(self.technique, "each cache cell keeps the dense K/V beside the codes, scales and zero points, "
                                     "so vLLM's KV cache holds fewer tokens than with a dense cache")
        if settings["fused"]:
            builder.note(self.technique, "batches in which every request decodes (at most 64 queries each, so "
                                         "verification passes too) attend with economize's paged KIVI kernel over "
                                         "the packed codes; prefill and mixed batches use vLLM's Triton kernel over "
                                         "the dequantized entries")
        if Role.DRAFTER in builder.roles:
            builder.note(self.technique, "draft passes leave the cache as it is: only the target's passes quantize, "
                                         "and only committed tokens, so a rejected draft needs no rollback; a "
                                         "separate draft model's own layers stay in full precision")
        builder.note(self.technique, "a request that vLLM preempts and recomputes is quantized again from its "
                                     "recomputed K/V; a scale or zero point beyond the fp16 range keeps that group in "
                                     "its fake-quantized dense form, as economize's runtime does")


@register_vllm_technique
class FusedKIVIVllm(KIVIVllm):
    """``kivi_fused``: KIVI's packed cache, attended by economize's paged KIVI kernel on decoding batches."""

    technique = "kivi_fused"
    reason = ("economize's CUSTOM attention backend with its paged KIVI kernel, which reads the packed 2/4/8-bit "
              "codes on decoding and verification batches; the cache keeps the dense entries beside the codes, so "
              "it saves no KV memory")
