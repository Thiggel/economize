"""Attention kernels for decoding (see :mod:`.decode`) and their registration with ``transformers``.

Usage::

    import economize.attention

    economize.attention.install()  # registers "economize" with transformers
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="economize")

Every economize technique that works under ``sdpa`` works under ``"economize"``: the runtime treats a registered
kernel as a member of its family (:func:`.registry.attention_family`). New kernels are added by registering an
:class:`.registry.AttentionKernel` in :data:`.registry.ATTENTION_KERNELS` (or through the
``economize.attention_kernels`` entry-point group). A compressed KV cache can hand the kernel its packed bytes
next to the dense buffers (:mod:`.packed`); ``"economize"`` reads KIVI codes (:mod:`.kivi`) and the compacted
prompt entries of a one-shot eviction (:mod:`.gathered`) this way, and the query-selected pages of
Quest (:mod:`.quest`). A sparse-prefill pattern attached to an attention module (:mod:`.prefill`) makes
``"economize"`` attend long fresh prompts over MInference's indexed keys only (:mod:`.minference`).
:mod:`.sdpa_guard` keeps ``sdpa`` exact where PyTorch 2.13's masked kernel errs (one K/V head, head dims above
128); adapters of ``sdpa`` models install it. A pass over whole sequences can hand the attention a band-shaped
:class:`.banded.BandMask` instead of a dense mask (:mod:`.banded`), which ``sdpa`` and ``"economize"`` attend in
blocks over the keys each block sees.
"""

from __future__ import annotations

from . import sdpa_guard
from .banded import BandMask, StructuredMask, accepts_structured_masks
from .decode import decode_attention, reference_attention, triton_available
from .gathered import GatheredKV, gathered_decode_attention
from .interface import (
    NAME,
    AttentionObserver,
    economize_attention_forward,
    install,
    kernel_calls,
    kernel_usage,
    usage_summary,
)
from .kivi import KIVIPacked, kivi_decode_attention
from .minference import MInferencePrefill, SparseIndex, sparse_prefill_attention
from .packed import PackedKV, attach_packed, packed_of
from .prefill import SparsePrefill, attach_prefill, detach_prefill, prefill_of
from .quest import QuestPages, quest_decode_attention
from .registry import (
    ATTENTION_KERNELS,
    AttentionKernel,
    accepts_fp8_kv,
    accepts_packed_kv,
    accepts_sparse_prefill,
    attention_family,
)

__all__ = [
    "ATTENTION_KERNELS",
    "NAME",
    "AttentionKernel",
    "AttentionObserver",
    "BandMask",
    "GatheredKV",
    "KIVIPacked",
    "MInferencePrefill",
    "PackedKV",
    "QuestPages",
    "SparseIndex",
    "SparsePrefill",
    "StructuredMask",
    "accepts_fp8_kv",
    "accepts_packed_kv",
    "accepts_sparse_prefill",
    "accepts_structured_masks",
    "attach_packed",
    "attach_prefill",
    "attention_family",
    "decode_attention",
    "detach_prefill",
    "economize_attention_forward",
    "gathered_decode_attention",
    "install",
    "kernel_calls",
    "kernel_usage",
    "kivi_decode_attention",
    "packed_of",
    "prefill_of",
    "quest_decode_attention",
    "reference_attention",
    "sdpa_guard",
    "sparse_prefill_attention",
    "triton_available",
    "usage_summary",
]
