"""Registry of attention kernels economize can run a model's layers with.

An entry names a ``transformers`` attention implementation (the value of ``config._attn_implementation``) that
economize may load a model with, and the built-in implementation whose mask encoding and semantics it shares (its
*family*). The runtime encodes masks, validates adapters and decides about score taps by the family, so a kernel
registered here works with every technique that works under its family::

    from economize.attention import ATTENTION_KERNELS, AttentionKernel

    ATTENTION_KERNELS.register("my_kernel", AttentionKernel(family="sdpa", forward=my_forward))

This module imports nothing heavy: :mod:`economize.arch` consults it on every adapter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..registry import Registry

__all__ = [
    "ATTENTION_KERNELS",
    "AttentionKernel",
    "accepts_fp8_kv",
    "accepts_packed_kv",
    "accepts_sparse_prefill",
    "attention_family",
]


@dataclass(frozen=True)
class AttentionKernel:
    """One attention implementation that economize can run a model with.

    Attributes:
        family: Built-in implementation (``"sdpa"`` or ``"eager"``) whose mask format and semantics the kernel
            shares.
        forward: The ``transformers`` attention function (the signature of
            ``transformers.integrations.sdpa_attention.sdpa_attention_forward``).
        mask_function: Mask builder for ``transformers.masking_utils.AttentionMaskInterface``; ``None`` reuses the
            family's.
        description: One line for listings.
        fp8_kv: Whether ``forward`` takes K/V in ``float8_e4m3fn`` / ``float8_e5m2`` with queries in a wider dtype
            (an fp8 KV cache can then hand its buffers over without upcasting them).
        packed_kv: Packed K/V formats ``forward`` reads from a :class:`~.packed.PackedKV` attached to the key
            tensor (see :mod:`.packed`), such as ``"kivi"``. A kernel that ignores the attachment still computes
            the same result from the dense buffers, only slower.
        sparse_prefill: Whether ``forward`` honours a :class:`~.prefill.SparsePrefill` attached to the attention
            module (MInference's patterns): a kernel that ignores it attends prompts densely, so a sparse-prefill
            technique refuses it.
        structured_masks: Whether ``forward`` takes a :class:`~.banded.StructuredMask` (such as a
            :class:`~.banded.BandMask`) in place of the mask tensor and lets it attend (see
            :func:`~.banded.accepts_structured_masks`).
    """

    family: str
    forward: Callable[..., tuple[Any, Any]]
    mask_function: Callable[..., Any] | None = None
    description: str = ""
    fp8_kv: bool = False
    packed_kv: frozenset[str] = frozenset()
    sparse_prefill: bool = False
    structured_masks: bool = False


ATTENTION_KERNELS: Registry[AttentionKernel] = Registry("attention kernel", "economize.attention_kernels")


def accepts_fp8_kv(implementation: str) -> bool:
    """Whether the registered kernel ``implementation`` takes fp8 K/V directly (``False`` for built-ins)."""
    return implementation in ATTENTION_KERNELS and ATTENTION_KERNELS.get(implementation).fp8_kv


def accepts_packed_kv(implementation: str, fmt: str) -> bool:
    """Whether the registered kernel ``implementation`` reads packed K/V of format ``fmt`` (``False`` for built-ins)."""
    return implementation in ATTENTION_KERNELS and fmt in ATTENTION_KERNELS.get(implementation).packed_kv


def accepts_sparse_prefill(implementation: str) -> bool:
    """Whether the registered kernel ``implementation`` attends prompts through an attached sparse prefill."""
    return implementation in ATTENTION_KERNELS and ATTENTION_KERNELS.get(implementation).sparse_prefill


def attention_family(implementation: str) -> str:
    """The built-in implementation whose masks and semantics ``implementation`` shares (itself if unregistered)."""
    if implementation in ATTENTION_KERNELS:
        return ATTENTION_KERNELS.get(implementation).family
    return implementation
