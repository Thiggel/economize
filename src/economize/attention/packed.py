"""Packed K/V handed to the attention kernel next to the dense cache buffers.

A compressed KV cache (KIVI's 2/4-bit codes, say) keeps its dense buffers, so that every attention function, every
fallback and every technique that reads the cache sees ordinary tensors, and additionally attaches a
:class:`PackedKV` to the key tensor it returns from ``update``. A kernel that reads the format (an
:class:`~.registry.AttentionKernel` whose ``packed_kv`` names it) calls :meth:`PackedKV.attend` and reads the packed
bytes instead of the dense ones; any other function ignores the attachment and computes the same attention from
the dense buffers, which hold the dequantized values. A sparse format whose kernel reads only part of the cache
(Quest's pages, :mod:`.quest`) also offers ``restrict_mask(query, key, mask, is_causal)``, which the fallbacks
apply so that they attend over the same entries. The attachment lives on a fresh view of the key buffer, so it
disappears as soon as anything copies, slices or reorders the tensor, and a stale attachment cannot be read.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

__all__ = ["PackedKV", "attach_packed", "packed_of"]

_ATTR = "_economize_packed"


@runtime_checkable
class PackedKV(Protocol):
    """Packed K/V of one layer that a kernel can attend over directly."""

    #: Format name, matched against :attr:`~.registry.AttentionKernel.packed_kv`.
    format: str

    def attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
        scale: float,
        softcap: float | None = None,
        sinks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attention of ``query`` ``[B, Hq, Q, D]`` over the layer's cache, ``[B, Q, Hq, Dv]`` out.

        ``key`` and ``value`` are the dense buffers the cache returned (the parts that are not packed are read
        from them); ``mask``, ``scale``, ``softcap`` and ``sinks`` are those of
        :func:`~.decode.decode_attention`.
        """
        ...


def attach_packed(key: torch.Tensor, packed: PackedKV) -> torch.Tensor:
    """A view of ``key`` that carries ``packed`` (the buffer itself is left untouched)."""
    view = key.view(key.shape)
    setattr(view, _ATTR, packed)
    return view


def packed_of(key: torch.Tensor) -> PackedKV | None:
    """The :class:`PackedKV` attached to ``key``, or ``None``."""
    return getattr(key, _ATTR, None)
