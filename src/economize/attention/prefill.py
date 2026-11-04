"""Sparse prefill attached to an attention module.

A sparse-prefill technique (MInference's patterns, :mod:`.minference`) attaches a :class:`SparsePrefill` to the
attention modules of the layers it covers. The ``"economize"`` attention implementation asks the attachment for a
:class:`~.minference.SparseIndex` whenever the module attends a fresh prompt (every query at its own position from
slot 0, no cache before it) of at least :attr:`SparsePrefill.min_queries` tokens, and attends over the indexed keys
only (:func:`~.minference.sparse_prefill_attention`, or the masked fallback on CPU). Every other call of the module,
decoding steps and chunked prefills among them, attends densely. The attachment is a plain attribute of the module,
so it belongs to that model alone and :func:`detach_prefill` removes it without a trace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from .minference import SparseIndex

__all__ = ["SparsePrefill", "attach_prefill", "detach_prefill", "prefill_of"]

_ATTR = "_economize_prefill"


@runtime_checkable
class SparsePrefill(Protocol):
    """Chooses the keys every query of a fresh prompt attends to, in one layer."""

    @property
    def name(self) -> str:
        """Name of the pattern family; the attention call is logged on path ``"kernel_<name>"`` (or as fallback
        reason ``"<name>_<reason>"``)."""
        ...

    @property
    def min_queries(self) -> int:
        """Shortest prompt (tokens per row) the pattern applies to; shorter prompts attend densely."""
        ...

    def index(self, query: torch.Tensor, key: torch.Tensor, scale: float) -> SparseIndex:
        """The keys every query attends to.

        Args:
            query: ``[B, Hq, Q, D]``, the prompt's queries at positions ``0 .. Q - 1``.
            key: ``[B, Hkv, Q, D]``, the prompt's keys.
            scale: The layer's score multiplier.
        """
        ...


def attach_prefill(module: torch.nn.Module, prefill: SparsePrefill) -> None:
    """Make ``module``'s fresh prompts attend through ``prefill`` (replaces an earlier attachment)."""
    object.__setattr__(module, _ATTR, prefill)  # a plain attribute: not a submodule, not part of the state dict


def detach_prefill(module: torch.nn.Module) -> None:
    """Remove ``module``'s attachment, if any."""
    if _ATTR in module.__dict__:
        object.__delattr__(module, _ATTR)


def prefill_of(module: torch.nn.Module) -> SparsePrefill | None:
    """The :class:`SparsePrefill` attached to ``module``, or ``None``."""
    return module.__dict__.get(_ATTR)
