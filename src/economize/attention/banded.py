"""Band-shaped attention masks of one pass over whole sequences, attended without a dense mask.

A pass that runs every row from position 0 without padding (the two passes of DuoAttention's gate training, any
teacher pass over whole sequences) needs three mask shapes only: causal (full-attention layers), the last ``window``
positions (sliding-window layers), and the first ``sinks`` positions plus the last ``window`` ones (DuoAttention's
streaming heads). Given as a dense ``[Q, K]`` boolean tensor, each of them makes ``scaled_dot_product_attention``
run its masked kernel (memory-efficient or math), which computes all ``Q * K`` scores, so a sliding-window layer
over 32K tokens costs as much as a full one and a causal layer twice as much as under ``is_causal``; the mask
itself takes ``Q * K`` bytes.

:class:`BandMask` describes such a mask by its window and sinks. The ``sdpa`` function of economize
(:mod:`.sdpa_guard`) and the ``"economize"`` kernel accept any :class:`StructuredMask` in place of the mask tensor
and let it attend: a band that reaches back to position 0 runs as one ``is_causal`` call, a bounded one as one
masked call per block of ``block`` queries over the keys that block can see (the sinks and the window). Every query
attends over the same keys as under the dense mask, so the outputs agree up to floating-point rounding, and
autograd differentiates through the blocks. :func:`accepts_structured_masks` tells whether an implementation
takes such masks; other layouts are added by subclassing :class:`StructuredMask`.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from ..errors import ConfigError
from .registry import ATTENTION_KERNELS

__all__ = ["AttentionForward", "BandMask", "StructuredMask", "accepts_structured_masks"]

#: A ``transformers`` attention function: ``(module, query, key, value, mask, **kwargs) -> (out [B, Q, H, D], _)``.
AttentionForward = Callable[..., tuple[torch.Tensor, Any]]

#: Default queries per block of a bounded band (the window, clipped to this range).
_MIN_BLOCK, _MAX_BLOCK = 128, 512


class StructuredMask(abc.ABC):
    """A mask passed to the attention function in place of a mask tensor, which then lets the mask attend.

    Subclasses describe a pattern that is cheaper to attend by its structure than through a dense mask.
    """

    @abc.abstractmethod
    def attend(
        self,
        forward: AttentionForward,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """The attention output ``[B, Q, H, D]`` under this mask.

        Args:
            forward: The attention function to run the pieces on; it takes a boolean mask tensor or ``None``
                (causal) in the mask argument.
            module: The attention module.
            query: ``[B, H, Q, D]``.
            key: ``[B, H_kv, K, D]``.
            value: ``[B, H_kv, K, D_v]``.
            kwargs: The call's other keyword arguments (``dropout``, ``scaling``, ...), passed on to ``forward``.
        """

    @abc.abstractmethod
    def dense(self, device: torch.device) -> torch.Tensor:
        """The same mask as a ``[1, 1, Q, K]`` boolean tensor, ``True`` where attention is allowed."""


@dataclass(frozen=True)
class BandMask(StructuredMask):
    """Query ``q`` of a pass over positions ``0 .. length - 1`` sees key ``k <= q`` when ``q - k < window`` or
    ``k < sinks``.

    ``window=None`` is causal attention, ``BandMask(n, window=w)`` a sliding window of ``w`` positions (the query's
    own included) and ``BandMask(n, window=r, sinks=s)`` DuoAttention's streaming heads.

    Attributes:
        length: Tokens per row; queries and keys are positions ``0 .. length - 1`` in order.
        window: Most recent positions a query sees, its own included (``None``: all earlier ones).
        sinks: Positions from the start every query sees.
        block: Queries per attention call of a bounded band (default: the window, clipped to 128 .. 512). Smaller
            blocks compute fewer masked scores in more calls.
    """

    length: int
    window: int | None = None
    sinks: int = 0
    block: int | None = None
    _masks: dict[tuple[int, int, int, int, torch.device], torch.Tensor] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.length < 1:
            raise ConfigError(f"a band mask needs at least one token, got length {self.length}")
        if self.window is not None and self.window < 1:
            raise ConfigError(f"a band mask's window must be positive, got {self.window}")
        if self.sinks < 0:
            raise ConfigError(f"a band mask's sinks must be non-negative, got {self.sinks}")
        if self.block is not None and self.block < 1:
            raise ConfigError(f"a band mask's block must be positive, got {self.block}")

    @property
    def causal(self) -> bool:
        """Whether every query sees all earlier positions (no window, or one at least as long as the pass)."""
        return self.window is None or self.window >= self.length

    @property
    def block_size(self) -> int:
        """Queries per attention call of a bounded band."""
        if self.block is not None:
            return self.block
        return min(max(self.window or self.length, _MIN_BLOCK), _MAX_BLOCK)

    def dense(self, device: torch.device) -> torch.Tensor:
        positions = torch.arange(self.length, device=device)
        return self._visible(positions[:, None], positions[None, :])[None, None]

    def attend(
        self,
        forward: AttentionForward,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        if query.shape[2] != self.length or key.shape[2] != self.length:
            raise ConfigError(
                f"a band mask over {self.length} tokens got {query.shape[2]} queries and {key.shape[2]} keys"
            )
        kwargs.pop("is_causal", None)
        if self.causal:
            out, _ = forward(module, query, key, value, None, is_causal=True, **kwargs)
            return out, None
        window = int(self.window or self.length)
        outs = []
        for start in range(0, self.length, self.block_size):
            end = min(start + self.block_size, self.length)
            low = max(0, start - window + 1)  # the oldest key a query of the block sees through the window
            sinks = min(self.sinks, low)  # sinks that fall outside the window
            keys, values = key[:, :, low:end], value[:, :, low:end]
            if sinks:
                keys = torch.cat([key[:, :, :sinks], keys], dim=2)
                values = torch.cat([value[:, :, :sinks], values], dim=2)
            mask = self._block_mask(end - start, start - low, sinks, max(0, self.sinks - low), query.device)
            out, _ = forward(module, query[:, :, start:end], keys, values, mask, is_causal=False, **kwargs)
            outs.append(out)
        return torch.cat(outs, dim=1), None

    # ---- internals ---------------------------------------------------------------------------------------------
    def _visible(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        visible = keys <= queries
        if self.window is not None:
            visible = visible & ((queries - keys < self.window) | (keys < self.sinks))
        return visible

    def _block_mask(self, queries: int, offset: int, sinks: int, early: int, device: torch.device) -> torch.Tensor:
        """``[1, 1, queries, sinks + offset + queries]`` mask of one block of queries over its keys: ``sinks`` sink
        keys, then the window keys, which start ``offset`` positions before the block's first query and whose first
        ``early`` are sink positions too. Blocks with the same numbers share the mask (every full block past the
        sinks and the first window)."""
        cache_key = (queries, offset, sinks, early, device)
        mask = self._masks.get(cache_key)
        if mask is None:
            q = torch.arange(queries, device=device)[:, None] + offset + sinks
            k = torch.arange(sinks + offset + queries, device=device)[None, :]
            window = int(self.window or self.length)
            mask = (k < sinks) | ((k <= q) & ((q - k < window) | (k - sinks < early)))
            mask = self._masks[cache_key] = mask[None, None]
        return mask


def accepts_structured_masks(implementation: str) -> bool:
    """Whether attention implementation ``implementation`` takes a :class:`StructuredMask` in place of a mask tensor.

    True for a registered kernel that declares ``structured_masks`` and for ``"sdpa"`` while economize's function
    (:mod:`.sdpa_guard`, which :meth:`~economize.arch.ArchAdapter.validate` installs for ``sdpa`` models) is the
    registered one.
    """
    if implementation in ATTENTION_KERNELS:
        return ATTENTION_KERNELS.get(implementation).structured_masks
    if implementation != "sdpa":
        return False
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from .sdpa_guard import sdpa_attention_forward

    return ALL_ATTENTION_FUNCTIONS.get("sdpa") is sdpa_attention_forward
