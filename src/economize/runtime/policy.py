"""Per-layer decisions for dynamic-depth techniques (layer skipping, early exit, token routing).

A :class:`LayerPolicy` is consulted by :meth:`economize.runtime.LayerRunner.run` before every layer. It returns a
:class:`LayerAction` for the whole batch, a :class:`Route` naming the tokens that run the layer, or a boolean
``[B, S]`` tensor (converted to a :class:`Route`).
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import torch

if TYPE_CHECKING:
    from .context import LayerContext

__all__ = ["Decision", "LayerAction", "LayerPolicy", "Route"]


class LayerAction(enum.Enum):
    """What happens to every token at one layer."""

    #: Run the layer.
    RUN = "run"
    #: Pass the hidden state through unchanged and write no K/V (later queries see stale slots at this layer).
    SKIP = "skip"
    #: Pass the hidden state through, but write the K/V the layer would have produced for it.
    SKIP_FILL_KV = "skip_fill_kv"


@dataclass(frozen=True, eq=False)
class Route:
    """The tokens of each row that run a layer; the others skip it with their K/V filled.

    Attributes:
        indices: Token columns ``[B, K]``. Within a row they must be distinct (padding entries included).
        valid: Which entries are real ``[B, K]``; rows routing fewer than ``K`` tokens pad with ``False``.
    """

    indices: torch.Tensor
    valid: torch.Tensor

    @classmethod
    def from_mask(cls, mask: torch.Tensor) -> Route:
        """Route the tokens where ``mask`` ``[B, S]`` is true; ``K`` is the largest count of any row."""
        count = int(mask.sum(-1).max())
        order = torch.sort(mask.to(torch.int8), dim=-1, descending=True, stable=True).indices[:, :count]
        return cls(order, mask.gather(1, order))


#: What :meth:`LayerPolicy.decide` may return.
Decision: TypeAlias = LayerAction | Route | torch.Tensor


class LayerPolicy(ABC):
    """Chooses, before each layer, which tokens run it."""

    @abstractmethod
    def decide(self, index: int, hidden: torch.Tensor, ctx: LayerContext) -> Decision:
        """Decision for layer ``index`` (1-indexed) given its input ``hidden`` ``[B, S, d]``."""

    def after(self, index: int, hidden: torch.Tensor, ctx: LayerContext) -> torch.Tensor | None:
        """Called with the output ``hidden`` of layer ``index``.

        Policies that track state (exit confidence) read it; a policy that returns a tensor replaces the hidden
        state passed on to the next layer (e.g. to freeze the rows that exited). The default returns ``None``.
        """
        return None

    def reorder(self, rows: torch.Tensor) -> None:
        """Keep only the batch rows ``rows`` (a LongTensor of row indices, in their new order) of any per-row state.

        The decoding loop calls it whenever rows leave the batch after the prefill, so that per-row state kept
        across passes (e.g. where each row's prompt ends) stays aligned with the rows of later passes. The default
        does nothing.
        """
