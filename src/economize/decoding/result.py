"""The output of one generated row."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .cost import CostLog

__all__ = ["DecodeResult"]


@dataclass
class DecodeResult:
    """Output of :meth:`~economize.decoding.CycleDecoder.decode` for one row.

    Attributes:
        tokens: Generated tokens, ``[n]`` (prompt excluded).
        cost: Every model call made for this row.
        acceptance: One ``(drafted, accepted)`` pair per verified cycle (lossless decoding only).
        prompt_length: Number of prompt tokens.
        num_layers: Decoder layers of the model.
        confidence: With :attr:`DecodeConfig.record_confidence <economize.DecodeConfig.record_confidence>`, one list per
            entry of ``acceptance``: the drafter's top probability at each drafted token (``nan`` where unknown,
            ``[]`` for a cycle whose drafter did not report it, e.g. trees or prompt lookup). Empty otherwise.
        match: With :attr:`DecodeConfig.record_distributions <economize.DecodeConfig.record_distributions>`, one
            list per entry of ``acceptance``: ``[top-1 agreement, KL(verifier || drafter), TV]`` at each drafted
            token, conditioned on the same prefix (the drafts before it), from
            :func:`~economize.decoding.commit.distribution_match` (``[]`` for a cycle whose drafter did not report
            its logits). Empty otherwise.
    """

    tokens: torch.Tensor
    cost: CostLog
    acceptance: list[tuple[int, int]] = field(default_factory=list)
    prompt_length: int = 0
    num_layers: int = 0
    confidence: list[list[float]] = field(default_factory=list)
    match: list[list[list[float]]] = field(default_factory=list)

    def cost_summary(self) -> dict[str, Any]:
        """Aggregate cost of this generation (see :meth:`CostLog.summary`)."""
        return self.cost.summary(self.num_layers, self.prompt_length, self.tokens.numel())

    @property
    def mean_accepted(self) -> float:
        """Accepted drafts per verified cycle."""
        return sum(a for _, a in self.acceptance) / max(1, len(self.acceptance))
