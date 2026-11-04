"""Built-in draft-length policies.

A policy is asked after every draft step which rows stop drafting for the rest of the cycle. Register a new one
with ``@POLICIES.register("name")`` and select it with ``DecodeConfig(draft_policy="name")``.
"""

from __future__ import annotations

import torch

from ..errors import ConfigError
from ..registry import Registry
from .config import DecodeConfig
from .protocols import DraftPolicy

__all__ = ["POLICIES", "ConfidenceStop", "FixedLength"]

POLICIES: Registry[type[DraftPolicy]] = Registry("draft-length policy", "economize.draft_policies")


@POLICIES.register("fixed")
class FixedLength(DraftPolicy):
    """Always draft ``cycle_length - 1`` tokens."""

    may_stop = False

    def stops(self, index: int, logits: torch.Tensor) -> torch.Tensor | None:
        return None


@POLICIES.register("confidence")
class ConfidenceStop(DraftPolicy):
    """Stop a row when the drafter's top probability drops below ``min_confidence`` (never on the first draft,
    so every cycle drafts at least one token)."""

    def __init__(self, config: DecodeConfig) -> None:
        super().__init__(config)
        if config.min_confidence is None:
            raise ConfigError("draft_policy 'confidence' needs min_confidence")
        self.threshold = config.min_confidence

    def stops(self, index: int, logits: torch.Tensor) -> torch.Tensor | None:
        if index == 0:
            return None
        return torch.softmax(logits.float(), -1).amax(-1) < self.threshold
