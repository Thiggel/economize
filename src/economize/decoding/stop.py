"""Stop conditions of the decoding loop."""

from __future__ import annotations

from collections.abc import Callable, Collection

import torch

__all__ = ["StopChecker", "StopCondition"]

#: A stop token id, a collection of stop token ids, or a predicate on all tokens so far (prompt included, a 1-D
#: tensor on the model's device).
StopCondition = int | Collection[int] | Callable[[torch.Tensor], bool] | None


class StopChecker:
    """A :data:`StopCondition` checked after every committed token.

    The decoding loops (:class:`~economize.decoding.CycleDecoder`, :mod:`economize.distributed.ssd`) normalise a user's
    stop condition into one of these once and call it after each commit.

    Args:
        stop: Stop token id(s), a predicate on all tokens so far, or ``None`` (never stops).
    """

    def __init__(self, stop: StopCondition) -> None:
        self.ids: frozenset[int] | None = None
        self.predicate: Callable[[torch.Tensor], bool] | None = None
        if callable(stop):
            self.predicate = stop
        elif stop is not None:
            self.ids = frozenset({stop} if isinstance(stop, int) else (int(i) for i in stop))

    def __call__(self, token: int, row: torch.Tensor, position: int) -> bool:
        """Whether ``token``, just committed at ``position`` of ``row`` (the device row of all tokens), stops."""
        if self.ids is not None:
            return token in self.ids
        if self.predicate is not None:
            return bool(self.predicate(row[: position + 1]))
        return False
