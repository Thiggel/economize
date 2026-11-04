"""The drafter of two-model speculative decoding with the draft LM on a device of its own."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import torch

from ...decoding.graphs import static_cache_class
from ...runtime.graphs import StaticStep
from ..draft_model.drafter import DraftModelDrafter

if TYPE_CHECKING:
    from ...decoding.request import DraftRequest
    from ...runtime import KVCache, LayerRunner

__all__ = ["PlacedDraftModelDrafter"]


class PlacedDraftModelDrafter(DraftModelDrafter):
    """:class:`~economize.techniques.draft_model.DraftModelDrafter` for a draft LM on another device than the model.

    Every draft-LM call runs with the draft device current: its inputs (tokens, positions, row mask) are copied
    there first and its float logits are copied back to ``home``, the model's device. The copies are ordinary
    stream-ordered device-to-device transfers, so neither device waits on the host.

    CUDA graphs (:meth:`enable_graphs`): the one-token draft step is captured on the draft device, in a memory pool
    of that device, from inputs already on it; the transfers stay outside the graph. The decoder's own graph wrapper
    records on the model's device and is not used.

    Args:
        runner: The layer runner of the draft LM (on the draft device).
        vocab_size: Vocabulary size of the model.
        home: Device of the model, where the drafts' logits go.
    """

    def __init__(self, runner: LayerRunner, vocab_size: int, home: torch.device) -> None:
        super().__init__(runner, vocab_size)
        self.home = torch.device(home)
        self.draft_device = runner.adapter.device
        self._pool: Any = None

    @contextlib.contextmanager
    def _on_draft(self) -> Iterator[None]:
        """The draft device as the current CUDA device (a no-op on other device types)."""
        if self.draft_device.type == "cuda":
            with torch.cuda.device(self.draft_device):
                yield
        else:
            yield

    def enable_graphs(self, wrap: Callable[[Callable[..., Any]], Callable[..., Any]]) -> bool:
        if static_cache_class(self.draft_runner.adapter.cache_kind) is None or self.draft_device.type != "cuda":
            return False
        with self._on_draft():
            self._pool = torch.cuda.graph_pool_handle()
        pool = self._pool
        self._step_fn.enable(lambda fn: StaticStep(fn, pool=pool))
        return True

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        with self._on_draft():
            super().begin(runner, cache, batch_size)

    def reorder(self, rows: torch.Tensor) -> None:
        with self._on_draft():
            super().reorder(rows)

    def step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        device = self.draft_device
        with self._on_draft():
            logits = self._step_fn(tokens.to(device), positions.to(device), None if valid is None else valid.to(device))
            return logits.to(self.home)

    def _catch_up(self, request: DraftRequest) -> torch.Tensor:
        with self._on_draft():
            return super()._catch_up(request).to(self.home)
