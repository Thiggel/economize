"""The drafters of sparse-KV self-speculation (MagicDec, TriForce): the whole model on a sparse view of its cache."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from ..early_exit.drafter import SelfDrafter

if TYPE_CHECKING:
    from ...runtime import KVCache, LayerRunner
    from .path import SparseKVPath
    from .retrieval import RetrievalPath

__all__ = ["MagicDecDrafter", "TriForceDrafter"]


class MagicDecDrafter(SelfDrafter):
    """The drafter of :class:`~economize.techniques.sparse_kv_draft.MagicDecTechnique`.

    Every draft step (chain or tree) runs all layers, the full-attention ones on a StreamingLLM view of the cache. The
    view is computed from the step's positions on the device, so the drafter keeps no state and its steps can be
    replayed from CUDA graphs.
    """

    def __init__(self, path: SparseKVPath, runner: LayerRunner) -> None:
        super().__init__(path, runner)


class TriForceDrafter(SelfDrafter):
    """The drafter of :class:`~economize.techniques.sparse_kv_draft.TriForceTechnique`.

    The first draft step of a generation runs eagerly on the whole cache and selects each full-attention layer's
    chunks (:class:`~.retrieval.RetrievalPath`); its logits are those of the full model. Later steps, chain or
    tree, read the selection from persistent buffers and the generation's tail (:attr:`~.retrieval.RetrievalPath.tail`)
    as an input tensor, so they can be replayed from CUDA graphs keyed by the tail's width. A tree step receives the
    tail as its state (:meth:`tree_state`). :meth:`reorder` keeps the selection of the surviving rows.
    """

    technique: RetrievalPath

    def __init__(self, path: RetrievalPath, runner: LayerRunner) -> None:
        super().__init__(path, runner)
        self._sparse_fn: Callable[..., Any] | None = None
        self._rows: tuple[int, torch.device] = (0, torch.device("cpu"))

    @property
    def path(self) -> RetrievalPath:
        """The retrieval path, which holds the current selection."""
        return self.technique

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        super().begin(runner, cache, batch_size)
        self.technique.reset()

    def enable_graphs(self, wrap: Callable[[Callable[..., Any]], Callable[..., Any]]) -> bool:
        enabled = super().enable_graphs(wrap)
        self._sparse_fn = self._step_fn.graph(self._sparse_step)
        return enabled

    def step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        if self._cache is None:
            raise RuntimeError(f"{type(self).__name__}.step needs begin() first")
        self._rows = (tokens.shape[0], tokens.device)
        tail = self.technique.tail
        if tail is not None:
            if self._sparse_fn is None:
                return self._sparse_step(tokens, positions, valid, tail)
            if valid is None:
                valid = self._step_fn.all_valid(tokens.shape[0], tokens.device)
            return self._sparse_fn(tokens, positions, valid, tail)
        self.technique.select_next(positions, valid)
        try:
            logits = self._step(tokens, positions, valid)
        finally:
            self.technique.finish_selection()
        return logits

    def _sparse_step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None,
                     tail: torch.Tensor) -> torch.Tensor:
        """A chain step on the selection, reading the tail ``tail`` ``[B, W]``."""
        with self.technique.using_tail(tail):
            return self._step(tokens, positions, valid)

    def tree_state(self) -> tuple[torch.Tensor, ...]:
        """The generation's tail ``[B, W]`` (empty ``W = 0`` without a gathered selection)."""
        tail = self.technique.tail
        if tail is None:
            tail = torch.zeros(self._rows[0], 0, dtype=torch.long, device=self._rows[1])
        return (tail,)

    def tree_step(self, tokens: torch.Tensor, parents: torch.Tensor, base: torch.Tensor,
                  *states: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """The draft path over a tree's frontier on the selection; ``states`` holds the frontier's copies of the
        tail ``[B, W_f, W]``, returned unchanged as the frontier's own states."""
        with self.technique.using_tail(states[0] if states else None):
            logits, _ = super().tree_step(tokens, parents, base)
        return logits, tuple(states)

    def reorder(self, rows: torch.Tensor) -> None:
        self.technique.reorder(rows)
