"""The drafters of sparse-KV self-speculation (MagicDec, TriForce): the whole model on a sparse view of its cache."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    tree, read the selection from persistent buffers and can be replayed from CUDA graphs. :meth:`reorder` keeps the
    selection of the surviving rows.
    """

    technique: RetrievalPath

    def __init__(self, path: RetrievalPath, runner: LayerRunner) -> None:
        super().__init__(path, runner)

    @property
    def path(self) -> RetrievalPath:
        """The retrieval path, which holds the current selection."""
        return self.technique

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        super().begin(runner, cache, batch_size)
        self.technique.reset()

    def step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        if self._cache is None:
            raise RuntimeError(f"{type(self).__name__}.step needs begin() first")
        if self.technique.selected:
            return self._step_fn(tokens, positions, valid)
        self.technique.select_next(positions, valid)
        try:
            logits = self._step(tokens, positions, valid)
        finally:
            self.technique.finish_selection()
        return logits

    def reorder(self, rows: torch.Tensor) -> None:
        self.technique.reorder(rows)
