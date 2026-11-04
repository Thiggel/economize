"""The drafter of PARD: the catch-up and all drafts of a cycle in one draft-LM call."""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from ...decoding.protocols import Draft, draft_footprint
from ..draft_model.drafter import CATCHUP, DraftModelDrafter

if TYPE_CHECKING:
    from ...decoding.request import DraftRequest
    from ...runtime import KVCache, LayerRunner

__all__ = ["PardDrafter"]


class PardDrafter(DraftModelDrafter):
    """Drafts ``n`` tokens per cycle with one call of a PARD draft LM.

    Per active row the call reads the owed committed tokens ``start .. end - 1`` (as the catch-up of
    :class:`~economize.techniques.draft_model.DraftModelDrafter`) followed by ``n - 1`` mask tokens at the positions
    ``end .. end + n - 2``, where ``n = min(budget, max_parallel)``. The logits at the last committed token give the
    first draft and the logits at the ``i``-th mask the draft ``i + 1``, so the drafts of a cycle do not depend on
    each other. The masks' K/V lie in the draft LM's cache beyond the row's ``done``, so they are never attended to
    and the next cycle's catch-up overwrites them (the reference code rolls its cache back by ``n - 1``). A row
    whose budget exceeds ``max_parallel`` drafts ``max_parallel`` tokens.

    Cost log: the owed tokens beyond the first are a ``"draft_catchup"`` event and every draft a ``"draft"`` event
    of one token, as for the two-model drafter (the step index of a draft is its depth, although all drafts come
    from one call).

    CUDA graphs (:meth:`enable_graphs`): the call is replayed from a graph of width ``n + 1`` whenever every row
    owes at most two tokens (after a rejection or a fully accepted cycle), one graph per ``(batch size, n)``. The
    first cycle (the prompt) and rows that skipped cycles run eagerly.

    Args:
        runner: The layer runner of the draft LM.
        vocab_size: Vocabulary size of the model.
        mask_token_id: Token id of the mask.
        max_parallel: The most drafts of one call.
    """

    def __init__(self, runner: LayerRunner, vocab_size: int, mask_token_id: int, max_parallel: int) -> None:
        super().__init__(runner, vocab_size)
        self.mask_token_id = mask_token_id
        self.max_parallel = max_parallel
        self._parallel_graph: Callable[..., Any] | None = None

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        previous = self._cache
        super().begin(runner, cache, batch_size)
        if self._cache is not previous:
            self._parallel_graph = None  # it captured the buffers of the previous cache

    def propose(self, request: DraftRequest) -> Draft:
        if self._cache is None:
            raise RuntimeError("PardDrafter.propose needs begin() first")
        if not request.drafting:
            return request.draft()
        active = request.active
        drafts = min(max(b for b, on in zip(request.budgets, active) if on), self.max_parallel)
        starts = [min(verified, end - 1) for verified, end in zip(self._verified(request), request.ends)]
        owed = [end - start if on else 0 for start, end, on in zip(starts, request.ends, active)]
        masks = [min(budget, drafts) - 1 if on else 0 for budget, on in zip(request.budgets, active)]
        graphed = self._step_fn.enabled and max(owed) <= 2
        width = drafts + 1 if graphed else max(o + m for o, m in zip(owed, masks))
        device = request.device
        offsets = torch.arange(width, device=device)
        start = torch.tensor(starts, device=device)
        count = torch.tensor(owed, device=device)
        committed = offsets < count[:, None]
        valid = offsets < (count + torch.tensor(masks, device=device))[:, None]
        positions = start[:, None] + offsets
        tokens = request.seq.gather(1, positions.clamp(max=request.scratch)).masked_fill(~committed, self.mask_token_id)
        positions = positions.masked_fill(~valid, 0)
        index = ((count - 1)[:, None] + torch.arange(drafts, device=device)).clamp(0, width - 1)
        if graphed:
            if self._parallel_graph is None:
                run = weakref.WeakMethod(self._run)  # as GraphedStep holds the step: no reference cycle
                self._parallel_graph = self._step_fn.graph(lambda *inputs: run()(*inputs))  # type: ignore[misc]
            logits = self._parallel_graph(tokens, positions, valid)
            logits = logits.gather(1, index.to(logits.device)[..., None].expand(-1, -1, logits.shape[-1]))
        else:  # e.g. the prompt: the logits of the last committed token and of the masks only
            logits = self._run(tokens, positions, valid, keep=index)
        for b, on in enumerate(active):
            if on:
                if owed[b] > 1:
                    request.costs[b].record_footprint(CATCHUP, owed[b] - 1, starts[b], draft_footprint(self))
                self._done[b] = self._anchor[b] = request.ends[b]
        for i in range(drafts):
            if not request.drafting:
                break
            request.extend(logits[:, i])
        return request.draft()
