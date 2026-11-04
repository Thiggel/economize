"""A whole chain proposal of ``k`` greedy draft steps as one CUDA graph (``DecodeConfig.draft_loop_graph``).

With per-step graphs a proposal of ``k`` drafts replays ``k`` graphs, and between two replays the host does the
bookkeeping of :meth:`~economize.decoding.DraftRequest.extend`: it records costs, writes the full logits row into a
float32 buffer, takes the arg-max and scatters the token into the sequence. :class:`ChainGraphs` records the ``k``
steps back to back in one graph per ``k``, each step's token chosen on the device by the drafter's
:class:`~economize.techniques._drafting.greedy_head.GreedyHead` (the fused LM-head kernel where the head allows it)
and fed to the next step, and hands all ``k`` tokens to
:meth:`~economize.decoding.DraftRequest.extend_tokens` at once. Proposals the request does not allow this for
(:meth:`~economize.decoding.DraftRequest.graphed_chain_steps` is ``None``: sampling, dynamic draft lengths, rows
with different budgets) run step by step as before.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from ...decoding.protocols import Draft, Drafter

if TYPE_CHECKING:
    from ...decoding.request import DraftRequest
    from .drafters import GraphedStep
    from .greedy_head import GreedyHead

__all__ = ["ChainGraphs"]


class ChainGraphs(Drafter):
    """Mixin of a graphed chain drafter that can record a proposal of ``k`` greedy steps as one CUDA graph.

    The drafter splits its step into :meth:`_advance` (everything up to the state its head reads, with the step's
    state and cache updates) and the head, and provides :meth:`greedy_head`; its ``step`` must equal the head's
    logits of :meth:`_advance`. It routes its step through ``_step_fn`` (a
    :class:`~economize.techniques._drafting.GraphedStep`); the chains are graphed by the same ``wrap`` once
    ``_step_fn`` is enabled, so they need the drafter's per-step graph contract (persistent state, fixed buffers).
    Put the mixin after :class:`~economize.techniques._drafting.BoundedDrafter` among the bases.

    Attributes:
        fused_head: Choose the tokens with the fused LM-head kernel where the head allows it (``False`` computes
            the logits and their arg-max inside the graph, for ablations).
    """

    _step_fn: GraphedStep
    fused_head: bool = True

    def _advance(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """One draft step up to the state the head reads (``step`` is the head's logits of it)."""
        raise NotImplementedError

    def greedy_head(self, fused: bool = True) -> GreedyHead:
        """The greedy choice of the drafter's head, from the states :meth:`_advance` returns."""
        raise NotImplementedError

    def _chain_ready(self) -> bool:
        """Whether the drafter holds the state a step needs (``step`` raises otherwise)."""
        return True

    def propose(self, request: DraftRequest) -> Draft:
        steps = request.graphed_chain_steps() if self._step_fn.enabled and self._chain_ready() else None
        if steps is None:
            return super().propose(request)
        tokens, top = self._chain(steps)(request.tokens, request.positions,
                                         self._step_fn.all_valid(request.batch_size, request.device))
        request.extend_tokens(tokens, top if request.record_confidence else None)
        return request.draft()

    def _chain(self, steps: int) -> Callable[..., Any]:
        """The graphed chain of ``steps`` draft steps."""
        chains: dict[tuple[int, bool], Callable[..., Any]] = self.__dict__.setdefault("_chains", {})
        key = (steps, self.fused_head)
        if key not in chains:
            head = self.greedy_head(self.fused_head)

            def chain(tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, ...]:
                drafted, tops = [], []
                for _ in range(steps):
                    tokens, top = head.select(self._advance(tokens, positions, valid))
                    positions = positions + 1
                    drafted.append(tokens)
                    tops.append(top)
                return torch.stack(drafted, 1), torch.stack(tops, 1)

            chains[key] = self._step_fn.graph(chain)
        return chains[key]

    def _forget_chains(self) -> None:
        """Drop the chains (after the head changed, e.g. :meth:`slice_head`)."""
        self.__dict__.pop("_chains", None)
