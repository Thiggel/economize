"""The ReDrafter drafter: a recurrent chain, or beam search verified as the tree of the beams' distinct prefixes."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from ...decoding.cost import CostKind, LayerFootprint
from ...decoding.protocols import Draft, Drafter
from .._drafting import take_rows
from .technique import LOG_0

if TYPE_CHECKING:
    from ...decoding.protocols import Verifier
    from ...decoding.request import DraftRequest
    from ...runtime import ExactPass, KVCache, LayerRunner
    from .technique import RedrafterTechnique

__all__ = ["RedrafterDrafter", "beam_tree"]


def beam_tree(beams: torch.Tensor, depths: list[int]) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """The tree of the distinct prefixes of beams ``[B, W, D]`` (the reference's dynamic tree attention), with the
    beams of row ``b`` cut after ``depths[b]`` tokens.

    Node ``(w, j)`` is the prefix ``beams[b, w, :j + 1]`` and is kept only for the first beam ``w`` that has it. Nodes
    are ordered by depth, then beam.

    Returns:
        ``(tokens, lengths, parents)`` of a tree :class:`~economize.decoding.Draft`: ``[B, T]`` padded with ``-1``,
        the number of nodes of every row and every node's parent ``[B, T]`` (``-1``: a child of the root).
    """
    batch, width, depth = beams.shape
    device = beams.device
    same = (beams[:, :, None] == beams[:, None]).long().cumprod(-1).bool()  # [B, W, W', D]: equal prefixes
    first = same.int().argmax(2)  # [B, W, D]: first beam with the prefix of (w, j)
    beam = torch.arange(width, device=device)
    level = torch.arange(depth, device=device)
    cut = torch.tensor(depths, device=device)
    unique = (first == beam[None, :, None]) & (level[None, None] < cut[:, None, None])
    order = unique.transpose(1, 2).reshape(batch, depth * width)  # depth-major
    index = (order.long().cumsum(-1) - 1).view(batch, depth, width).transpose(1, 2)  # [B, W, D]: node of (w, j)
    lengths = order.sum(-1).tolist()
    nodes = max(lengths, default=0)
    parent = torch.full_like(index, -1)
    if depth > 1:
        parent[..., 1:] = index[..., :-1].gather(1, first[..., :-1])  # the node of the prefix one token shorter
    target = torch.where(unique, index, torch.full_like(index, nodes))  # dropped nodes land in a spare column
    tokens = beams.new_full((batch, nodes + 1), -1)
    tokens.scatter_(1, target.flatten(1), beams.flatten(1))
    parents = parent.new_full((batch, nodes + 1), -1)
    parents.scatter_(1, target.flatten(1), parent.flatten(1))
    return tokens[:, :nodes], lengths, parents[:, :nodes]


class RedrafterDrafter(Drafter):
    """Drafts with the recurrent head of a :class:`~economize.techniques.redrafter.RedrafterTechnique`.

    After every exact pass :meth:`observe` keeps the context ``h_t`` of every row's last processed token ``t``. A
    proposal starts the recurrent state at zero. A chain draft feeds it the committed token ``x_{t+1}`` and then
    every draft, through :meth:`DraftRequest.extend`, so the selector, the draft-length policy and rejection sampling
    see the head's (top-k warped) distribution at every step. With ``beam_width > 1`` and a tree verifier a proposal
    runs the reference's beam search for as many steps as the deepest row may draft and hands the verifier the tree
    of the beams' distinct prefixes (deterministic drafts, ``probs=None``).

    The cost log records one zero-layer ``draft`` event of the network ``"redrafter"`` per step and row; a beam step
    runs the head on all ``W`` beams at once and records ``W`` tokens. Drafts beyond ``beam_length - 1`` are never
    made: cycles longer than ``beam_length`` end early.

    With CUDA graphs (:meth:`enable_graphs`) every chain step and every beam step is replayed from a graph (one per
    batch size); the bookkeeping of the beams' histories and the tree runs eagerly.
    """

    layer_span = (1, 0)
    footprint = LayerFootprint.none("redrafter")

    def __init__(self, technique: RedrafterTechnique) -> None:
        self.technique = technique
        self.source_layers = (technique.adapter.num_layers,)
        self._chain_step: Callable[..., Any] = technique.draft_step
        self._beam_step: Callable[..., Any] = technique.beam_step
        self._context: torch.Tensor | None = None  # h_t of every row

    @property
    def observed_layers(self) -> tuple[int, ...]:
        return ()  # the head reads the exact pass's final output, which every pass returns

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        self._context = None

    def enable_graphs(self, wrap: Callable[[Callable[..., Any]], Callable[..., Any]]) -> bool:
        self._chain_step = wrap(self.technique.draft_step)
        self._beam_step = wrap(self.technique.beam_step)
        return True

    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None:
        final = exact.final
        rows = torch.arange(final.shape[0], device=final.device)
        self._context = self.technique.base(final[rows, columns.to(final.device)])

    def reorder(self, rows: torch.Tensor) -> None:
        self._context = take_rows(self._context, rows)

    def draft_capacity(self, cycle_length: int) -> int:
        config = self.technique.config
        depth = max(0, min(cycle_length - 1, config.num_drafts))
        return depth * config.beam_width

    def drafts_trees(self, verifier: Verifier) -> bool:
        return self.technique.config.beam_width > 1 and verifier.supports_trees

    def propose(self, request: DraftRequest) -> Draft:
        context = self._context
        if context is None:
            raise RuntimeError("RedrafterDrafter.propose needs observe() first")
        config = self.technique.config
        depths = [min(n, config.num_drafts) if a else 0 for n, a in zip(request.budgets, request.active)]
        if self.drafts_trees(request.verifier):
            return self._beams(request, context, depths)
        state = torch.zeros_like(context)
        for _ in range(max(depths, default=0)):
            if not request.drafting:
                break
            logits, state = self._chain_step(context, state, request.tokens.to(context.device))
            request.extend(logits)
        return request.draft()

    def _beams(self, request: DraftRequest, context: torch.Tensor, depths: list[int]) -> Draft:
        depth = max(depths, default=0)
        batch, device = request.batch_size, request.device
        if depth == 0:
            return Draft.empty(batch, device)
        width = self.technique.config.beam_width
        tokens = request.tokens.to(context.device)[:, None].expand(batch, width).contiguous()
        state = context.new_zeros(batch, width, context.shape[-1])
        log_p = torch.full((batch, width), LOG_0, device=context.device)
        log_p[:, 0] = 0.0
        history = tokens.new_empty(batch, width, 0)
        for j in range(depth):
            log_p, tokens, beams, state = self._beam_step(context, state, tokens, log_p)
            history = torch.cat([history.gather(1, beams[..., None].expand(-1, -1, j)), tokens[..., None]], -1)
            for b, d in enumerate(depths):
                if j < d:
                    request.costs[b].record_footprint(CostKind.DRAFT, width, request.ends[b] - 1 + j, self.footprint)
        nodes, lengths, parents = beam_tree(history, depths)
        return Draft(nodes.to(device), lengths, parents.to(device), None)
