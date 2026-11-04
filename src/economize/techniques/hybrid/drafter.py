"""The drafter of hybrid drafting."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from ...decoding.protocols import Draft, Drafter, Verifier
from ...decoding.request import DraftRequest
from ...errors import ConfigError
from .._drafting import DelegatingDrafter, PartialDrafter
from ..lookup.proposal import one_hot_probs

if TYPE_CHECKING:
    from ...runtime import ExactPass, KVCache, LayerRunner

__all__ = ["HybridDrafter", "merge_drafts"]


class HybridDrafter(DelegatingDrafter):
    """The primary drafter (prompt lookup) for the rows it serves, ``fallback`` for the others.

    Both drafters see every exact pass (:meth:`observe`) and every reordering, so their state stays in step whether
    or not they drafted in a cycle. In a cycle the fallback serves a request of its own in which the primary's rows
    have no budget: those rows stay inactive (they neither write the cache nor record draft steps). A fallback that
    drafts trees gets the primary's chains merged in as single-branch trees. Everything else (the layers it reads
    and runs, its cost footprint, its cache) is the fallback's.

    Args:
        primary: The drafter tried first (a :class:`~economize.techniques._drafting.PartialDrafter`, e.g. a
            :class:`~economize.techniques.lookup.LookupDrafter`).
        fallback: The drafter of the rows the primary leaves.
        max_drafts: Most tokens the fallback drafts per row and cycle (``None``: the cycle's budget).
        vocab_size: Vocabulary size (for point-mass distributions of the primary's drafts).

    Raises:
        ConfigError: ``fallback`` is not a :class:`~economize.decoding.WrappableDrafter`.
    """

    def __init__(self, primary: PartialDrafter, fallback: Drafter, max_drafts: int | None, vocab_size: int) -> None:
        super().__init__(fallback, "the hybrid fallback")
        self.primary = primary
        self.max_drafts, self.vocab_size = max_drafts, vocab_size

    @property
    def fallback(self) -> Drafter:
        """The drafter of the rows the primary leaves (:attr:`inner`)."""
        return self.inner

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        self.primary.begin(runner, cache, batch_size)
        super().begin(runner, cache, batch_size)

    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None:
        self.primary.observe(exact, columns)
        super().observe(exact, columns)

    def reorder(self, rows: torch.Tensor) -> None:
        self.primary.reorder(rows)
        super().reorder(rows)

    def draft_capacity(self, cycle_length: int) -> int:
        return max(cycle_length - 1, self.fallback.draft_capacity(cycle_length))

    def drafts_trees(self, verifier: Verifier) -> bool:
        """Whether the fallback drafts trees (the primary's chains then become single-branch trees)."""
        return self.fallback.drafts_trees(verifier)

    def enable_graphs(self, wrap: Callable[[Callable[..., Any]], Callable[..., Any]]) -> bool:
        return False  # the fallback runs eagerly: graph replay is not validated with the hybrid's sub-requests

    def step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        raise NotImplementedError("HybridDrafter drafts whole proposals (propose), not single steps")

    def propose(self, request: DraftRequest) -> Draft:
        copied = self.primary.propose_partial(request)
        cap = self.max_drafts
        budgets = [
            0 if n > 0 else (budget - done if cap is None else min(budget - done, cap))
            for n, budget, done in zip(copied.lengths, request.budgets, request.lengths)
        ]
        if not any(budgets):
            return copied
        sub = DraftRequest(
            request.seq,
            request.ends,
            budgets,
            selector=request.selector,
            verifier=request.verifier,
            policy=request.policy,
            costs=request.costs,
            footprint=self.footprint,
            selectors=request.selectors,
        )
        drafted = self.fallback.propose(sub)
        for b, n in enumerate(drafted.lengths):
            request.lengths[b] += n
        use_fallback = [n == 0 for n in copied.lengths]
        return merge_drafts(copied, drafted, use_fallback, self.vocab_size)


def merge_drafts(first: Draft, second: Draft, use_second: list[bool], vocab_size: int) -> Draft:
    """Row ``b`` of ``second`` where ``use_second[b]``, else row ``b`` of ``first`` (``first`` must be a chain).

    The result is a tree when ``second`` is one (chains become single-branch trees). When one side has draft
    distributions and the other does not, the other side's tokens get point masses."""
    if first.is_tree:
        raise ConfigError("merge_drafts expects a chain as its first draft")
    lengths = [s if pick else f for f, s, pick in zip(first.lengths, second.lengths, use_second)]
    width = max(lengths, default=0)
    batch = len(lengths)
    device = first.tokens.device if first.width else second.tokens.device
    if width == 0:
        return Draft.empty(batch, device)
    pick = torch.tensor(use_second, device=device)[:, None]

    def widen(tensor: torch.Tensor, fill: float) -> torch.Tensor:
        tensor = tensor[:, :width].to(device)
        if tensor.shape[1] == width:
            return tensor
        pad = torch.full((batch, width - tensor.shape[1], *tensor.shape[2:]), fill, dtype=tensor.dtype, device=device)
        return torch.cat([tensor, pad], 1)

    tokens = torch.where(pick, widen(second.tokens, -1), widen(first.tokens, -1))
    parents = None
    if second.parents is not None:
        chain = torch.arange(width, device=device)[None].expand(batch, -1) - 1
        parents = torch.where(pick, widen(second.parents, -1), chain)
        parents = parents.masked_fill(tokens < 0, -1)
    probs = None
    if first.probs is not None or second.probs is not None:
        p1 = widen(first.probs, 0.0) if first.probs is not None else one_hot_probs(widen(first.tokens, -1), vocab_size)
        p2 = widen(second.probs, 0.0) if second.probs is not None else one_hot_probs(widen(second.tokens, -1),
                                                                                     vocab_size)
        probs = torch.where(pick[..., None], p2, p1)
    return Draft(tokens, lengths, parents, probs)
