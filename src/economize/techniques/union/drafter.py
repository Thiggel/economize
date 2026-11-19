"""The drafters of union drafting: the merged trie of several drafters, and the cascade of one drafter along another."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from ...decoding.policies import FixedLength
from ...decoding.protocols import Draft, Drafter, DraftPolicy, Verifier, draft_footprint
from ...decoding.request import DraftRequest
from ...decoding.tree_search import accepted_chain, chain_logits
from ...errors import ConfigError
from ...runtime.transfer import device_tensor
from .merge import MergedTree, TreeBuilder, accepted_path, build_tree, leading_counts, merge_drafts, path_source

if TYPE_CHECKING:
    from ...decoding.cost import CostLog, LayerFootprint
    from ...decoding.protocols import TokenSelector
    from ...runtime import ExactPass, KVCache, LayerRunner
    from .._drafting import StepWrap

__all__ = ["CascadeDrafter", "CycleRecord", "ForcedRequest", "UnionDrafter", "UnionStats"]


@dataclass(frozen=True)
class CycleRecord:
    """What one row accepted from a verified union tree.

    Attributes:
        accepted: Draft tokens the exact pass accepted.
        leading: For every member, the leading tokens of the accepted path it proposed (under greedy verification
            of chain members, the tokens it would have had accepted drafting alone).
        source: The member the accepted path's deepest node came from (its index as a string), ``"shared"`` when
            several members proposed that node, ``"none"`` when no draft was accepted.
        nodes: Nodes of the verified tree.
    """

    accepted: int
    leading: tuple[int, ...]
    source: str
    nodes: int


@dataclass
class UnionStats:
    """Per-row, per-cycle acceptance of a union drafter (kept across generations until :meth:`reset`).

    Attributes:
        members: Technique name of every member.
        cycles: One record per row and verified cycle. The last cycle of a generation (in which every remaining row
            finishes) is not observed and so not recorded.
    """

    members: tuple[str, ...]
    cycles: list[CycleRecord] = field(default_factory=list)

    def record(self, record: CycleRecord) -> None:
        self.cycles.append(record)

    def reset(self) -> None:
        """Forget every recorded cycle."""
        self.cycles.clear()

    def summary(self) -> dict[str, object]:
        """Means over the recorded cycles: accepted tokens, every member's leading tokens (``E[member]``), the best
        member per cycle (``E[max]``), tree nodes, and the count of every source."""
        n = len(self.cycles)
        if n == 0:
            return {"members": list(self.members), "cycles": 0}
        leading = [sum(c.leading[i] for c in self.cycles) / n for i in range(len(self.members))]
        return {
            "members": list(self.members),
            "cycles": n,
            "mean_accepted": sum(c.accepted for c in self.cycles) / n,
            "mean_leading": leading,
            "mean_max_leading": sum(max(c.leading) for c in self.cycles) / n,
            "mean_nodes": sum(c.nodes for c in self.cycles) / n,
            "sources": dict(Counter(c.source for c in self.cycles)),
        }


def _overrides_observe_logits(drafter: Drafter) -> bool:
    return type(drafter).observe_logits is not Drafter.observe_logits


class UnionDrafter(Drafter):
    """Every member drafts for every row; the drafts are merged into one tree and verified in one exact pass.

    The members draft one after another into the same sequence buffer, each through a request of its own with its
    own budget (at most ``max_drafts[i]``) and its own cost footprint, so the cost log holds every member's draft
    steps. After a tree pass every member observes the chain of the accepted path
    (:func:`~economize.decoding.tree_search.accepted_chain`), as a chain drafter does after a chain pass, and every
    member sees every reordering of the batch.

    Args:
        members: The member drafters.
        max_drafts: Most tokens every member drafts per row and cycle (``None``: the cycle's budget).
        vocab_size: Vocabulary size (for point-mass draft distributions).
        stats: Where the acceptance of every verified cycle goes.
    """

    def __init__(self, members: Sequence[Drafter], max_drafts: Sequence[int | None], vocab_size: int,
                 stats: UnionStats) -> None:
        if len(members) != len(max_drafts):
            raise ConfigError("a union needs one max_drafts entry per member")
        self.members = list(members)
        self.max_drafts = list(max_drafts)
        self.vocab_size = vocab_size
        self.stats = stats
        self.source_layers = tuple(sorted({layer for m in self.members for layer in m.source_layers}))
        self.streams_prompt = all(m.streams_prompt for m in self.members)
        self._trash = 0
        self._index: torch.Tensor | None = None
        self._pending: MergedTree | None = None

    @property
    def footprint(self) -> LayerFootprint:  # type: ignore[override]
        """The first member's layers (every member records its own draft steps with its own footprint)."""
        return draft_footprint(self.members[0])

    @property
    def observed_layers(self) -> tuple[int, ...]:
        return tuple(sorted({layer for m in self.members for layer in m.observed_layers}))

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        if runner.has_recurrent_layers:
            raise ConfigError("a union drafter needs a model without recurrent layers (it verifies draft trees whose "
                              "accepted path its members observe as a chain)")
        self._trash = cache.trash_slot
        self._pending = None
        self._index = None
        for member in self.members:
            member.begin(runner, cache, batch_size)

    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None:
        ctx = exact.context
        self._index = None
        pending, self._pending = self._pending, None
        if ctx is None or ctx.tree is None:
            for member in self.members:
                member.observe(exact, columns)
            return
        width = (pending.depth if pending is not None else ctx.tree.visible.shape[1]) + 1
        chain, path_columns, self._index = accepted_chain(exact, columns, width, self._trash)
        if pending is not None:
            self._record(pending, columns.tolist())
        for member in self.members:
            member.observe(chain, path_columns)

    def _record(self, tree: MergedTree, columns: list[int]) -> None:
        count = len(self.members)
        for b, column in enumerate(columns):
            path = accepted_path(tree.parents[b], column - 1)
            masks = tree.masks[b]
            self.stats.record(CycleRecord(len(path), tuple(leading_counts(masks, path, count)),
                                          path_source(masks, path), tree.draft.lengths[b]))

    def observe_logits(self, logits: torch.Tensor) -> None:
        chained: torch.Tensor | None = None
        for member in self.members:
            if _overrides_observe_logits(member):
                if chained is None:
                    chained = chain_logits(logits, self._index)
                member.observe_logits(chained)

    def reorder(self, rows: torch.Tensor) -> None:
        for member in self.members:
            member.reorder(rows)

    def draft_capacity(self, cycle_length: int) -> int:
        return sum(self._limit(i, member.draft_capacity(cycle_length)) for i, member in enumerate(self.members))

    def enable_graphs(self, wrap: StepWrap) -> bool:
        return False  # the members draft eagerly: graph replay is not validated with the union's sub-requests

    def drafts_trees(self, verifier: Verifier) -> bool:
        return True

    def step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} drafts whole proposals (propose), not single steps")

    def _limit(self, index: int, budget: int) -> int:
        cap = self.max_drafts[index]
        return budget if cap is None else min(budget, cap)

    def _sub_request(self, request: DraftRequest, member: Drafter, budgets: Sequence[int],
                     policy: DraftPolicy | None = None) -> DraftRequest:
        """A request of ``member`` over the same rows and buffer as ``request``, with ``budgets``."""
        return DraftRequest(
            request.seq,
            request.ends,
            budgets,
            selector=request.selector,
            verifier=request.verifier,
            policy=request.policy if policy is None else policy,
            costs=request.costs,
            footprint=draft_footprint(member),
            selectors=request.selectors,
            draft_loop_graph=request.draft_loop_graph,
        )

    def _budgets(self, request: DraftRequest, index: int) -> list[int]:
        return [self._limit(index, budget - done) if active else 0
                for budget, done, active in zip(request.budgets, request.lengths, request.active)]

    def _finish(self, request: DraftRequest, tree: MergedTree) -> Draft:
        request.lengths = list(tree.draft.lengths)
        self._pending = tree if tree.draft.width else None
        return tree.draft

    def propose(self, request: DraftRequest) -> Draft:
        drafts = []
        for i, member in enumerate(self.members):
            budgets = self._budgets(request, i)
            drafts.append(member.propose(self._sub_request(request, member, budgets)) if any(budgets)
                          else Draft.empty(request.batch_size, request.device))
        verifier = request.verifier
        tree = merge_drafts(drafts, dedupe=not verifier.needs_draft_probs, keep_probs=verifier.needs_draft_probs,
                            vocab_size=self.vocab_size)
        return self._finish(request, tree)


class ForcedRequest(DraftRequest):
    """A draft request whose first tokens are given: the drafter steps over ``forced`` tokens as its inputs.

    Row ``b``'s first ``forced_lengths[b]`` drafts are replaced by ``forced[b]`` after every step (teacher forcing),
    and the drafter's own choice at every step is kept in :attr:`predictions`. Steps past the forced tokens draft
    freely. Rows choose greedily through the verifier as in any request.

    Args:
        forced: The given tokens ``[B, T]``.
        forced_lengths: Number of given tokens of every row.
        *args, **kwargs: As for :class:`~economize.decoding.DraftRequest`.
    """

    def __init__(
        self,
        forced: torch.Tensor,
        forced_lengths: Sequence[int],
        seq: torch.Tensor,
        ends: Sequence[int],
        budgets: Sequence[int],
        *,
        selector: TokenSelector,
        verifier: Verifier,
        policy: DraftPolicy,
        costs: Sequence[CostLog],
        footprint: LayerFootprint | None = None,
        layer_span: tuple[int, int] | None = None,
        selectors: Sequence[TokenSelector] | None = None,
        record_confidence: bool = False,
        record_distributions: bool = False,
        draft_loop_graph: bool = False,
    ) -> None:
        super().__init__(seq, ends, budgets, selector=selector, verifier=verifier, policy=policy, costs=costs,
                         footprint=footprint, layer_span=layer_span, selectors=selectors,
                         record_confidence=record_confidence, record_distributions=record_distributions,
                         draft_loop_graph=draft_loop_graph)
        self.forced = forced.to(self.device)
        self.forced_lengths = list(forced_lengths)
        width = max(max(self.budgets, default=0), 1)
        #: The drafter's own choice at every step ``[B, max budget]`` (``-1`` where it did not step).
        self.predictions = torch.full((self.batch_size, width), -1, dtype=torch.long, device=self.device)

    def graphed_chain_steps(self) -> int | None:
        return None  # every step's input may be a given token, so the drafter must step through extend

    def extend_tokens(self, tokens: torch.Tensor, confidence: torch.Tensor | None = None) -> None:
        raise ConfigError("the secondary member of a cascade must draft step by step (extend), not in blocks")

    def extend(self, logits: torch.Tensor) -> None:
        before = list(self.lengths)
        super().extend(logits)
        rows = [b for b in range(self.batch_size) if self.lengths[b] > before[b]]
        if not rows:
            return
        index = device_tensor(rows, self.device)
        steps = device_tensor([self.lengths[b] - 1 for b in rows], self.device)
        positions = self._positions.index_select(0, index)
        chosen = self.seq[index, positions]
        self.predictions[index, steps] = chosen.long()
        forced = [i for i, b in enumerate(rows) if self.lengths[b] <= self.forced_lengths[b]]
        if forced:
            pick = device_tensor(forced, self.device)
            given = self.forced[index[pick], steps[pick]]
            self.seq[index[pick], positions[pick]] = given.to(self.seq.dtype)


class CascadeDrafter(UnionDrafter):
    """The primary member drafts a chain; the secondary steps over the primary's tokens and branches where it
    disagrees (see :class:`~economize.techniques.union.UnionConfig`, ``mode="cascade"``).

    The tree of a row is the primary's chain, plus a sibling with the secondary's token beside every primary token
    the secondary would have chosen differently (the earliest ``max_siblings``), plus the secondary's ``extend``
    further tokens below the primary's last one. A primary node counts as the secondary's when the secondary chose
    the same token there.

    Raises:
        ConfigError: (on :meth:`propose`) the verifier needs draft distributions, or the primary drafts a tree.
    """

    def __init__(self, members: Sequence[Drafter], max_drafts: Sequence[int | None], vocab_size: int,
                 stats: UnionStats, max_siblings: int, extend: int) -> None:
        if len(members) != 2:
            raise ConfigError(f"a cascade has exactly two members, got {len(members)}")
        super().__init__(members, max_drafts, vocab_size, stats)
        self.max_siblings, self.extend = max_siblings, extend

    def draft_capacity(self, cycle_length: int) -> int:
        primary = self._limit(0, self.members[0].draft_capacity(cycle_length))
        return primary + self.max_siblings + self.extend

    def propose(self, request: DraftRequest) -> Draft:
        verifier = request.verifier
        if verifier.needs_draft_probs:
            raise ConfigError(f"a cascade drafts the secondary's tokens along the primary's, which has no draft "
                              f"distribution for verifier {type(verifier).__name__}; use the greedy verifier or "
                              "mode='union'")
        primary, secondary = self.members
        budgets = self._budgets(request, 0)
        first = (primary.propose(self._sub_request(request, primary, budgets)) if any(budgets)
                 else Draft.empty(request.batch_size, request.device))
        if first.is_tree:
            raise ConfigError(f"the primary member of a cascade must draft chains; {type(primary).__name__} drafted "
                              "a tree")
        lengths = list(first.lengths)
        forced = first.tokens.clamp_min(0) if first.width else torch.zeros(request.batch_size, 1, dtype=torch.long,
                                                                             device=request.device)
        follow = [n + self.extend if active and (n or self.extend) else 0
                  for n, active in zip(lengths, request.active)]
        predictions: list[list[int]] = [[] for _ in lengths]
        if any(follow):
            sub = ForcedRequest(forced, lengths, request.seq, request.ends, follow, selector=request.selector,
                                verifier=verifier, policy=FixedLength(request.policy.config), costs=request.costs,
                                footprint=draft_footprint(secondary), selectors=request.selectors)
            secondary.propose(sub)
            predictions = sub.predictions.tolist()
        tokens = first.tokens.tolist() if first.width else [[] for _ in lengths]
        rows = []
        for b, n in enumerate(lengths):
            row = TreeBuilder(dedupe=True)
            mine = predictions[b]
            chain: list[int] = []
            siblings = 0
            for j in range(n):
                token = tokens[b][j]
                parent = chain[-1] if chain else -1
                node = row.add(parent, token, 0, j)
                guess = mine[j] if j < len(mine) else -1
                if guess == token:
                    row.masks[node] |= 2
                elif guess >= 0 and siblings < self.max_siblings:
                    row.add(parent, guess, 1)
                    siblings += 1
                chain.append(node)
            parent = chain[-1] if chain else -1
            for j in range(n, min(n + self.extend, len(mine))):
                if mine[j] < 0:
                    break
                parent = row.add(parent, mine[j], 1)
            rows.append(row)
        return self._finish(request, build_tree(rows, request.device))
