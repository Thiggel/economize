"""The request a drafter serves in one cycle: per-row budgets, the token buffer and the draft bookkeeping."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..errors import ConfigError
from ..runtime.transfer import device_tensor
from .cost import CostKind, CostLog, LayerFootprint
from .protocols import Draft, DraftPolicy, TokenSelector, Verifier
from .selectors import GreedySelector

__all__ = ["DraftRequest", "group_rows"]


def group_rows(selectors: Sequence[TokenSelector]) -> list[tuple[TokenSelector, list[int]]]:
    """Rows that share a selector object, in order of first appearance: ``[(selector, rows), ...]``."""
    groups: dict[int, tuple[TokenSelector, list[int]]] = {}
    for b, selector in enumerate(selectors):
        groups.setdefault(id(selector), (selector, []))[1].append(b)
    return list(groups.values())


class DraftRequest:
    """One cycle of drafting for a batch of rows.

    The loop builds a request after every exact pass and hands it to :meth:`Drafter.propose`. A chain drafter
    reads :attr:`tokens`, :attr:`positions` and :attr:`valid`, runs one step and passes the step's logits to
    :meth:`extend`, until :attr:`drafting` is false. :meth:`extend` records the cost of the step, applies the draft
    policy, chooses the tokens with the selector (through the verifier, which may keep the draft distribution) and
    writes them into the sequence buffer at their positions.

    Args:
        seq: All tokens so far ``[B, capacity + 1]``, indexed by position; the last column is a scratch column for
            rows that are not drafting. Drafted tokens are written into it.
        ends: Number of committed tokens of every row; row ``b`` drafts from position ``ends[b]`` on.
        budgets: Most tokens every row may draft in this cycle.
        selector: The token selector of the generation.
        verifier: The verifier of the generation (it chooses draft tokens through :meth:`Verifier.draft`).
        policy: The draft-length policy.
        costs: The cost log of every row.
        footprint: The layers one draft step runs, for the cost log (usually
            :func:`~economize.decoding.protocols.draft_footprint` of the drafter).
        layer_span: First and last decoder layer one draft step runs; the older spelling of ``footprint`` for a
            contiguous range of the decoded model's layers. Give exactly one of the two.
        selectors: One selector per row when the rows choose tokens differently (overrides ``selector``).
        record_confidence: Keep the softmax maximum of every step's logits for the drafted tokens
            (:attr:`Draft.confidence`), on the device.
        record_distributions: Keep every step's logits for the drafted tokens (:attr:`Draft.logits`), on the
            device.
        draft_loop_graph: The drafter may run a whole chain proposal as one CUDA graph
            (:meth:`graphed_chain_steps`, ``DecodeConfig.draft_loop_graph``).
    """

    def __init__(
        self,
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
        if not (len(ends) == len(budgets) == len(costs) == seq.shape[0]):
            raise ConfigError("seq, ends, budgets and costs must describe the same rows")
        if (footprint is None) == (layer_span is None):
            raise ConfigError("give exactly one of footprint and layer_span")
        if selectors is not None and len(selectors) != seq.shape[0]:
            raise ConfigError("selectors must hold one selector per row")
        self.seq, self.selector, self.verifier, self.policy = seq, selector, verifier, policy
        self.selectors = list(selectors) if selectors is not None else [selector] * seq.shape[0]
        self.footprint = footprint if footprint is not None else LayerFootprint(*layer_span)  # type: ignore[misc]
        self.costs = list(costs)
        self.ends, self.budgets = list(ends), list(budgets)
        self.index = 0
        self.lengths = [0] * len(self.ends)
        self.stopped = [False] * len(self.ends)
        device = seq.device
        self._ends = device_tensor(self.ends, device)
        self._positions = self._ends - 1
        self._cpu_positions = [end - 1 for end in self.ends]
        self._valid: torch.Tensor | None = None
        self._valid_key: tuple[bool, ...] | None = None
        self._probs: torch.Tensor | None = None
        self.record_confidence = record_confidence
        self._confidence: torch.Tensor | None = None
        self.record_distributions = record_distributions
        self._logits: torch.Tensor | None = None
        self._groups = group_rows(self.selectors)
        self.draft_loop_graph = draft_loop_graph

    # ---- what the drafter reads ------------------------------------------------------------------------------
    @property
    def batch_size(self) -> int:
        return len(self.ends)

    @property
    def device(self) -> torch.device:
        return self.seq.device

    @property
    def scratch(self) -> int:
        """Column of :attr:`seq` that absorbs the writes of rows that are not drafting."""
        return self.seq.shape[1] - 1

    @property
    def layer_span(self) -> tuple[int, int]:
        """First and last layer of :attr:`footprint`."""
        return self.footprint.first_layer, self.footprint.last_layer

    @property
    def active(self) -> list[bool]:
        """Which rows still draft in this cycle."""
        return [not s and n < budget for s, n, budget in zip(self.stopped, self.lengths, self.budgets)]

    @property
    def drafting(self) -> bool:
        """Some row still drafts."""
        return any(self.active)

    @property
    def tokens(self) -> torch.Tensor:
        """The token every row processes next (its last token), ``[B]``."""
        return self.seq.gather(1, self._positions[:, None])[:, 0]

    @property
    def positions(self) -> torch.Tensor:
        """Position of :attr:`tokens`, ``[B]``."""
        return self._positions

    @property
    def valid(self) -> torch.Tensor | None:
        """Rows that draft in this step ``[B]`` (bool), ``None`` when all do."""
        active = tuple(self.active)
        if all(active):
            return None
        if active != self._valid_key:
            self._valid = device_tensor(active, self.device, torch.bool)
            self._valid_key = active
        return self._valid

    # ---- what the drafter calls ------------------------------------------------------------------------------
    def extend(self, logits: torch.Tensor) -> None:
        """Turn one draft step's float logits ``[B, V]`` (the next token of every row) into draft tokens."""
        active = self.active
        rows = [b for b, a in enumerate(active) if a]
        for b in rows:
            self.costs[b].record_footprint(CostKind.DRAFT, 1, self._cpu_positions[b], self.footprint, step=self.index)
        if self.policy.may_stop:
            stop = self.policy.stops(self.index, logits)
            if stop is not None:
                flags = stop.tolist()
                for b in rows:
                    if flags[b]:
                        self.costs[b].mark_last_wasted()
                        self.stopped[b] = True
                rows = [b for b in rows if not self.stopped[b]]
        self.index += 1
        if not rows:
            return
        if self.record_confidence:
            self._store_confidence(logits, rows)
        if self.record_distributions:
            self._store_logits(logits, rows)
        tokens = self._select(logits, rows)
        keep = None
        if len(rows) < self.batch_size:
            mask = [False] * self.batch_size
            for b in rows:
                mask[b] = True
            keep = device_tensor(mask, self.device, torch.bool)
        target = self._positions + 1
        if keep is not None:
            target = torch.where(keep, target, torch.full_like(target, self.scratch))
        self.seq.scatter_(1, target[:, None], tokens[:, None].to(self.seq.device, self.seq.dtype))
        self._positions = target if keep is None else torch.where(keep, target, self._positions)
        for b in rows:
            self._cpu_positions[b] += 1
            self.lengths[b] += 1

    def graphed_chain_steps(self) -> int | None:
        """The number of greedy draft steps ``k`` a drafter may run at once and hand to :meth:`extend_tokens`, or
        ``None`` when the proposal must go step by step through :meth:`extend`.

        It is ``k`` when ``draft_loop_graph`` is set, no row has drafted yet, every row drafts the same ``k > 0``
        tokens, the draft policy never stops early, every row chooses greedily through the verifier's default
        :meth:`~Verifier.draft_many` (so a step's draft is the arg-max of its logits) and no step's logits are
        kept.
        """
        if not self.draft_loop_graph or self.index or self.record_distributions or self.policy.may_stop:
            return None
        verifier = self.verifier
        if verifier.needs_draft_probs or type(verifier).draft_many is not Verifier.draft_many:
            return None
        if not all(isinstance(selector, GreedySelector) for selector in self.selectors):
            return None
        budgets = set(self.budgets)
        if len(budgets) != 1 or any(self.stopped):
            return None
        steps = budgets.pop()
        return steps if steps > 0 else None

    def extend_tokens(self, tokens: torch.Tensor, confidence: torch.Tensor | None = None) -> None:
        """Append ``k`` draft tokens ``[B, k]`` to every row at once (``k`` from :meth:`graphed_chain_steps`), with
        the softmax maximum of every step ``[B, k]`` when :attr:`record_confidence` is set.

        The cost log and the sequence buffer end up as after ``k`` calls of :meth:`extend` whose logits had these
        tokens as their arg-max.
        """
        steps = self.graphed_chain_steps() if self.draft_loop_graph else None
        if steps is None or tuple(tokens.shape) != (self.batch_size, steps):
            raise ConfigError(f"extend_tokens needs [B, k] tokens with k = graphed_chain_steps() ({steps}); "
                              f"got {tuple(tokens.shape)}")
        if self.record_confidence and confidence is None:
            raise ConfigError("record_confidence is set: extend_tokens needs the confidence of every step")
        for i in range(steps):
            for b in range(self.batch_size):
                self.costs[b].record_footprint(CostKind.DRAFT, 1, self._cpu_positions[b] + i, self.footprint,
                                               step=i)
        if self.record_confidence:
            assert confidence is not None
            self._confidence = confidence.to(self.device, torch.float32).clone()
        target = self._positions[:, None] + torch.arange(1, steps + 1, device=self.device)
        self.seq.scatter_(1, target, tokens.to(self.seq.device, self.seq.dtype))
        self._positions = self._positions + steps
        self._cpu_positions = [position + steps for position in self._cpu_positions]
        self.lengths = [steps] * self.batch_size
        self.index = steps

    def draft(self) -> Draft:
        """The chain drafts made so far."""
        width = max(self.lengths, default=0)
        if width == 0:
            return Draft.empty(self.batch_size, self.device)
        offsets = torch.arange(width, device=self.device)
        lengths = device_tensor(self.lengths, self.device)
        index = (self._ends[:, None] + offsets).clamp(max=self.scratch)
        tokens = self.seq.gather(1, index).masked_fill(offsets >= lengths[:, None], -1)
        probs = self._probs[:, :width] if self._probs is not None else None
        confidence = self._confidence[:, :width] if self._confidence is not None else None
        draft_logits = self._logits[:, :width] if self._logits is not None else None
        return Draft(tokens, list(self.lengths), None, probs, confidence, draft_logits)

    # ---- internals -------------------------------------------------------------------------------------------
    def _select(self, logits: torch.Tensor, rows: list[int]) -> torch.Tensor:
        """Draft tokens ``[B]`` for the rows ``rows`` (other rows get arbitrary tokens), group by group."""
        if len(self._groups) == 1:
            batched = self.verifier.draft_many(logits, self._groups[0][0])
            if batched is not None:
                tokens, probs = batched
                if probs is not None:
                    self._store_probs(rows, probs, rows)
                return tokens.to(self.device)
        active = set(rows)
        chosen = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        for selector, members in self._groups:
            index = device_tensor(members, logits.device)
            batched = self.verifier.draft_many(logits.index_select(0, index), selector)
            if batched is not None:
                tokens, probs = batched
                chosen[index.to(self.device)] = tokens.to(self.device)
                if probs is not None:
                    self._store_probs([b for b in members if b in active], probs,
                                      [i for i, b in enumerate(members) if b in active])
                continue
            for b in members:
                if b not in active:
                    continue
                history = self.seq[b, : self._cpu_positions[b] + 1]
                token, probs = self.verifier.draft(logits[b], selector, history)
                chosen[b] = token
                if probs is not None:
                    self._store_probs([b], probs[None], [0])
        return chosen

    def _store_confidence(self, logits: torch.Tensor, rows: list[int]) -> None:
        """Keep the top probability of ``logits`` ``[B, V]`` for the current draft of every row in ``rows``."""
        if self._confidence is None:
            width = max(self.budgets)
            self._confidence = torch.full((self.batch_size, width), float("nan"), dtype=torch.float32,
                                          device=self.device)
        top = torch.softmax(logits.float(), -1).amax(-1).to(self.device)
        steps = {self.lengths[b] for b in rows}
        if len(rows) == self.batch_size and len(steps) == 1:  # the common case: no index tensors to copy
            self._confidence[:, steps.pop()] = top
            return
        index = device_tensor(rows, self.device)
        steps_of = device_tensor([self.lengths[b] for b in rows], self.device)
        self._confidence[index, steps_of] = top.index_select(0, index)

    def _store_logits(self, logits: torch.Tensor, rows: list[int]) -> None:
        """Keep ``logits`` ``[B, V]`` as the logits of the current draft of every row in ``rows``."""
        if self._logits is None:
            self._logits = torch.zeros(self.batch_size, max(self.budgets), logits.shape[-1], dtype=torch.float32,
                                       device=self.device)
        values = logits.to(self.device, torch.float32)
        steps = {self.lengths[b] for b in rows}
        if len(rows) == self.batch_size and len(steps) == 1:
            self._logits[:, steps.pop()] = values
            return
        index = device_tensor(rows, self.device)
        steps_of = device_tensor([self.lengths[b] for b in rows], self.device)
        self._logits[index, steps_of] = values.index_select(0, index)

    def _store_probs(self, rows: list[int], probs: torch.Tensor, picks: list[int]) -> None:
        """Keep ``probs[picks[j]]`` as the distribution of row ``rows[j]``'s current draft."""
        if not rows:
            return
        if self._probs is None:
            width = max(self.budgets)
            self._probs = torch.zeros(self.batch_size, width, probs.shape[-1], dtype=torch.float32, device=self.device)
        device = self._probs.device
        index = device_tensor(rows, device)
        steps = device_tensor([self.lengths[b] for b in rows], device)
        picked = probs.to(device, torch.float32)
        if len(picks) != probs.shape[0] or picks != list(range(len(picks))):
            picked = picked.index_select(0, device_tensor(picks, device))
        self._probs[index, steps] = picked
