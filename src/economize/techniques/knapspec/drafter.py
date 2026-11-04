"""The drafter of KnapSpec: the self-drafter of early exit with a skip set that it re-searches while decoding."""

from __future__ import annotations

import math
from collections import OrderedDict, deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from ...decoding.protocols import draft_footprint
from .._drafting import CommittedColumns, GraphedStep
from ..early_exit.drafter import SelfDrafter
from .latency import measure_latency
from .search import SearchStats, SearchTokens

if TYPE_CHECKING:
    from ...decoding import Draft, DraftRequest
    from ...runtime import ExactPass, KVCache, LayerRunner
    from .technique import KnapSpecTechnique

__all__ = ["SEARCH_COST", "KnapSpecDrafter"]

#: Cost kind of a search in the cost log: the candidate sublayer calls, counted as calls of all ``L`` layers.
SEARCH_COST = "knapspec_search"


class KnapSpecDrafter(SelfDrafter):
    """A :class:`~economize.techniques.early_exit.SelfDrafter` whose skip set and draft length come from KnapSpec's
    search, repeated every ``interval`` decoded tokens.

    The drafter keeps the embeddings (and Gemma-4's per-layer inputs) of the committed tokens of its last
    ``history`` exact passes. The first proposal of a generation measures the sublayer latencies when the
    configuration gives none (once per technique: later generations reuse them) and searches on the prompt's tail.
    A search runs inside :meth:`propose`, so it counts to the draft phase's wall clock, and it is logged as one
    :data:`SEARCH_COST` event per row. A draft step reads :attr:`KnapSpecTechnique.footprint`, which follows the
    current set; the step proposes ``plan.draft_length`` tokens per cycle at most (the loop's ``cycle_length`` and
    ``min_confidence`` may stop it earlier).

    With CUDA graphs every skip set gets its own graphed step (a graph records the kernels of one set); the
    ``max_graphs`` most recently used are kept.

    Attributes:
        last_search: What the latest search computed (``None`` before the first).
        searches: Number of searches in this generation.
    """

    technique: KnapSpecTechnique

    def __init__(self, technique: KnapSpecTechnique, runner: LayerRunner) -> None:
        super().__init__(technique, runner)
        self._history: deque[SearchTokens] = deque(maxlen=technique.config.history)
        self._wrap: Callable[[Callable[..., Any]], Callable[..., Any]] | None = None
        self._graphs: OrderedDict[tuple[tuple[int, ...], tuple[int, ...]], GraphedStep] = OrderedDict()
        self._searched_at: int | None = None
        self.last_search: SearchStats | None = None
        self.searches = 0

    @property
    def observed_layers(self) -> tuple[int, ...]:
        return (0,)

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        super().begin(runner, cache, batch_size)
        self._history.clear()
        self._searched_at = None
        self.searches = 0
        self.technique.plan = None

    def enable_graphs(self, wrap: Callable[[Callable[..., Any]], Callable[..., Any]]) -> bool:
        self._wrap = wrap
        self._step_fn.enable(wrap)  # the step before the first search (unused: a proposal searches first)
        return True

    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None:
        hidden = exact.hidden[0]
        device = hidden.device
        committed = CommittedColumns.of(exact, columns, device, type(self).__name__)
        length = hidden.shape[1]
        keep = min(length, self.technique.config.prompt_tokens)
        # the last ``keep`` committed columns of every row (committed columns score by their index, the others 0)
        score = committed.committed * torch.arange(1, length + 1, device=device)
        index = score.topk(keep, dim=1).indices
        valid = committed.committed.gather(1, index)
        positions = torch.where(valid, committed.positions.gather(1, index), 0)
        inputs = exact.context.token_inputs if exact.context is not None else {}
        self._history.append(SearchTokens(_gather(hidden, index), positions, valid,
                                          {name: _gather(t, index) for name, t in inputs.items()}))

    def reorder(self, rows: torch.Tensor) -> None:
        for entry in self._history:
            index = rows.to(entry.hidden.device)
            entry.hidden, entry.positions, entry.valid = entry.hidden[index], entry.positions[index], entry.valid[index]
            entry.token_inputs = {name: t[index] for name, t in entry.token_inputs.items()}

    def propose(self, request: DraftRequest) -> Draft:
        end = max(request.ends)
        interval = self.technique.config.interval
        if self.technique.plan is None or self._searched_at is None or end - self._searched_at >= interval:
            self._search(request)
            self._searched_at = end
        plan = self.technique.plan
        assert plan is not None
        while request.drafting and request.index < plan.draft_length:
            request.extend(self.step(request.tokens, request.positions, request.valid))
        return request.draft()

    # ---- the search ----------------------------------------------------------------------------------------------
    def _search(self, request: DraftRequest) -> None:
        assert self._cache is not None
        technique, config = self.technique, self.technique.config
        if technique.latency is None:
            technique.latency = measure_latency(self.runner, self._cache, technique.skipper,
                                                config.calibration_lengths, config.calibration_repeats)
        entries = list(self._history)
        tokens = SearchTokens(
            torch.cat([e.hidden for e in entries], 1),
            torch.cat([e.positions for e in entries], 1),
            torch.cat([e.valid for e in entries], 1),
            {name: torch.cat([e.token_inputs[name] for e in entries], 1) for name in entries[-1].token_inputs},
        )
        plan, stats = technique.search(self.runner, self._cache, tokens, technique.latency)
        technique.plan = plan
        self._select_graph(plan.key)
        self.last_search, self.searches = stats, self.searches + 1
        L = technique.adapter.num_layers
        active = [b for b, a in enumerate(request.active) if a]
        per_row = math.ceil(stats.layer_tokens / max(len(request.ends), 1) / L)
        for b in active:
            request.costs[b].record(SEARCH_COST, per_row, stats.position, 1, L)
        request.footprint = draft_footprint(self)

    def _select_graph(self, key: tuple[tuple[int, ...], tuple[int, ...]]) -> None:
        """Route the draft steps through the graphed step of skip set ``key`` (recorded on first use)."""
        if self._wrap is None:
            return
        if key in self._graphs:
            self._graphs.move_to_end(key)
        else:
            step = GraphedStep(self._step)
            step.enable(self._wrap)
            self._graphs[key] = step
            while len(self._graphs) > self.technique.config.max_graphs:
                _, old = self._graphs.popitem(last=False)
                device = torch.device(self.technique.adapter.device)
                if device.type == "cuda":  # no replay of the evicted graphs may still be in flight
                    torch.cuda.synchronize(device)
                old.reset()
        self._step_fn = self._graphs[key]


def _gather(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``tensor`` ``[B, S, ...]`` at the columns ``index`` ``[B, K]``."""
    view = index.to(tensor.device).view(*index.shape, *([1] * (tensor.dim() - 2)))
    return tensor.gather(1, view.expand(-1, -1, *tensor.shape[2:]))
