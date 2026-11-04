"""Mechanics shared by the drafters: graphed steps, bounded proposals, per-row state, observe-time costs, and the
:class:`DraftModules` capability of drafter techniques."""

from __future__ import annotations

import inspect
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
from torch import nn

from ...decoding.cost import CostKind, LayerFootprint
from ...decoding.protocols import Draft, Drafter

if TYPE_CHECKING:
    from ...decoding.cost import CostLog
    from ...decoding.request import DraftRequest
    from ...runtime import ExactPass

__all__ = [
    "BoundedDrafter",
    "CommittedColumns",
    "DraftModules",
    "GraphedStep",
    "ObserveCosts",
    "StepFn",
    "previous_states",
    "take_rows",
]

#: A draft step: ``(tokens [B], positions [B], valid [B] | None) -> logits [B, V]``.
StepFn = Callable[[torch.Tensor, torch.Tensor, "torch.Tensor | None"], torch.Tensor]


@runtime_checkable
class DraftModules(Protocol):
    """A drafter technique that names its own modules, for techniques that transform the draft path only
    (``draft_quant``). Without it they fall back to the technique's whole module tree."""

    def draft_modules(self) -> dict[str, nn.Module]:
        """The drafter's own modules by qualified name (no shared base-model layers)."""
        ...


def take_rows(tensor: torch.Tensor | None, rows: torch.Tensor) -> torch.Tensor | None:
    """Rows ``rows`` of a per-row state ``[B, ...]`` (``None`` stays ``None``), for :meth:`Drafter.reorder`."""
    return None if tensor is None else tensor.index_select(0, rows.to(tensor.device))


class GraphedStep:
    """A draft step that runs eagerly, or from a CUDA graph once :meth:`enable` was called.

    A graph is recorded per input signature, so the all-valid case (``valid is None``) runs with a cached all-true
    mask and shares the graph of the masked case.

    A bound method (the usual ``GraphedStep(self._step)`` of the drafter that owns the step) is held weakly. A
    strong reference would make every drafter a reference cycle, and an eager decoder builds a new drafter per call,
    so each call's KV cache would stay allocated until the cyclic garbage collector happened to run.

    Args:
        step: The eager step.
    """

    def __init__(self, step: StepFn) -> None:
        self._step_ref: Callable[[], StepFn | None] = (
            weakref.WeakMethod(step) if inspect.ismethod(step) else lambda: step
        )
        self._graph: Callable[..., torch.Tensor] | None = None
        self._wrap: Callable[[Callable[..., Any]], Callable[..., Any]] | None = None
        self._all_valid: dict[tuple[int, torch.device], torch.Tensor] = {}

    @property
    def _eager(self) -> StepFn:
        step = self._step_ref()
        if step is None:
            raise RuntimeError("the object whose method this GraphedStep runs was deleted")
        return step

    @property
    def enabled(self) -> bool:
        """Whether the step runs from CUDA graphs."""
        return self._graph is not None

    def enable(self, wrap: Callable[[Callable[..., Any]], Callable[..., Any]]) -> None:
        """Route the step through ``wrap(step)`` (see :meth:`~economize.decoding.Drafter.enable_graphs`)."""
        self._graph = wrap(self._eager)
        self._wrap = wrap

    def graph(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """``fn`` (a function of tensors next to the step, e.g. several steps in a row) graphed by the same ``wrap``
        as the step, so the decoder resets it with the step's graphs; needs :meth:`enable` first."""
        if self._wrap is None:
            raise RuntimeError("GraphedStep.graph needs enable() first")
        return self._wrap(fn)

    def all_valid(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """The cached all-true row mask ``[batch_size]`` the graphs read when every row drafts."""
        key = (batch_size, device)
        if key not in self._all_valid:
            self._all_valid[key] = torch.ones(batch_size, dtype=torch.bool, device=device)
        return self._all_valid[key]

    def reset(self) -> None:
        """Drop the recorded graphs (after the buffers the step reads or writes were reallocated)."""
        reset = getattr(self._graph, "reset", None)
        if reset is not None:
            reset()

    def __call__(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        if self._graph is None:
            return self._eager(tokens, positions, valid)
        if valid is None:
            valid = self.all_valid(tokens.shape[0], tokens.device)
        return self._graph(tokens, positions, valid)


class BoundedDrafter(Drafter):
    """A drafter whose steps attend to a cache of its own, read only up to the slots the proposal can reach.

    :meth:`propose` sets :attr:`step_limit` to the end of the furthest active row's budget (at most
    :meth:`draft_slots`) for the duration of the proposal, so a step reads ``step_limit`` slots instead of the
    whole cache. Outside a proposal :attr:`step_limit` is ``None``.
    """

    step_limit: int | None = None

    def draft_slots(self) -> int | None:
        """Slots of the drafter's own cache, ``None`` when the drafter keeps none (the proposal is then unbounded)."""
        return None

    def propose(self, request: DraftRequest) -> Draft:
        slots = self.draft_slots()
        if slots is None:
            return super().propose(request)
        ends = [end + budget for end, budget, active in zip(request.ends, request.budgets, request.active) if active]
        self.step_limit = min(slots, max(ends, default=1))
        try:
            return super().propose(request)
        finally:
            self.step_limit = None


@dataclass(frozen=True)
class CommittedColumns:
    """Which columns of an exact pass hold tokens, and which of them every row committed.

    Attributes:
        valid: Columns holding a token, ``[B, S]`` bool.
        committed: Valid columns up to the row's last accepted one, ``[B, S]`` bool.
        positions: Position of every column, ``[B, S]``.
        columns: Column of every row's last accepted token, ``[B]``.
    """

    valid: torch.Tensor
    committed: torch.Tensor
    positions: torch.Tensor
    columns: torch.Tensor

    @classmethod
    def of(cls, exact: ExactPass, columns: torch.Tensor, device: torch.device | str, owner: str) -> CommittedColumns:
        """The columns of ``exact`` (whose context must be set; ``owner`` names the drafter in the error)."""
        ctx = exact.context
        if ctx is None:
            raise RuntimeError(f"{owner}.observe needs the exact pass's context")
        batch, length = exact.final.shape[:2]
        positions = ctx.positions.to(device).expand(batch, length)
        valid = (torch.ones(batch, length, dtype=torch.bool, device=device) if ctx.valid is None
                 else ctx.valid.to(device))
        columns = columns.to(device)
        committed = valid & (torch.arange(length, device=device)[None, :] <= columns[:, None])
        return cls(valid, committed, positions, columns)

    def last(self, states: torch.Tensor) -> torch.Tensor:
        """``states`` ``[B, S, ...]`` at every row's last accepted column, ``[B, ...]``."""
        return states[torch.arange(states.shape[0], device=states.device), self.columns.to(states.device)]


def previous_states(states: torch.Tensor, valid: torch.Tensor, carried: torch.Tensor) -> torch.Tensor:
    """The state of the token before every column, ``[B, S, d]``.

    Column ``j`` reads ``states[:, j - 1]`` where column ``j - 1`` holds a token of this pass, else ``carried``
    ``[B, d]`` (the state of the row's last token of an earlier pass). EAGLE-3 and MTP modules pair token ``p``
    with the state of token ``p - 1``.
    """
    before = torch.cat([valid.new_zeros(valid.shape[0], 1), valid[:, :-1]], 1)
    shifted = torch.cat([torch.zeros_like(states[:, :1]), states[:, :-1]], 1)
    return torch.where(before[..., None], shifted, carried[:, None].to(states.dtype))


class ObserveCosts:
    """The passes a drafter's own network makes in :meth:`~economize.decoding.Drafter.observe`, for the cost log.

    :meth:`observe` gets no cost log, so the drafter adds every such pass here (tokens per row and the position of
    the row's first one, as device tensors: no host synchronization) and :meth:`flush` records them as
    :data:`~economize.decoding.CostKind.DRAFT_OBSERVE` events at the start of the next proposal, before its draft
    steps. Passes of rows that finish before drafting again are dropped with them by :meth:`reorder`.

    Args:
        footprint: The layers one pass runs (usually the drafter's :attr:`~economize.decoding.Drafter.footprint`).
    """

    def __init__(self, footprint: LayerFootprint) -> None:
        self.footprint = footprint
        self._pending: list[tuple[torch.Tensor, torch.Tensor, LayerFootprint]] = []

    def reset(self) -> None:
        """Forget every pending pass (a new generation)."""
        self._pending = []

    def add(self, tokens: torch.Tensor, starts: torch.Tensor, footprint: LayerFootprint | None = None) -> None:
        """One pass over ``tokens`` ``[B]`` tokens per row, starting at position ``starts`` ``[B]``, through
        ``footprint`` (default: :attr:`footprint`)."""
        self._pending.append((tokens.detach(), starts.detach(), footprint or self.footprint))

    def add_columns(self, columns: torch.Tensor, positions: torch.Tensor) -> None:
        """One pass over the columns ``columns`` ``[B, S]`` (bool) at ``positions`` ``[B, S]``."""
        width = columns.shape[1]
        first = torch.where(columns, torch.arange(width, device=columns.device)[None], width).amin(1)
        start = positions.gather(1, first.clamp(max=width - 1)[:, None])[:, 0]
        self.add(columns.sum(1), start)

    def reorder(self, rows: torch.Tensor) -> None:
        self._pending = [(take_rows(t, rows), take_rows(s, rows), f) for t, s, f in self._pending]  # type: ignore[misc]

    def flush(self, costs: Sequence[CostLog]) -> None:
        """Record every pending pass in the cost log of its row and forget it."""
        for tokens, starts, footprint in self._pending:
            for cost, n, start in zip(costs, tokens.tolist(), starts.tolist()):
                if n > 0:
                    cost.record_footprint(CostKind.DRAFT_OBSERVE, n, start, footprint)
        self._pending = []
