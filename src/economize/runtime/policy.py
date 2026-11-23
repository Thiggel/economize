"""Per-layer decisions for dynamic-depth techniques (layer skipping, early exit, token routing).

A :class:`LayerPolicy` is consulted by :meth:`economize.runtime.LayerRunner.run` before every layer. It returns a
:class:`LayerAction` for the whole batch, a :class:`Route` naming the tokens that run the layer, or a boolean
``[B, S]`` tensor (converted to a :class:`Route`).
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, TypeAlias

import torch

if TYPE_CHECKING:
    from .context import LayerContext

__all__ = ["STEP_FILL", "STEP_RUN", "STEP_SKIP", "Decision", "LayerAction", "LayerPolicy", "Route", "StepAction",
           "StepOps"]


class LayerAction(enum.Enum):
    """What happens to every token at one layer."""

    #: Run the layer.
    RUN = "run"
    #: Pass the hidden state through unchanged and write no K/V (later queries see stale slots at this layer).
    SKIP = "skip"
    #: Pass the hidden state through, but write the K/V the layer would have produced for it.
    SKIP_FILL_KV = "skip_fill_kv"


@dataclass(frozen=True, eq=False)
class Route:
    """The tokens of each row that run a layer; the others skip it with their K/V filled.

    Attributes:
        indices: Token columns ``[B, K]``. Within a row they must be distinct (padding entries included).
        valid: Which entries are real ``[B, K]``; rows routing fewer than ``K`` tokens pad with ``False``.
    """

    indices: torch.Tensor
    valid: torch.Tensor

    @classmethod
    def from_mask(cls, mask: torch.Tensor) -> Route:
        """Route the tokens where ``mask`` ``[B, S]`` is true; ``K`` is the largest count of any row."""
        count = int(mask.sum(-1).max())
        order = torch.sort(mask.to(torch.int8), dim=-1, descending=True, stable=True).indices[:, :count]
        return cls(order, mask.gather(1, order))


#: What :meth:`LayerPolicy.decide` may return.
Decision: TypeAlias = LayerAction | Route | torch.Tensor


@dataclass(frozen=True)
class StepAction:
    """What a one-token decode step does at one layer (see :meth:`LayerPolicy.planned`).

    The step engine (:class:`economize.decoding.DepthSteps`) records one CUDA graph per sequence of
    ``(layer, action.name)`` pairs, so a name must determine the computation :meth:`LayerPolicy.step_op` records.

    Attributes:
        name: Identifies the computation (part of the graph key).
        runs: Whether the layer is executed (counted as executed).
        fills_kv: Whether only the layer's K/V are written (counted as filled).
        emits_signal: Whether :meth:`LayerPolicy.step_op` returns a scalar that the host reads before it plans the
            next layer (CALM's confidence); the engine ends a graph after such a layer.
    """

    name: str
    runs: bool
    fills_kv: bool = False
    emits_signal: bool = False


#: Run the layer unchanged.
STEP_RUN = StepAction("run", runs=True)
#: Skip the layer: no K/V, the state passes through.
STEP_SKIP = StepAction("skip", runs=False)
#: Skip the layer but write its K/V from the current state.
STEP_FILL = StepAction("fill", runs=False, fills_kv=True)


class StepOps(Protocol):
    """The layer operations the step engine hands to :meth:`LayerPolicy.step_op` (the state is on the layer's
    device)."""

    def run(self, index: int, hidden: torch.Tensor) -> torch.Tensor:
        """Layer ``index`` applied to ``hidden`` (its K/V written)."""
        ...

    def fill(self, index: int, hidden: torch.Tensor) -> None:
        """Write the K/V of layer ``index`` computed from ``hidden``, without the rest of the layer."""
        ...


class LayerPolicy(ABC):
    """Chooses, before each layer, which tokens run it."""

    @abstractmethod
    def decide(self, index: int, hidden: torch.Tensor, ctx: LayerContext) -> Decision:
        """Decision for layer ``index`` (1-indexed) given its input ``hidden`` ``[B, S, d]``."""

    def after(self, index: int, hidden: torch.Tensor, ctx: LayerContext) -> torch.Tensor | None:
        """Called with the output ``hidden`` of layer ``index``.

        Policies that track state (exit confidence) read it; a policy that returns a tensor replaces the hidden
        state passed on to the next layer (e.g. to freeze the rows that exited). The default returns ``None``.
        """
        return None

    def reorder(self, rows: torch.Tensor) -> None:
        """Keep only the batch rows ``rows`` (a LongTensor of row indices, in their new order) of any per-row state.

        The decoding loop calls it whenever rows leave the batch after the prefill, so that per-row state kept
        across passes (e.g. where each row's prompt ends) stays aligned with the rows of later passes. The default
        does nothing.
        """

    # ---- the step interface ------------------------------------------------------------------------------------
    # One-token decode steps of a single row can leave skipped layers out of the computation under CUDA graphs
    # (:class:`economize.decoding.DepthSteps`). Per step the engine calls :meth:`begin_step`, then per layer
    # :meth:`planned`; ``None`` makes it record :meth:`probe` in a graph, read its values on the host (one
    # synchronization, however many values the probe returns) and ask :meth:`after_probe`. Every layer's action is
    # recorded with :meth:`step_op`; an action with ``emits_signal`` ends the graph and the host passes the signal
    # to :meth:`after_op`. :meth:`end_step` receives the step's counts.
    # Host-side decisions use Python numbers, so nothing but the probes and signals synchronizes. A policy that
    # implements the interface sets ``supports_steps``; its steps must give what :meth:`decide` and :meth:`after`
    # give for a pass of one token.

    #: Whether the policy implements the step interface.
    supports_steps: ClassVar[bool] = False

    def reset(self) -> None:
        """Forget the previous generation (a decoder that records graphs keeps its policy across generations)."""

    def begin_step(self, position: int) -> None:
        """Start the decode step of the token at ``position`` (its absolute position in the row)."""

    def planned(self, index: int) -> StepAction | None:
        """The action at layer ``index``, or ``None`` when it depends on :meth:`probe` (default: run)."""
        return STEP_RUN

    def probe(self, index: int, hidden: torch.Tensor) -> torch.Tensor:
        """A 1-D float tensor computed from the input ``hidden`` of layer ``index`` (recorded in a graph) that
        :meth:`after_probe` decides on. It may hold several values (e.g. the scores that decide a group of layers),
        which the host reads with one synchronization."""
        raise NotImplementedError(f"{type(self).__name__} plans every layer without a probe")

    def after_probe(self, index: int, values: tuple[float, ...]) -> StepAction:
        """The action at layer ``index`` given the values of :meth:`probe`."""
        raise NotImplementedError(f"{type(self).__name__} plans every layer without a probe")

    def step_op(self, index: int, action: StepAction, hidden: torch.Tensor,
                ops: StepOps) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The state after layer ``index`` under ``action`` (recorded in a graph, so without host synchronization)
        and the one-element signal of an action with ``emits_signal`` (else ``None``). Default: run the layer, fill
        its K/V, or pass the state through."""
        if action.runs:
            return ops.run(index, hidden), None
        if action.fills_kv:
            ops.fill(index, hidden)
        return hidden, None

    def after_op(self, index: int, action: StepAction, signal: float) -> None:
        """Receive the signal of an action with ``emits_signal`` at layer ``index``."""

    def end_step(self, counts: torch.Tensor) -> None:
        """Receive the step's counts, a host ``[L + 1, 3]`` long tensor whose row ``l`` holds whether the token
        entered layer ``l``, executed it, and had only its K/V filled."""
