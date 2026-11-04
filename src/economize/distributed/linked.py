"""Drafters linked to the target: SSD workers that draft from the target's exact states.

A drafter that runs part of the target's computation (a jump drafter, an EAGLE draft layer) cannot draft well from
its own outputs alone: its draft steps continue from states that only the target computes exactly. A
:class:`LinkedSpeculativeDecoder` therefore keeps the states of the layers the drafter observes from every
verification (:class:`CapturingTargetWorker`) and lets its :class:`LinkedWorker` *pull* them, together with whatever
the drafter derives from them, for the committed positions before it drafts on.

:class:`LinkedWorker` holds what every such drafter shares: the speculation of the next chain behind the likeliest
outcomes, one tree pass per depth, and the adoption of a hit branch together with the states it was drafted from. A
new drafter subclasses it, implements its step (:meth:`LinkedWorker.begin`, :meth:`~LinkedWorker.draft`,
:meth:`~LinkedWorker._pull`, :meth:`~LinkedWorker._tree_pass`, :meth:`~LinkedWorker._move_branch`), pairs it with a
:class:`LinkedSpeculativeDecoder` subclass and registers both in :data:`economize.distributed.DRAFTER_WORKERS` under
its technique's name.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from typing import ClassVar, Generic, Protocol, TypeVar

import torch
from torch import nn

from ..decoding.cost import CostKind, CostLog
from ..errors import ConfigError
from ..runtime.context import LayerContext
from ..validation import check_positive
from .ssd import AsyncSpeculativeDecoder, SSDConfig
from .tp import is_tensor_parallel
from .workers import (
    Adoption,
    Chooser,
    Drafts,
    Speculation,
    SpeculationWorker,
    Spine,
    TargetWorker,
    TreeTemplate,
    TreeTemplates,
    outcome_candidates,
)

__all__ = [
    "CapturingTargetWorker",
    "LinkedDrafter",
    "LinkedSpeculativeDecoder",
    "LinkedWorker",
    "TreePass",
    "attached_drafting_technique",
]

#: The state one draft step continues from (per position: a dict of source states, a fused vector, ...).
StateT = TypeVar("StateT")
TechniqueT = TypeVar("TechniqueT")


class CapturingTargetWorker(TargetWorker):
    """The target of a :class:`LinkedSpeculativeDecoder`: a :class:`TargetWorker` that also keeps the states of
    ``layers`` from its last pass.

    Attributes:
        start: Position of the first token of the last pass.
        hidden: Its states ``{layer: [S, d]}``.
    """

    def __init__(self, model: nn.Module, layers: Iterable[int]) -> None:
        super().__init__(model)
        self.layers = frozenset(layers)
        self.start = 0
        self.hidden: dict[int, torch.Tensor] = {}

    def _run(self, tokens: torch.Tensor, position: int, ctx: LayerContext) -> torch.Tensor:
        self.cost.record(self.kind, tokens.numel(), position, 1, self.num_layers)
        exact = self.runner.forward(tokens.view(1, -1), ctx, capture=self.layers)
        self.start, self.hidden = position, {layer: state[0] for layer, state in exact.hidden.items()}
        return self.runner.logits(exact.final)[0]

    def states_at(self, position: int) -> dict[int, torch.Tensor] | None:
        """Exact states of the token at ``position`` if the last pass covered it."""
        column = position - self.start
        if not self.hidden or not 0 <= column < next(iter(self.hidden.values())).shape[0]:
            return None
        return {layer: state[column] for layer, state in self.hidden.items()}


def attached_drafting_technique(model: nn.Module, kind: type[TechniqueT], worker: str, what: str) -> TechniqueT:
    """The drafting technique of a drafter replica, checked for a :class:`LinkedWorker`.

    Args:
        model: The replica (``economize.load`` or ``economize.apply``) on the drafter's device.
        kind: The technique class the worker runs.
        worker: The worker's name, for the messages.
        what: The technique as the messages name it, with its article (``"a jump"``).

    Raises:
        ConfigError: ``model`` has no drafting technique of ``kind`` or is split by tensor parallelism.
    """
    technique = getattr(model, "drafting_technique", None)
    if not isinstance(technique, kind):
        raise ConfigError(f"{worker} needs a model with {what} technique attached (economize.load / apply)")
    if is_tensor_parallel(model):
        raise ConfigError(f"{worker} runs on one device; load the replica unsplit")
    return technique


class TreePass(Protocol[StateT]):
    """One depth of a speculation: the draft step on the tree nodes ``nodes`` ``[1, S]`` (``valid`` ``[1, S]``)
    holding ``tokens`` ``[S]``, each continuing from its parent's state. Returns the logits ``[S, V]`` and the nodes'
    own states."""

    def __call__(self, nodes: torch.Tensor, valid: torch.Tensor, tokens: torch.Tensor,
                 parents: list[StateT]) -> tuple[torch.Tensor, list[StateT]]: ...


class LinkedDrafter(SpeculationWorker):
    """What a :class:`LinkedSpeculativeDecoder` needs of its drafter: the layers to observe and a link.

    Attributes:
        observed_layers: Layers whose states the linked target keeps (``0``: the embeddings).
        target: The linked target, or ``None`` (the worker then drafts open loop if it can).
    """

    observed_layers: tuple[int, ...] = ()
    target: CapturingTargetWorker | None = None

    def link(self, target: CapturingTargetWorker) -> None:
        """Take exact states from ``target`` from now on (see :class:`LinkedSpeculativeDecoder`)."""
        self.target = target


class LinkedWorker(LinkedDrafter, Generic[StateT]):
    """A linked drafter whose steps continue from per-position states, with the speculation and adoption all such
    drafters share.

    The worker keeps the state of every position it may continue from in ``_states``: the spine's own, then those of
    an adopted branch, overwritten by the target's exact ones as :meth:`_pull` brings them in. A subclass supplies
    the draft step and the cache moves.

    Args:
        device: The drafter's device.
        fanout: Candidates for the appended token per outcome.
    """

    kind = CostKind.DRAFT

    def __init__(self, device: torch.device, fanout: int) -> None:
        check_positive("fanout", fanout)
        self.device = device
        self.fanout = fanout
        self.cost = CostLog()
        self.target = None
        self._templates = TreeTemplates(fanout, device)
        self._states: dict[int, StateT] = {}  # by position: the spine's, then the adopted branch's
        self._branch_states: list[list[StateT]] = []  # per branch, per depth, of the last speculation
        self._synced = 0  # positions below this hold what the worker derives from the target's exact states

    # ---- link to the target ----------------------------------------------------------------------------------------

    @abstractmethod
    def _pull(self, end: int) -> None:
        """Bring in what the worker derives from the target's exact states for positions ``_synced..end-1`` (they
        all lie in the target's last pass) and the exact state of ``end - 1``; a no-op without a linked target."""

    # ---- the draft step ----------------------------------------------------------------------------------------
    @abstractmethod
    def _tree_pass(self, template: TreeTemplate, base: int) -> TreePass[StateT]:
        """The draft step over nodes of the speculation tree ``template`` laid out from cache slot ``base``."""

    @abstractmethod
    def _move_branch(self, hit: Adoption) -> None:
        """Move the cache entries of the branch ``hit`` from its tree slots to its sequence positions."""

    # ---- SpeculationWorker ---------------------------------------------------------------------------------------
    @torch.no_grad()
    def speculate(self, spine: Spine, base: int, choose: Chooser) -> Speculation:
        k, f = spine.tokens.numel(), self.fanout
        template = self._templates(k)
        step = self._tree_pass(template, base)
        # Depth k of the spine: the step on d_k (the outcome "all accepted"), from the state of d_{k-1}.
        last, (self._states[base + k - 1],) = step(template.last, template.valid[:, :1], spine.tokens[-1:],
                                                   [self._states[base + k - 2]])
        candidates, ready, event = outcome_candidates(spine, last, f)
        # outcome a appends its candidate at base + a, after the state of position base + a - 1 (the root for a = 0)
        parents = [self._states[base + b // f - 1] for b in range((k + 1) * f)]
        drafts, depth_states = Drafts(), []
        current = candidates.reshape(-1)
        for nodes in template.levels:
            rows, parents = step(nodes, template.valid, current, parents)
            current, dist = choose(rows)
            drafts.add(current, rows, dist)
            depth_states.append(parents)
        self._branch_states = [[depth_states[j][b] for j in range(k)] for b in range((k + 1) * f)]
        return drafts.speculation(base, candidates, ready, event)

    @torch.no_grad()
    def adopt(self, speculation: Speculation, accepted: int, token: int) -> Spine | None:
        hit = speculation.hit(accepted, token)
        if hit is None:
            return None
        self._move_branch(hit)
        for j, state in enumerate(self._branch_states[hit.branch]):
            self._states[hit.destination + j] = state
        self._pull(hit.destination)  # exact entries of the root and d_1..d_a, exact state of d_a
        self._forget(hit.destination - 1)
        return hit.spine

    def _forget(self, keep_from: int) -> None:
        """Drop the states of positions before ``keep_from`` (they are never read again)."""
        for position in [p for p in self._states if p < keep_from]:
            del self._states[position]


class LinkedSpeculativeDecoder(AsyncSpeculativeDecoder):
    """:class:`~economize.distributed.AsyncSpeculativeDecoder` with a :class:`LinkedDrafter` that receives the
    target's exact states of every committed token.

    A subclass names its worker class in :attr:`worker_class`.

    Args:
        target: The model whose output is reproduced, on its device.
        drafter: The worker on another device.
        config: Decoding settings (``fanout`` is the worker's own).

    Raises:
        ConfigError: ``drafter`` is not a :attr:`worker_class`.
    """

    #: The worker class this decoder links (subclasses included); every subclass sets it.
    worker_class: ClassVar[type[LinkedDrafter]]
    drafter: LinkedDrafter
    target: CapturingTargetWorker

    def __init__(self, target: nn.Module, drafter: LinkedDrafter, config: SSDConfig | None = None) -> None:
        if not isinstance(drafter, self.worker_class):
            name = self.worker_class.__name__
            article = "an" if name[0] in "AEIOU" else "a"
            raise ConfigError(f"{type(self).__name__} needs {article} {name}")
        super().__init__(target, drafter, config)
        self.drafter.link(self.target)

    def _target_worker(self, target: nn.Module) -> CapturingTargetWorker:
        return CapturingTargetWorker(target, self.drafter.observed_layers)

