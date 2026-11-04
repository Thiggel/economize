"""The common base of the draft policies that turn a chain drafter into a tree drafter.

A tree policy (``dynamic_tree``, ``sequoia``, ``specexec``) wraps any drafter that implements
:class:`TreeStepDrafter` in a :class:`TreeSearchDrafter`. The policies differ only in how a proposal grows the tree
from the drafter's steps; this module holds what they share: the checks of :meth:`TreePolicy.wrap`, the members the
wrapper forwards to the drafter, and the rewriting of a verified tree pass into the chain of its accepted path, which
is what the chain drafter observes (:meth:`TreeSearchDrafter.observe`).
"""

from __future__ import annotations

import weakref
from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import torch

from ..errors import ConfigError
from ..runtime.context import ContextLayout, LayerContext
from ..runtime.runner import ExactPass
from .cost import LayerFootprint
from .protocols import Draft, Drafter, DraftPolicy, Verifier

if TYPE_CHECKING:
    from ..runtime import KVCache, LayerRunner
    from .request import DraftRequest

__all__ = ["LOGP_FLOOR", "TreePolicy", "TreeSearchDrafter", "TreeStepDrafter", "select_candidates", "weak_method"]

#: Floor of a candidate's log-probability, so that tokens of probability zero keep a finite score and a child always
#: scores below its parent.
LOGP_FLOOR = -1e4


@runtime_checkable
class TreeStepDrafter(Protocol):
    """A drafter that can run one draft step over the frontier of a draft tree.

    A tree policy calls :meth:`tree_begin` at the start of every proposal, runs the root with the drafter's ordinary
    :meth:`~economize.decoding.Drafter.step`, reads the per-row state that step left with :meth:`tree_state`, and then
    calls :meth:`tree_step` once per draft step of the tree.
    """

    def supports_tree_steps(self) -> bool:
        """Whether this drafter, as configured, implements :meth:`tree_step` (a drafter may support it for some
        configurations only)."""
        ...

    def tree_begin(self, request: DraftRequest) -> None:
        """Prepare a proposal for ``request`` (flush the costs of :meth:`observe`, check that the drafter observed
        an exact pass)."""
        ...

    def tree_state(self) -> tuple[torch.Tensor, ...]:
        """The per-row state ``[B, ...]`` tensors that the root's draft step produced and its children read (an empty
        tuple for a stateless drafter)."""
        ...

    def tree_step(
        self, tokens: torch.Tensor, parents: torch.Tensor, base: torch.Tensor, *states: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """One draft step over the frontier of a tree.

        Args:
            tokens: The frontier's tokens ``[B, W]``.
            parents: The parents of every node the tree forwarded so far ``[B, n]``, the frontier being the last
                ``W``; ``parents[b, i]`` is the index of node ``i``'s parent among them, ``-1`` for a child of the
                root. Node ``i`` sits at slot ``base + i`` and at position ``base`` plus its depth below the root's
                children. The frontier may hold nodes of different depths.
            base: First slot after the committed prefix ``[B]`` (the position of the root's children).
            states: The state of every frontier node's parent ``[B, W, ...]``, in the order of :meth:`tree_state`.

        Returns:
            The float logits after every frontier node ``[B, W, V]`` and the frontier's own states (same order and
            shapes as ``states``). The step writes the frontier's keys and values to its slots and has no host
            synchronization, so it can be recorded as a CUDA graph.
        """
        ...


class TreePolicy(DraftPolicy):
    """A draft policy that drafts a tree with any :class:`TreeStepDrafter` (a subclass builds the wrapper).

    The tree replaces the chain, so no row stops early (:attr:`may_stop` is false).
    """

    may_stop = False
    #: The name the policy is registered under, for error messages.
    policy_name: ClassVar[str] = "tree"

    def stops(self, index: int, logits: torch.Tensor) -> torch.Tensor | None:
        return None

    def check_drafter(self, drafter: Drafter, verifier: Verifier) -> None:
        """Raise :class:`~economize.ConfigError` unless ``drafter`` can step over trees that ``verifier`` verifies.

        A tree policy chooses draft tokens by rank, so it has no draft distribution to offer a verifier that needs
        one (rejection sampling), and it wraps chain drafters only.
        """
        name, checker, policy = type(drafter).__name__, type(verifier).__name__, self.policy_name
        if verifier.commits_drafts or not verifier.supports_trees:
            raise ConfigError(f"draft_policy {policy!r} drafts trees, which verifier {checker} cannot verify")
        if verifier.needs_draft_probs:
            raise ConfigError(f"draft_policy {policy!r} chooses draft tokens by rank, so it has no draft "
                              f"distribution for verifier {checker}; use the greedy or typical verifier")
        if drafter.drafts_trees(verifier):
            raise ConfigError(f"{name} drafts trees of its own; draft_policy {policy!r} wraps chain drafters")
        if not isinstance(drafter, TreeStepDrafter) or not drafter.supports_tree_steps():
            raise ConfigError(f"{name} cannot draft a tree for draft_policy {policy!r} (it does not implement "
                              "TreeStepDrafter for this configuration); see docs/dynamic_tree.md for the drafters "
                              "that do")


def weak_method(method: Callable[..., Any]) -> Callable[..., Any]:
    """``method`` held through a weak reference to its object (graphed steps outlive no drafter)."""
    ref = weakref.WeakMethod(method)

    def call(*args: Any) -> Any:
        bound = ref()
        if bound is None:
            raise RuntimeError("the drafter whose tree step this graph runs was deleted")
        return bound(*args)

    return call


def select_candidates(tokens: torch.Tensor, scores: torch.Tensor, parents: torch.Tensor, depth: torch.Tensor,
                      total: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``total`` best candidates of a tree search as a draft tree in topological order.

    Args:
        tokens: Token of every candidate ``[B, C]``; candidates are numbered so that a parent precedes its children.
        scores: Cumulative log-probability of every candidate ``[B, C]`` (a child never scores above its parent).
        parents: Candidate index of every candidate's parent ``[B, C]``, ``-1`` for a child of the root.
        depth: Depth of every candidate ``[C]`` or ``[B, C]``; a tie goes to the shallower candidate, so the selection
            is closed under ancestors.
        total: Nodes of the draft tree.

    Returns:
        The tree's tokens ``[B, total]`` and the parent of every node among them ``[B, total]`` (``-1`` for the root's
        children). No host synchronization.
    """
    batch, device = tokens.shape[0], tokens.device
    key = scores.double() - depth.double() * 1e-9
    chosen = key.topk(total, -1).indices.sort(-1).values  # candidates are numbered parents first
    place = torch.full_like(tokens, -1).scatter(1, chosen, torch.arange(total, device=device).expand(batch, total))
    parent = parents.gather(1, chosen)
    tree_parents = torch.where(parent >= 0, place.gather(1, parent.clamp_min(0)), -1)
    return tokens.gather(1, chosen), tree_parents


class TreeSearchDrafter(Drafter):
    """A chain :class:`TreeStepDrafter` that proposes draft trees; subclasses implement :meth:`propose`.

    The wrapper forwards the inner drafter's members (its layers, footprint and any attribute it does not define
    itself), runs the root through the inner drafter's chain step, and turns every verified tree pass into the chain
    of the accepted path before the inner drafter observes it, so a chain drafter keeps the state it knows.

    Args:
        inner: The chain drafter.
        policy_name: Name of the draft policy that made this drafter, for error messages.
    """

    def __init__(self, inner: Drafter, policy_name: str) -> None:
        self.inner = inner
        self.policy_name = policy_name
        self._trash = 0
        self._index: torch.Tensor | None = None

    @property
    @abstractmethod
    def max_depth(self) -> int:
        """Depth of the deepest node a proposal can draft (the longest path the exact pass can accept)."""

    # ---- forwarded members of the inner drafter ---------------------------------------------------------------
    @property
    def source_layers(self) -> tuple[int, ...]:  # type: ignore[override]
        return self.inner.source_layers

    @property
    def layer_span(self) -> tuple[int, int]:  # type: ignore[override]
        return self.inner.layer_span

    @property
    def footprint(self) -> LayerFootprint | None:  # type: ignore[override]
        return self.inner.footprint

    @property
    def observed_layers(self) -> tuple[int, ...]:
        return self.inner.observed_layers

    def __getattr__(self, name: str) -> Any:
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        if runner.has_recurrent_layers:
            raise ConfigError(f"draft_policy {self.policy_name!r} needs a model without recurrent layers (its draft "
                              "steps cover part of a tree at a time, and recurrent layers run trees only as whole "
                              "root-to-node paths)")
        self._trash = cache.trash_slot
        self.inner.begin(runner, cache, batch_size)

    def step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        return self.inner.step(tokens, positions, valid)

    def reorder(self, rows: torch.Tensor) -> None:
        self.inner.reorder(rows)

    def drafts_trees(self, verifier: Verifier) -> bool:
        return True

    @abstractmethod
    def propose(self, request: DraftRequest) -> Draft:
        """One draft tree per row (see :class:`~economize.decoding.Drafter`)."""

    # ---- the exact pass ---------------------------------------------------------------------------------------
    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None:
        ctx = exact.context
        self._index = None
        if ctx is None or ctx.tree is None:
            self.inner.observe(exact, columns)
            return
        chain, path_columns = self._accepted_chain(exact, ctx, columns)
        self.inner.observe(chain, path_columns)

    def _accepted_chain(self, exact: ExactPass, ctx: LayerContext,
                        columns: torch.Tensor) -> tuple[ExactPass, torch.Tensor]:
        """The tree pass ``exact`` rewritten as the chain of every row's accepted path (root first), and the column
        of the last accepted token in it."""
        tree = ctx.tree
        assert tree is not None
        device = tree.visible.device
        batch, size = tree.visible.shape[:2]
        rows = torch.arange(batch, device=device)
        path = tree.visible[rows, columns.to(device)]  # [B, S]: the last accepted node and its ancestors
        width = min(size, self.max_depth + 1)
        order = torch.where(path, torch.arange(size, device=device), size).sort(-1).values[:, :width]
        valid = order < size
        index = order.clamp(max=size - 1)
        picked = ctx.select(index, valid)
        layout = ContextLayout(picked.positions, torch.where(valid, picked.positions, self._trash), valid, None)
        context = LayerContext.from_layout(picked.adapter, picked.cache, layout, picked.token_inputs)

        def gather(states: torch.Tensor) -> torch.Tensor:
            local = index.to(states.device)
            return states[torch.arange(batch, device=states.device)[:, None], local]

        chain = ExactPass(gather(exact.final), {layer: gather(h) for layer, h in exact.hidden.items()}, context)
        self._index = index
        return chain, (path.sum(-1) - 1).to(columns.device)

    def observe_logits(self, logits: torch.Tensor) -> None:
        if type(self.inner).observe_logits is Drafter.observe_logits:
            return
        index = self._index
        if index is not None and index.shape[1] < logits.shape[1]:
            local = index.to(logits.device)
            logits = logits[torch.arange(logits.shape[0], device=logits.device)[:, None], local]
        self.inner.observe_logits(logits)
