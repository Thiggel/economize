"""Dynamic draft trees (EAGLE-2) for any drafter that can step over the frontier of a tree.

``DecodeConfig(draft_policy="dynamic_tree", draft_policy_options={"depth": 6, "topk": 10, "total_tokens": 60})``
turns a chain drafter into a tree drafter. Every cycle the drafter grows a tree of candidates level by level:

1. The first draft step processes the root (the last committed token) as in chain drafting. Its ``topk`` most likely
   tokens are the candidates of depth 1.
2. At every further depth the ``topk`` candidates of the previous depth with the highest cumulative log-probability
   (the frontier) run through one batched draft step under a tree mask, and each contributes its own ``topk`` most
   likely tokens as candidates of the next depth.
3. After ``depth`` levels the ``total_tokens`` candidates with the highest cumulative log-probability form the draft.
   A child never scores higher than its parent, so the selection is closed under ancestors.

The exact pass verifies the tree as usual (one pass with tree attention, the longest accepted path, the cache entries
of that path moved to their positions), so decoding stays lossless under every verifier that compares draft tokens
with the tokens the exact model chooses. Before the drafter adopts the exact states of the pass, the tree pass is
rewritten as the chain of the accepted path, so a chain drafter observes the pass it knows.

A drafter takes part by implementing :class:`TreeStepDrafter`; EAGLE-3, jump drafting (``jump``, ``jump_exit``),
the self-drafters (early exit, LayerSkip) and FR-Spec around any of them do. See ``docs/dynamic_tree.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..errors import ConfigError
from ..runtime.transfer import device_tensor
from ..validation import is_positive_int
from .cost import CostKind
from .policies import POLICIES
from .protocols import Draft, Drafter, Verifier, draft_footprint
from .tree_search import LOGP_FLOOR, TreePolicy, TreeSearchDrafter, TreeStepDrafter, select_candidates, weak_method

if TYPE_CHECKING:
    from .config import DecodeConfig
    from .graphs import GraphWrap
    from .request import DraftRequest

__all__ = ["DEFAULT_TOPK", "DEFAULT_TOTAL_TOKENS", "DynamicTreeDrafter", "DynamicTreePolicy", "TreeOptions",
           "TreeStepDrafter"]

#: Default frontier width and children per node (``top_k=10`` in the EAGLE-2 reference code).
DEFAULT_TOPK = 10
#: Default number of verified tree nodes (``total_token=60`` in the EAGLE-2 reference code).
DEFAULT_TOTAL_TOKENS = 60

@dataclass(frozen=True)
class TreeOptions:
    """The shape of a dynamic draft tree.

    Attributes:
        depth: Levels of candidates below the root (the longest path the exact pass can accept).
        topk: Frontier width per level and children per frontier node.
        total_tokens: Nodes of the draft tree the exact pass verifies (the economize attention kernel takes a tree
            pass of up to 64 queries, i.e. ``total_tokens <= 63``).
    """

    depth: int
    topk: int
    total_tokens: int

    @property
    def forwarded(self) -> int:
        """Nodes every proposal runs through the drafter besides the root: ``(depth - 1) * topk``."""
        return (self.depth - 1) * self.topk

    @property
    def candidates(self) -> int:
        """Candidates every proposal ranks: ``topk`` at depth 1 and ``topk**2`` at every further depth."""
        return self.topk + (self.depth - 1) * self.topk * self.topk

    @classmethod
    def from_config(cls, config: DecodeConfig) -> TreeOptions:
        """The options in ``config.draft_policy_options``.

        Defaults follow the EAGLE-2 reference code: ``topk=10`` and ``total_tokens=60``, with ``depth`` set to
        ``cycle_length - 1``. A shallow tree with fewer than 60 candidates (``cycle_length=2``: ``topk`` of them)
        verifies all of them by default, so every cycle length works without options.
        """
        options: Mapping[str, object] = config.draft_policy_options
        unknown = sorted(set(options) - {"depth", "topk", "total_tokens"})
        if unknown:
            raise ConfigError(f"draft_policy 'dynamic_tree' has no options {unknown}; it takes depth, topk and "
                              "total_tokens")
        values: dict[str, object] = {"depth": max(config.cycle_length - 1, 1), "topk": DEFAULT_TOPK, **options}
        if "total_tokens" not in values:
            depth, topk = values["depth"], values["topk"]
            values["total_tokens"] = (
                min(DEFAULT_TOTAL_TOKENS, cls(depth, topk, 1).candidates)
                if is_positive_int(depth) and is_positive_int(topk) else DEFAULT_TOTAL_TOKENS
            )
        checked: dict[str, int] = {}
        for name, value in values.items():
            if not is_positive_int(value):
                raise ConfigError(f"draft_policy 'dynamic_tree' needs a positive integer {name}, got {value!r}")
            checked[name] = value
        tree = cls(checked["depth"], checked["topk"], checked["total_tokens"])
        if tree.total_tokens > tree.candidates:
            raise ConfigError(f"total_tokens={tree.total_tokens} exceeds the {tree.candidates} candidates of a tree "
                              f"of depth {tree.depth} and topk {tree.topk}")
        return tree


@POLICIES.register("dynamic_tree")
class DynamicTreePolicy(TreePolicy):
    """Drafts an EAGLE-2 dynamic tree with any :class:`TreeStepDrafter` (see the module docstring).

    Options (``DecodeConfig.draft_policy_options``): ``depth`` (default ``cycle_length - 1``), ``topk`` (default 10,
    as in EAGLE-2) and ``total_tokens`` (default 60, or every candidate of a smaller tree). The tree replaces the
    chain, so no row stops early.
    """

    policy_name = "dynamic_tree"

    def __init__(self, config: DecodeConfig) -> None:
        super().__init__(config)
        self.options = TreeOptions.from_config(config)

    def wrap(self, drafter: Drafter, verifier: Verifier) -> Drafter:
        self.check_drafter(drafter, verifier)
        return DynamicTreeDrafter(drafter, self.options)


class DynamicTreeDrafter(TreeSearchDrafter):
    """A chain :class:`TreeStepDrafter` that proposes EAGLE-2 dynamic trees (made by :class:`DynamicTreePolicy`).

    The cost log records the root's step as one ``draft`` call of one token at the root's position, every further
    depth ``d`` as one ``draft`` call of ``topk`` tokens at the position of depth ``d`` (both over the inner drafter's
    footprint), and the exact pass as a call over the ``total_tokens + 1`` tree tokens.

    With CUDA graphs the inner drafter's chain step runs from its own graphs and every tree step from a graph per
    depth (the shapes depend only on the batch size, ``topk`` and the depth).
    """

    def __init__(self, inner: Drafter, options: TreeOptions) -> None:
        super().__init__(inner, "dynamic_tree")
        self.options = options
        # the three tensor functions of a cycle; enable_graphs replaces each by its CUDA graph
        self._root_fn: Callable[..., tuple[torch.Tensor, ...]] = self._root
        self._grow_fn: Callable[..., tuple[torch.Tensor, ...]] = self._grow
        self._select_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = self._select

    @property
    def max_depth(self) -> int:
        return self.options.depth

    def draft_capacity(self, cycle_length: int) -> int:
        # the verified tree holds total_tokens nodes; the draft steps write the forwarded nodes after the prefix
        return max(self.options.total_tokens, self.options.forwarded)

    def enable_graphs(self, wrap: GraphWrap) -> bool:
        if not self.inner.enable_graphs(wrap):
            return False
        self._root_fn = wrap(weak_method(self._root))
        self._grow_fn = wrap(weak_method(self._grow))  # one graph per depth: the shapes grow with it
        self._select_fn = wrap(weak_method(self._select))
        return True

    # ---- drafting ---------------------------------------------------------------------------------------------
    def propose(self, request: DraftRequest) -> Draft:
        inner = self._stepper
        k, depth = self.options.topk, self.options.depth
        batch, device = request.batch_size, request.device
        footprint = draft_footprint(self.inner)
        inner.tree_begin(request)
        logits = self.inner.step(request.tokens, request.positions, None)
        for b, cost in enumerate(request.costs):
            cost.record_footprint(CostKind.DRAFT, 1, request.ends[b] - 1, footprint, step=0)
        n_states = len(inner.tree_state())
        tokens, scores, *states = self._root_fn(logits.to(device), *inner.tree_state())
        frontier = torch.arange(k, device=device).expand(batch, k)  # candidate index of every frontier node
        levels = [(tokens, scores, torch.full((batch, k), -1, dtype=torch.long, device=device))]
        forwarded = torch.full((batch, k), -1, dtype=torch.long, device=device)  # parents of the forwarded nodes
        base = device_tensor(request.ends, device)
        for d in range(1, depth):
            out = self._grow_fn(tokens, scores, frontier, forwarded, base, *states)
            for b, cost in enumerate(request.costs):
                cost.record_footprint(CostKind.DRAFT, k, request.ends[b] + d - 1, footprint, step=d)
            levels.append((out[0], out[1], out[2]))
            if d < depth - 1:
                tokens, scores, frontier, forwarded = out[3:7]
                states = list(out[7:7 + n_states])
        chosen, parents = self._select_fn(*(lv[0] for lv in levels), *(lv[1] for lv in levels),
                                          *(lv[2] for lv in levels))
        return Draft(chosen.clone(), [self.options.total_tokens] * batch, parents.clone())

    def _root(self, logits: torch.Tensor, *states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The root's ``k`` children (tokens and log-probabilities ``[B, k]``) and the root's states per child."""
        k = self.options.topk
        top = torch.log_softmax(logits.float(), -1).clamp_min(LOGP_FLOOR).topk(k, -1)
        expanded = tuple(s[:, None].expand(s.shape[0], k, *s.shape[1:]).contiguous() for s in states)
        return (top.indices, top.values, *expanded)

    def _grow(self, frontier_tokens: torch.Tensor, frontier_scores: torch.Tensor, frontier: torch.Tensor,
              forwarded: torch.Tensor, base: torch.Tensor, *states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """One depth: the frontier's draft step and its ``k * k`` children ``[B, k*k]`` (tokens, scores, candidate
        index of the parent), then
        (except at the last depth) the next frontier's tokens, scores, candidate indices, the parents of all
        forwarded nodes, and its states. The depth follows from the shapes, so this is a function of tensors."""
        k, depth = self.options.topk, self.options.depth
        batch = frontier_tokens.shape[0]
        d = forwarded.shape[1] // k  # forwarded holds k parents per forwarded level
        logits, out = self._stepper.tree_step(frontier_tokens, forwarded, base, *states)
        top = torch.log_softmax(logits.float(), -1).clamp_min(LOGP_FLOOR).topk(k, -1)
        here = frontier_scores.device
        child_scores = (frontier_scores[:, :, None] + top.values.to(here)).reshape(batch, k * k)
        child_tokens = top.indices.to(here).reshape(batch, k * k)
        child_parents = frontier[:, :, None].expand(batch, k, k).reshape(batch, k * k)
        if d >= depth - 1:
            return child_tokens, child_scores, child_parents
        pick = child_scores.topk(k, -1).indices  # the next frontier
        origin = pick // k  # its parents' places in the current frontier
        rows = torch.arange(batch, device=pick.device)[:, None]
        count = k + (d - 1) * k * k  # candidates before this depth's children
        return (child_tokens, child_scores, child_parents, child_tokens.gather(1, pick), child_scores.gather(1, pick),
                count + pick, torch.cat([forwarded, (d - 1) * k + origin], 1), *(s[rows, origin] for s in out))

    def _select(self, *levels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``total_tokens`` best candidates of all levels (the tokens of every level, then their scores, then
        their parents' candidate indices) as a draft tree in topological order: its tokens and the parent of every
        node (``-1`` for the root's children)."""
        n = len(levels) // 3
        tokens, scores, parents = (torch.cat(levels[i * n:(i + 1) * n], 1) for i in range(3))
        depth_of = torch.cat([torch.full((lv.shape[1],), i, dtype=torch.float64, device=tokens.device)
                              for i, lv in enumerate(levels[:n])])
        return select_candidates(tokens, scores, parents, depth_of, self.options.total_tokens)
