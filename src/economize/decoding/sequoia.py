"""Sequoia draft trees: a static tree of the best shape for measured acceptance rates and the hardware's latencies.

``DecodeConfig(draft_policy="sequoia", draft_policy_options={"acceptance": (0.6, 0.15, 0.06, 0.03),
"total_tokens": 32, "depth": 8})`` turns a chain drafter into a drafter of one fixed tree per cycle (Chen et al.,
2024, "Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding"). Every node of the tree names its rank
among its siblings; a proposal fills the tree with the drafter's ``rank``-th most likely token after every parent, one
batched draft step per depth over the nodes of that depth that have children.

The shape follows from the acceptance vector by dynamic programming. ``acceptance[r]`` is the probability that the
exact model's token equals the drafter's ``r``-th most likely token after an accepted node, and the events are
disjoint across siblings, so a tree's expected number of tokens per cycle is the sum over its nodes of the product
of the acceptance rates along the node's rank path (the root, which always yields one token, included). For every
size ``n <= total_tokens`` and every depth ``D <= depth`` :func:`sequoia_tree` finds the tree of at most ``n`` nodes
and depth at most ``D`` that maximizes this sum (Sequoia's Eq. 2, solved exactly). With measured latencies
(``verify_latency``: entry ``i`` is the time of a base pass over ``i + 1`` tokens; ``draft_latency``: one draft step)
the policy picks the size and depth that maximize expected tokens per second, which is Sequoia's hardware-aware
choice; without them it takes the largest tree.

:func:`measure_acceptance` estimates the vector on prompts with a probe tree, and an explicit tree can be given as
rank paths (``paths``, the format of Medusa's tree choices). The exact pass verifies the tree as any draft tree, so
decoding stays lossless under every verifier that compares draft tokens with the exact model's tokens.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ..errors import ConfigError
from ..runtime.transfer import device_tensor
from .cost import CostKind
from .policies import POLICIES
from .protocols import Draft, Drafter, Verifier, draft_footprint
from .tree_search import TreePolicy, TreeSearchDrafter, weak_method

if TYPE_CHECKING:
    from ..runtime.runner import ExactPass
    from .config import DecodeConfig
    from .request import DraftRequest

__all__ = [
    "AcceptanceProfile",
    "SequoiaDrafter",
    "SequoiaPolicy",
    "StaticTree",
    "expected_tokens",
    "measure_acceptance",
    "probe_paths",
    "sequoia_tree",
]

Path = tuple[int, ...]


@dataclass(frozen=True)
class StaticTree:
    """A draft tree of fixed shape, its nodes in breadth-first order.

    Attributes:
        paths: Rank path of every node (``(0, 2)`` is the third most likely child of the most likely child of the
            root), sorted by depth and then lexicographically, so that a parent precedes its children and the
            children of a node follow in rank order.
        expected: Expected tokens per cycle under the acceptance rates the tree was built for (root included),
            ``None`` for an explicit tree.
    """

    paths: tuple[Path, ...]
    expected: float | None = None

    @classmethod
    def from_paths(cls, paths: Sequence[Sequence[int]], expected: float | None = None) -> StaticTree:
        """The tree of the rank paths ``paths`` (any order; every prefix of a path must be a path too)."""
        unique = sorted({tuple(int(r) for r in p) for p in paths}, key=lambda p: (len(p), p))
        if not unique:
            raise ConfigError("a sequoia tree needs at least one node")
        present = set(unique)
        for p in unique:
            if not p or min(p) < 0:
                raise ConfigError(f"tree path {p} must be a non-empty tuple of ranks >= 0")
            if len(p) > 1 and p[:-1] not in present:
                raise ConfigError(f"tree path {p} has no parent {p[:-1]} in the tree")
        return cls(tuple(unique), expected)

    @property
    def size(self) -> int:
        """Nodes besides the root (the draft tokens of a cycle)."""
        return len(self.paths)

    @property
    def depth(self) -> int:
        """Depth of the deepest node."""
        return len(self.paths[-1])

    @property
    def parents(self) -> tuple[int, ...]:
        """Breadth-first index of every node's parent, ``-1`` for a child of the root."""
        index = {p: i for i, p in enumerate(self.paths)}
        return tuple(index[p[:-1]] if len(p) > 1 else -1 for p in self.paths)

    @property
    def ranks(self) -> tuple[int, ...]:
        """Rank of every node among its siblings (the last entry of its path)."""
        return tuple(p[-1] for p in self.paths)

    def children(self) -> list[list[int]]:
        """Breadth-first indices of every node's children, the root's last (index ``-1``)."""
        out: list[list[int]] = [[] for _ in range(self.size + 1)]
        for i, parent in enumerate(self.parents):
            out[parent].append(i)
        return out

    @property
    def forwarded(self) -> int:
        """Nodes a proposal runs through the drafter besides the root: the nodes with children."""
        return sum(1 for c in self.children()[:-1] if c)


def _acceptance_matrix(acceptance: Sequence[float] | Sequence[Sequence[float]], depth: int) -> np.ndarray:
    """The acceptance rates as a matrix ``[depth, K]`` (row ``l``: after a node of depth ``l``, the root being 0)."""
    rows: Sequence[Any] = acceptance if acceptance and isinstance(acceptance[0], Sequence) else [acceptance]
    matrix = [[float(v) for v in row] for row in rows]
    width = max(len(row) for row in matrix)
    if width == 0:
        raise ConfigError("the acceptance vector is empty")
    for row in matrix:
        if any(not 0.0 <= v <= 1.0 for v in row) or sum(row) > 1.0 + 1e-6:
            raise ConfigError(f"acceptance rates {row} must lie in [0, 1] and sum to at most 1 (they are the "
                              "probabilities of disjoint events)")
    padded = np.zeros((max(depth, len(matrix)), width))
    for i in range(padded.shape[0]):
        row = matrix[min(i, len(matrix) - 1)]
        padded[i, :len(row)] = row
    return padded[:depth]


def expected_tokens(tree: StaticTree, acceptance: Sequence[float] | Sequence[Sequence[float]]) -> float:
    """Expected tokens per cycle of ``tree`` (the root included) when the exact model's token after an accepted node
    of depth ``l`` is the drafter's rank-``r`` token with probability ``acceptance[l][r]`` (one vector: every depth)."""
    p = _acceptance_matrix(acceptance, max(tree.depth, 1))
    reach: dict[Path, float] = {(): 1.0}
    for path in tree.paths:
        rank, depth = path[-1], len(path) - 1
        reach[path] = reach[path[:-1]] * (p[depth, rank] if rank < p.shape[1] else 0.0)
    return float(sum(reach.values()))


def _dp(p: np.ndarray, size: int) -> tuple[np.ndarray, list[list[np.ndarray]]]:
    """Sequoia's dynamic program for the acceptance matrix ``p`` ``[D, K]`` (``D`` the largest depth).

    ``G[l, n]`` is the largest expected token count of a subtree of at most ``n`` nodes rooted at depth ``l``, with
    ``G[D, n] = 1`` for ``n >= 1``. The root's children take ranks ``0, 1, ...`` in turn: ``H_l(m, i)`` is the best
    use of ``m`` nodes by the children of rank ``i`` and above, ``max(0, max_s p[l, i] G[l+1, s] + H_l(m-s, i+1))``,
    and ``G[l, n] = 1 + H_l(n - 1, 0)``. Returns ``G`` and the size ``s`` chosen for every ``(l, i, m)``.
    """
    depth, width = p.shape
    g = np.zeros((depth + 1, size + 1))
    g[depth, 1:] = 1.0
    choices: list[list[np.ndarray]] = [[] for _ in range(depth)]
    m = np.arange(size + 1)
    s = np.arange(size + 1)
    legal = (s[None, :] >= 1) & (s[None, :] <= m[:, None])
    rest = np.clip(m[:, None] - s[None, :], 0, size)
    for level in range(depth - 1, -1, -1):
        h = np.zeros(size + 1)  # H(., width): no ranks left
        picks: list[np.ndarray] = []
        for rank in range(width - 1, -1, -1):
            gain = np.where(legal, p[level, rank] * g[level + 1][s][None, :] + h[rest], -np.inf)
            best = gain.argmax(1)
            value = gain[m, best]
            take = value > 0.0  # stop (value 0) unless a child of this rank adds expected tokens
            h = np.where(take, value, 0.0)
            picks.append(np.where(take, best, 0))
        choices[level] = picks[::-1]
        g[level, 1:] = 1.0 + h[:-1]
    return g, choices


def _build(choices: list[list[np.ndarray]], level: int, nodes: int, prefix: Path, out: list[Path]) -> None:
    """Append the rank paths of the best subtree of ``nodes`` nodes rooted at ``prefix`` (of depth ``level``)."""
    budget, rank = nodes - 1, 0
    while level < len(choices) and rank < len(choices[level]) and budget > 0:
        take = int(choices[level][rank][budget])
        if take == 0:
            break
        child = (*prefix, rank)
        out.append(child)
        _build(choices, level + 1, take, child, out)
        budget -= take
        rank += 1


def sequoia_tree(acceptance: Sequence[float] | Sequence[Sequence[float]], total_tokens: int, depth: int, *,
                 verify_latency: Sequence[float] | None = None, draft_latency: float = 0.0) -> StaticTree:
    """The Sequoia tree for the acceptance rates ``acceptance`` (see the module docstring).

    Args:
        acceptance: ``acceptance[r]``, the probability that the exact model's token after an accepted node is the
            drafter's ``r``-th most likely token; or one such vector per depth of the parent (the root's first, the
            last reused below).
        total_tokens: Largest number of draft nodes.
        depth: Largest depth.
        verify_latency: Entry ``i``: seconds of a base pass over ``i + 1`` tokens (at least ``total_tokens + 1``
            entries). With it the tree maximizes expected tokens per second over all sizes and depths; without it
            the tree maximizes expected tokens per cycle.
        draft_latency: Seconds of one draft step (one per depth of the tree), used with ``verify_latency``.

    Returns:
        The tree, its :attr:`StaticTree.expected` set.
    """
    if total_tokens < 1 or depth < 1:
        raise ConfigError(f"a sequoia tree needs total_tokens >= 1 and depth >= 1, got {total_tokens} and {depth}")
    p = _acceptance_matrix(acceptance, depth)
    if verify_latency is not None and len(verify_latency) < total_tokens + 1:
        raise ConfigError(f"verify_latency needs an entry for every pass of 1..{total_tokens + 1} tokens, got "
                          f"{len(verify_latency)}")
    if verify_latency is None:
        g, choices = _dp(p, total_tokens + 1)
        paths: list[Path] = []
        _build(choices, 0, total_tokens + 1, (), paths)
        return StaticTree.from_paths(paths, float(g[0, total_tokens + 1]))
    best: tuple[float, int, int] | None = None
    tables = {}
    for d in range(1, depth + 1):
        g, choices = _dp(p[:d], total_tokens + 1)
        tables[d] = choices
        for n in range(1, total_tokens + 1):
            rate = g[0, n + 1] / (float(verify_latency[n]) + d * draft_latency)
            if best is None or rate > best[0] + 1e-12:
                best = (rate, d, n)
    assert best is not None
    _, d, n = best
    paths = []
    _build(tables[d], 0, n + 1, (), paths)
    tree = StaticTree.from_paths(paths)
    return StaticTree(tree.paths, expected_tokens(tree, p[:d].tolist()))


def probe_paths(depth: int, width: int) -> tuple[Path, ...]:
    """The rank paths of the probe tree of :func:`measure_acceptance`: a spine of rank-0 nodes ``depth`` deep, every
    spine node (and the root) with children of ranks ``0..width-1``."""
    return tuple((0,) * level + (rank,) for level in range(depth) for rank in range(width))


# ---- the policy -----------------------------------------------------------------------------------------------------
_OPTIONS = ("acceptance", "paths", "total_tokens", "depth", "verify_latency", "draft_latency", "record_acceptance")


def _tree_from_config(config: DecodeConfig) -> tuple[StaticTree, bool]:
    options: Mapping[str, Any] = config.draft_policy_options
    unknown = sorted(set(options) - set(_OPTIONS))
    if unknown:
        raise ConfigError(f"draft_policy 'sequoia' has no options {unknown}; it takes {', '.join(_OPTIONS)}")
    record = options.get("record_acceptance", False)
    if not isinstance(record, bool):
        raise ConfigError(f"record_acceptance must be a bool, got {record!r}")
    if ("paths" in options) == ("acceptance" in options):
        raise ConfigError("draft_policy 'sequoia' needs exactly one of paths (an explicit tree) and acceptance "
                          "(rates to build the tree from)")
    if "paths" in options:
        extra = sorted(set(options) & {"total_tokens", "depth", "verify_latency", "draft_latency"})
        if extra:
            raise ConfigError(f"draft_policy 'sequoia' builds no tree from an explicit paths option; drop {extra}")
        return StaticTree.from_paths(options["paths"]), record
    depth = options.get("depth", max(config.cycle_length - 1, 1))
    total = options.get("total_tokens", 32)
    for name, value in (("depth", depth), ("total_tokens", total)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(f"draft_policy 'sequoia' needs a positive integer {name}, got {value!r}")
    latency = options.get("verify_latency")
    draft = options.get("draft_latency", 0.0)
    if isinstance(draft, bool) or not isinstance(draft, (int, float)) or draft < 0:
        raise ConfigError(f"draft_latency must be a number >= 0, got {draft!r}")
    if "draft_latency" in options and latency is None:
        raise ConfigError("draft_latency needs verify_latency")
    return sequoia_tree(options["acceptance"], total, depth, verify_latency=latency, draft_latency=float(draft)), record


@POLICIES.register("sequoia")
class SequoiaPolicy(TreePolicy):
    """Drafts one Sequoia tree per cycle with any :class:`~economize.decoding.TreeStepDrafter`.

    Options (``DecodeConfig.draft_policy_options``; lists are stored as tuples):

    * ``acceptance``: the acceptance vector (or one per depth) to build the tree from with :func:`sequoia_tree`;
      ``total_tokens`` (default 32), ``depth`` (default ``cycle_length - 1``), ``verify_latency`` and
      ``draft_latency`` as there.
    * ``paths``: an explicit tree as rank paths, instead of ``acceptance``.
    * ``record_acceptance``: count, after every exact pass, which rank the exact token had at every accepted node
      (:attr:`SequoiaDrafter.acceptance`; one host synchronization per cycle).
    """

    policy_name = "sequoia"

    def __init__(self, config: DecodeConfig) -> None:
        super().__init__(config)
        self.tree, self.record = _tree_from_config(config)

    def wrap(self, drafter: Drafter, verifier: Verifier) -> Drafter:
        self.check_drafter(drafter, verifier)
        return SequoiaDrafter(drafter, self.tree, record_acceptance=self.record)


@dataclass
class AcceptanceProfile:
    """Counts of the exact model's token ranks among the drafter's candidates at accepted tree nodes.

    Attributes:
        hits: ``hits[l, r]``: accepted nodes of depth ``l`` whose exact next token was the drafter's rank-``r`` token.
        trials: ``trials[l, r]``: accepted nodes of depth ``l`` that had a child of rank ``r`` in the tree.
    """

    hits: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    trials: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    def add(self, level: int, children: int, outcome: int | None) -> None:
        """One accepted node of depth ``level`` with ``children`` children, whose exact token was its rank-``outcome``
        child (``None``: none of them)."""
        if children == 0:
            return
        rows, cols = max(self.hits.shape[0], level + 1), max(self.hits.shape[1], children)
        if (rows, cols) != self.hits.shape:
            hits, trials = np.zeros((rows, cols)), np.zeros((rows, cols))
            hits[:self.hits.shape[0], :self.hits.shape[1]] = self.hits
            trials[:self.trials.shape[0], :self.trials.shape[1]] = self.trials
            self.hits, self.trials = hits, trials
        self.trials[level, :children] += 1
        if outcome is not None:
            self.hits[level, outcome] += 1

    def rates(self, per_depth: bool = False) -> tuple[float, ...] | tuple[tuple[float, ...], ...]:
        """The acceptance vector (pooled over depths), or one vector per depth: ``hits / trials`` (0 where there
        were no trials). The result can be passed as the ``acceptance`` option of the ``sequoia`` policy."""
        if per_depth:
            rate = np.divide(self.hits, self.trials, out=np.zeros_like(self.hits), where=self.trials > 0)
            return tuple(tuple(float(v) for v in row) for row in rate)
        hits, trials = self.hits.sum(0), self.trials.sum(0)
        return tuple(float(v) for v in np.divide(hits, trials, out=np.zeros_like(hits), where=trials > 0))


class SequoiaDrafter(TreeSearchDrafter):
    """A chain :class:`~economize.decoding.TreeStepDrafter` that proposes one fixed tree per cycle (made by
    :class:`SequoiaPolicy`).

    The root runs through the inner drafter's chain step; every further depth ``d`` with nodes that have children runs
    those nodes (the frontier) in one tree step. The cost log records the root's step as one ``draft`` call of one
    token and the step of depth ``d`` as one call of the frontier's width, both over the inner drafter's footprint.
    With CUDA graphs the root's expansion and every depth's step run from graphs (their shapes are fixed by the tree).

    Args:
        inner: The chain drafter.
        tree: The tree.
        record_acceptance: Keep :attr:`acceptance` up to date after every exact pass.
    """

    def __init__(self, inner: Drafter, tree: StaticTree, *, record_acceptance: bool = False) -> None:
        super().__init__(inner, "sequoia")
        self.tree = tree
        self.record_acceptance = record_acceptance
        self.acceptance = AcceptanceProfile()
        self._plan = _Plan(tree)
        self._consts: dict[torch.device, _Consts] = {}
        self._root_fn: Callable[..., tuple[torch.Tensor, ...]] = self._root
        self._level_fn: Callable[..., tuple[torch.Tensor, ...]] = self._level

    @property
    def max_depth(self) -> int:
        return self.tree.depth

    def draft_capacity(self, cycle_length: int) -> int:
        return max(self.tree.size, self.tree.forwarded)

    def enable_graphs(self, wrap: Callable[[Callable[..., Any]], Callable[..., Any]]) -> bool:
        if not self.inner.enable_graphs(wrap):
            return False
        self._root_fn = wrap(weak_method(self._root))
        self._level_fn = wrap(weak_method(self._level))  # one graph per depth: the shapes differ
        return True

    def _const(self, device: torch.device) -> _Consts:
        consts = self._consts.get(device)
        if consts is None:
            consts = self._consts[device] = _Consts(self._plan, device)
        return consts

    # ---- drafting ---------------------------------------------------------------------------------------------
    def propose(self, request: DraftRequest) -> Draft:
        inner: Any = self.inner
        plan, batch, device = self._plan, request.batch_size, request.device
        consts = self._const(torch.device(device))
        footprint = draft_footprint(inner)
        inner.tree_begin(request)
        logits = inner.step(request.tokens, request.positions, None)
        for b, cost in enumerate(request.costs):
            cost.record_footprint(CostKind.DRAFT, 1, request.ends[b] - 1, footprint, step=0)
        n_states = len(inner.tree_state())
        out = self._root_fn(logits.to(device), *inner.tree_state())
        levels = [out[0]]
        base = device_tensor(request.ends, device)
        for d in range(1, plan.depth):
            width = plan.frontier[d]
            if width == 0:
                break
            tokens, states = out[1], out[2:2 + n_states]
            forwarded = consts.forwarded[d].expand(batch, -1)
            out = self._level_fn(tokens, forwarded, base, *states)
            for b, cost in enumerate(request.costs):
                cost.record_footprint(CostKind.DRAFT, width, request.ends[b] + d - 1, footprint, step=d)
            levels.append(out[0])
        tokens = torch.cat(levels, 1) if len(levels) > 1 else levels[0]
        return Draft(tokens.clone(), [self.tree.size] * batch, consts.parents.expand(batch, -1).clone())

    def _root(self, logits: torch.Tensor, *states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The tokens of depth 1 ``[B, n_1]`` and, when depth 1 has nodes with children, their tokens and the root's
        states per such node."""
        plan = self._plan
        consts = self._const(logits.device)
        tokens = logits.float().topk(plan.width[0], -1).indices.index_select(1, consts.root_pick)
        if plan.frontier[1] == 0:
            return (tokens,)
        frontier = tokens.index_select(1, consts.next_pick[1])
        w = plan.frontier[1]
        expanded = tuple(s[:, None].expand(s.shape[0], w, *s.shape[1:]).contiguous() for s in states)
        return (tokens, frontier, *expanded)

    def _level(self, frontier_tokens: torch.Tensor, forwarded: torch.Tensor, base: torch.Tensor,
               *states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The draft step of the frontier of one depth ``d`` (found from the number of forwarded nodes): the tokens of
        depth ``d + 1`` and, when that depth has nodes with children, their tokens and states."""
        plan = self._plan
        d = plan.depth_of_forwarded[forwarded.shape[1]]
        logits, out = self.inner.tree_step(frontier_tokens, forwarded, base, *states)  # type: ignore[attr-defined]
        consts = self._const(frontier_tokens.device)
        here = frontier_tokens.device
        top = logits.float().topk(plan.width[d], -1).indices.to(here)
        tokens = top.reshape(top.shape[0], -1).index_select(1, consts.child_pick[d])
        if d + 1 >= plan.depth or plan.frontier[d + 1] == 0:
            return (tokens,)
        nxt = tokens.index_select(1, consts.next_pick[d + 1])
        origin = consts.next_origin[d + 1]
        return (tokens, nxt, *(s.index_select(1, origin.to(s.device)) for s in out))

    # ---- acceptance statistics --------------------------------------------------------------------------------
    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None:
        if self.record_acceptance:
            self._record(exact, columns)
        super().observe(exact, columns)

    def _record(self, exact: ExactPass, columns: torch.Tensor) -> None:
        ctx = exact.context
        if ctx is None or ctx.tree is None:
            return
        tree = ctx.tree
        rows = torch.arange(tree.visible.shape[0], device=tree.visible.device)
        path = tree.visible[rows, columns.to(tree.visible.device)].cpu()
        children = self.tree.children()
        size = min(self.tree.size, path.shape[1] - 1)
        for b in range(path.shape[0]):
            accepted = {i for i in range(size) if bool(path[b, i + 1])}
            node, level = -1, 0
            while True:
                kids = [c for c in children[node] if c < size]
                nxt = next((c for c in kids if c in accepted), None)
                self.acceptance.add(level, len(kids), None if nxt is None else self.tree.paths[nxt][-1])
                if nxt is None:
                    break
                node, level = nxt, level + 1


class _Plan:
    """The per-depth bookkeeping of a static tree's proposal (host side)."""

    def __init__(self, tree: StaticTree) -> None:
        paths, parents = tree.paths, tree.parents
        children = tree.children()
        self.depth = tree.depth
        by_depth: list[list[int]] = [[] for _ in range(tree.depth + 1)]
        for i, p in enumerate(paths):
            by_depth[len(p)].append(i)
        # frontier[d]: nodes of depth d with children (forwarded at step d); width[d]: top-k of the step of depth d
        internal = [[i for i in by_depth[d] if children[i]] for d in range(tree.depth + 1)]
        self.frontier = [len(nodes) for nodes in internal] + [0]
        self.root_pick = [paths[i][-1] for i in by_depth[1]]
        self.width = [max(self.root_pick) + 1] + [max((paths[c][-1] + 1 for i in internal[d] for c in children[i]),
                                               default=0) for d in range(1, tree.depth + 1)]
        level_pos = {i: by_depth[len(paths[i])].index(i) for i in range(tree.size)}
        frontier_pos = {i: internal[len(paths[i])].index(i) for d in range(1, tree.depth + 1) for i in internal[d]}
        # child_pick[d]: flat index (frontier position * width[d] + rank) of every node of depth d + 1
        self.child_pick = {d: [frontier_pos[parents[i]] * self.width[d] + paths[i][-1] for i in by_depth[d + 1]]
                           for d in range(1, tree.depth)}
        # next_pick[d]: positions among the nodes of depth d of the frontier of depth d; next_origin[d]: its parents'
        # positions in the frontier of depth d - 1
        self.next_pick = {d: [level_pos[i] for i in internal[d]] for d in range(1, tree.depth)}
        self.next_origin = {d: [frontier_pos[parents[i]] for i in internal[d]] for d in range(2, tree.depth)}
        order = [i for d in range(1, tree.depth) for i in internal[d]]
        place = {i: k for k, i in enumerate(order)}
        fwd_parents = [place[parents[i]] if parents[i] >= 0 else -1 for i in order]
        self.forwarded: dict[int, list[int]] = {}
        self.depth_of_forwarded: dict[int, int] = {}
        count = 0
        for d in range(1, tree.depth):
            count += len(internal[d])
            if internal[d]:
                self.forwarded[d] = fwd_parents[:count]
                self.depth_of_forwarded[count] = d
        self.parents = list(parents)


class _Consts:
    """The device tensors of a :class:`_Plan` (built once per device, so graphs read them at fixed addresses)."""

    def __init__(self, plan: _Plan, device: torch.device) -> None:
        def make(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.long, device=device)

        self.root_pick = make(plan.root_pick)
        self.child_pick = {d: make(v) for d, v in plan.child_pick.items()}
        self.next_pick = {d: make(v) for d, v in plan.next_pick.items()}
        self.next_origin = {d: make(v) for d, v in plan.next_origin.items()}
        self.forwarded = {d: make(v)[None] for d, v in plan.forwarded.items()}
        self.parents = make(plan.parents)[None]


def measure_acceptance(model: Any, prompts: Sequence[torch.Tensor], max_new_tokens: int, *, depth: int = 6,
                       width: int = 8, per_depth: bool = False,
                       decoding: Mapping[str, Any] | None = None) -> tuple[float, ...] | tuple[tuple[float, ...], ...]:
    """The acceptance vector of ``model``'s drafter on ``prompts``, for the ``acceptance`` option of ``sequoia``.

    Decodes the prompts with the probe tree of :func:`probe_paths` (``depth`` deep, ``width`` children per spine node)
    and counts at every accepted node which of the drafter's ranks the exact token had.

    Args:
        model: An :class:`~economize.EfficientModel` with a drafter that supports tree steps.
        prompts: Prompt token ids.
        max_new_tokens: Tokens generated per prompt.
        depth: Depth of the probe tree.
        width: Ranks measured per node.
        per_depth: Return one vector per depth of the parent node.
        decoding: Further :class:`~economize.DecodeConfig` fields (for example ``cuda_graphs``).
    """
    from .config import DecodeConfig

    config = DecodeConfig(cycle_length=depth + 1, draft_policy="sequoia",
                          draft_policy_options={"paths": probe_paths(depth, width), "record_acceptance": True},
                          **dict(decoding or {}))
    decoder = model.decoder(config)
    drafter = decoder.active_drafter
    if not isinstance(drafter, SequoiaDrafter):
        raise ConfigError("measure_acceptance needs a model with a drafting technique")
    drafter.acceptance = AcceptanceProfile()
    decoder.decode_batch(list(prompts), max_new_tokens)
    return drafter.acceptance.rates(per_depth)
