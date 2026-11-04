"""SpecExec draft trees: best-first parallel search over the drafter's continuations.

``DecodeConfig(draft_policy="specexec", draft_policy_options={"depth": 8, "width": 8, "topk": 8,
"total_tokens": 60, "decay": 0.95})`` turns a chain drafter into the tree search of SpecExec (Svirschevski et al., 2024,
"SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices"). A proposal
keeps a pool of candidates, each scored by the cumulative log-probability the drafter gives its path from the root
plus ``log(decay)`` per depth (the official code's ``decay_factor``, which discounts deep nodes for the chance that an
ancestor is rejected):

1. The root's draft step adds the root's ``topk`` most likely tokens.
2. Each of ``depth - 1`` search steps takes the ``width`` candidates of the pool with the highest cumulative
   log-probability that have not been expanded yet, whatever their depth, runs them through one batched draft step
   under a tree mask, and adds each one's ``topk`` most likely children to the pool.
3. The ``total_tokens`` candidates with the highest score form the draft. A child never scores higher than its
   parent, so the selection is closed under ancestors.

The official implementation (``specdec/specexec.py``, version 4b) also stops the search early once no new candidate
reaches the current top ``total_tokens``, and drops such candidates while it grows. Both only skip work whose result
the final selection would discard. economize runs all ``depth - 1`` steps, so the shapes of a cycle stay fixed and the
search runs under CUDA graphs.

Unlike the level-by-level growth of ``dynamic_tree`` (EAGLE-2), a search step may expand a shallow and a deep
candidate together, so a confident drafter grows long branches and an unsure one grows wide ones. The exact pass
verifies the tree as any draft tree, so decoding stays lossless under every verifier that compares draft tokens with
the exact model's tokens.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ..errors import ConfigError
from ..runtime.transfer import device_tensor
from .cost import CostKind
from .policies import POLICIES
from .protocols import Draft, Drafter, Verifier, draft_footprint
from .tree_search import LOGP_FLOOR, TreePolicy, TreeSearchDrafter, select_candidates, weak_method

if TYPE_CHECKING:
    from .config import DecodeConfig
    from .request import DraftRequest

__all__ = ["SearchOptions", "SpecExecDrafter", "SpecExecPolicy"]


@dataclass(frozen=True)
class SearchOptions:
    """The budget of a SpecExec tree search.

    Attributes:
        depth: Largest depth of a draft node; the search runs ``depth - 1`` steps after the root's.
        width: Candidates expanded per search step (at most ``topk``).
        topk: Children every expanded candidate adds to the pool.
        total_tokens: Nodes of the draft tree the exact pass verifies.
        decay: Factor in ``(0, 1]`` on the probability of every further depth (``decay_factor`` of the official
            code, which defaults to 0.95; 1 ranks by the drafter's probabilities alone).
    """

    depth: int
    width: int
    topk: int
    total_tokens: int
    decay: float = 0.95

    @property
    def steps(self) -> int:
        """Search steps after the root's draft step."""
        return self.depth - 1

    @property
    def candidates(self) -> int:
        """Size of the pool after the search: ``topk + steps * width * topk``."""
        return self.topk + self.steps * self.width * self.topk

    @classmethod
    def from_config(cls, config: DecodeConfig) -> SearchOptions:
        """The options in ``config.draft_policy_options`` (``depth`` defaults to ``cycle_length - 1``)."""
        options: Mapping[str, Any] = config.draft_policy_options
        names = ("depth", "width", "topk", "total_tokens", "decay")
        unknown = sorted(set(options) - set(names))
        if unknown:
            raise ConfigError(f"draft_policy 'specexec' has no options {unknown}; it takes {', '.join(names)}")
        values = {"depth": max(config.cycle_length - 1, 1), "width": 8, "topk": 8, "total_tokens": 60, **options}
        decay = values.pop("decay", 0.95)
        if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not 0.0 < decay <= 1.0:
            raise ConfigError(f"draft_policy 'specexec' needs a decay in (0, 1], got {decay!r}")
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(f"draft_policy 'specexec' needs a positive integer {name}, got {value!r}")
        search = cls(values["depth"], values["width"], values["topk"], values["total_tokens"], float(decay))
        if search.width > search.topk:
            raise ConfigError(f"width={search.width} exceeds topk={search.topk}: the first search step expands "
                              "the root's topk children only")
        if search.total_tokens > search.candidates:
            raise ConfigError(f"total_tokens={search.total_tokens} exceeds the {search.candidates} candidates of a "
                              f"search of depth {search.depth}, width {search.width} and topk {search.topk}")
        return search


@POLICIES.register("specexec")
class SpecExecPolicy(TreePolicy):
    """Drafts a SpecExec tree with any :class:`~economize.decoding.TreeStepDrafter` (see the module docstring).

    Options (``DecodeConfig.draft_policy_options``): ``depth`` (default ``cycle_length - 1``), ``width`` (default 8),
    ``topk`` (default 8), ``total_tokens`` (default 60) and ``decay`` (default 0.95).
    """

    policy_name = "specexec"

    def __init__(self, config: DecodeConfig) -> None:
        super().__init__(config)
        self.options = SearchOptions.from_config(config)

    def wrap(self, drafter: Drafter, verifier: Verifier) -> Drafter:
        self.check_drafter(drafter, verifier)
        return SpecExecDrafter(drafter, self.options)


class SpecExecDrafter(TreeSearchDrafter):
    """A chain :class:`~economize.decoding.TreeStepDrafter` that proposes SpecExec trees (made by
    :class:`SpecExecPolicy`).

    The cost log records the root's step as one ``draft`` call of one token and search step ``s`` as one call of
    ``width`` tokens at the position of depth ``s`` (the deepest a node of that step can be), both over the inner
    drafter's footprint. With CUDA graphs the root's expansion and every search step run from graphs (one per step:
    the pool grows by a fixed amount per step), and so does the final selection.
    """

    def __init__(self, inner: Drafter, options: SearchOptions) -> None:
        super().__init__(inner, "specexec")
        self.options = options
        self._root_fn: Callable[..., tuple[torch.Tensor, ...]] = self._root
        self._search_fn: Callable[..., tuple[torch.Tensor, ...]] = self._search
        self._select_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = self._select

    @property
    def max_depth(self) -> int:
        return self.options.depth

    def draft_capacity(self, cycle_length: int) -> int:
        return max(self.options.total_tokens, self.options.steps * self.options.width)

    def enable_graphs(self, wrap: Callable[[Callable[..., Any]], Callable[..., Any]]) -> bool:
        if not self.inner.enable_graphs(wrap):
            return False
        self._root_fn = wrap(weak_method(self._root))
        self._search_fn = wrap(weak_method(self._search))
        self._select_fn = wrap(weak_method(self._select))
        return True

    # ---- drafting ---------------------------------------------------------------------------------------------
    def propose(self, request: DraftRequest) -> Draft:
        inner: Any = self.inner
        width, steps = self.options.width, self.options.steps
        batch, device = request.batch_size, request.device
        footprint = draft_footprint(inner)
        inner.tree_begin(request)
        logits = inner.step(request.tokens, request.positions, None)
        for b, cost in enumerate(request.costs):
            cost.record_footprint(CostKind.DRAFT, 1, request.ends[b] - 1, footprint, step=0)
        n_states = len(inner.tree_state())
        pool = self._root_fn(logits.to(device), *inner.tree_state())
        forwarded = torch.empty(batch, 0, dtype=torch.long, device=device)
        base = device_tensor(request.ends, device)
        for s in range(1, steps + 1):
            out = self._search_fn(*pool[:5], forwarded, base, *pool[5:5 + n_states])
            for b, cost in enumerate(request.costs):
                cost.record_footprint(CostKind.DRAFT, width, request.ends[b] + s - 1, footprint, step=s)
            pool, forwarded = out[:5] + out[6:], out[5]
        chosen, parents = self._select_fn(*pool[:4])
        return Draft(chosen.clone(), [self.options.total_tokens] * batch, parents.clone())

    def _root(self, logits: torch.Tensor, *states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The pool after the root's step: tokens, scores, parent candidates, depths, the forwarded index of every
        candidate (``-1``: not expanded) ``[B, topk]``, and the root's states ``[B, 1, ...]`` (the state buffers: the
        root's, then every forwarded node's)."""
        k = self.options.topk
        top = torch.log_softmax(logits.float(), -1).clamp_min(LOGP_FLOOR).topk(k, -1)
        batch, device = top.indices.shape[0], top.indices.device
        scores = top.values + math.log(self.options.decay)
        parents = torch.full((batch, k), -1, dtype=torch.long, device=device)
        depth = torch.ones(batch, k, dtype=torch.long, device=device)
        fwd = torch.full((batch, k), -1, dtype=torch.long, device=device)
        return (top.indices, scores, parents, depth, fwd, *(s[:, None] for s in states))

    def _search(self, tokens: torch.Tensor, scores: torch.Tensor, parents: torch.Tensor, depth: torch.Tensor,
                fwd: torch.Tensor, forwarded: torch.Tensor, base: torch.Tensor,
                *states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """One search step on the pool: expand the ``width`` best unexpanded candidates and add their children.
        Returns the grown pool in the order of :meth:`_root` (the parents of all forwarded nodes after its first five
        tensors, the grown state buffers last)."""
        k, width = self.options.topk, self.options.width
        batch = tokens.shape[0]
        rows = torch.arange(batch, device=tokens.device)[:, None]
        step = forwarded.shape[1] // width  # search steps so far
        pick = torch.where(fwd >= 0, float("-inf"), scores).topk(width, -1).indices  # [B, W]
        pick_parent = parents.gather(1, pick)
        fwd_parent = torch.where(pick_parent >= 0, fwd.gather(1, pick_parent.clamp_min(0)), -1)
        forwarded = torch.cat([forwarded, fwd_parent], 1)
        frontier_states = [s[rows, fwd_parent + 1] for s in states]
        logits, out = self.inner.tree_step(tokens.gather(1, pick), forwarded, base,  # type: ignore[attr-defined]
                                           *frontier_states)
        top = torch.log_softmax(logits.float(), -1).clamp_min(LOGP_FLOOR).topk(k, -1)
        here = scores.device
        child_logp = top.values.to(here) + math.log(self.options.decay)
        child_scores = (scores.gather(1, pick)[:, :, None] + child_logp).reshape(batch, width * k)
        child_tokens = top.indices.to(here).reshape(batch, width * k)
        child_parents = pick[:, :, None].expand(batch, width, k).reshape(batch, width * k)
        child_depth = (depth.gather(1, pick) + 1)[:, :, None].expand(batch, width, k).reshape(batch, width * k)
        index = step * width + torch.arange(width, device=here).expand(batch, width)
        fwd = fwd.scatter(1, pick, index)
        unexpanded = torch.full((batch, width * k), -1, dtype=torch.long, device=here)
        grown = [torch.cat([s, o.to(s.device)], 1) for s, o in zip(states, out)]
        return (torch.cat([tokens, child_tokens], 1), torch.cat([scores, child_scores], 1),
                torch.cat([parents, child_parents], 1), torch.cat([depth, child_depth], 1),
                torch.cat([fwd, unexpanded], 1), forwarded, *grown)

    def _select(self, tokens: torch.Tensor, scores: torch.Tensor, parents: torch.Tensor,
                depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``total_tokens`` best candidates of the pool as a draft tree in topological order."""
        return select_candidates(tokens, scores, parents, depth, self.options.total_tokens)
