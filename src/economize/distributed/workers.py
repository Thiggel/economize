"""The two sides of speculative speculative decoding: the target that verifies and the drafter that speculates.

A :class:`SpeculationWorker` owns a drafter on its own device. Besides drafting a chain of tokens it can
*speculate*: while the target verifies a chain, it drafts the next chain for the likeliest verification outcomes in
one tree pass per depth, so that a hit needs no further drafter work. :class:`ModelWorker` implements this for any
causal LM economize has an architecture adapter for. Other drafters (for example a jump-decoded draft model) are added
by subclassing :class:`SpeculationWorker`; the decoder only uses its four methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from ..arch import adapter_for
from ..decoding.cost import CostKind, CostLog
from ..errors import ConfigError
from ..runtime import LayerRunner
from ..runtime.cache import KVCache
from ..runtime.context import LayerContext
from ..runtime.mask import TreeLayout, tree_ancestors
from ..validation import check_positive
from .tp import is_tensor_parallel

__all__ = [
    "Adoption",
    "Chooser",
    "Drafts",
    "ModelWorker",
    "Speculation",
    "SpeculationWorker",
    "Spine",
    "TargetWorker",
    "TreeTemplate",
    "TreeTemplates",
    "outcome_candidates",
    "to_host",
]


class _CachedModel:
    """A model on its device with one KV cache and a cost log: the passes the target and :class:`ModelWorker` share.

    Every pass is logged as one call of all the model's layers under the subclass's :attr:`kind`.
    """

    #: The cost-event kind of every pass.
    kind: str

    def __init__(self, model: nn.Module) -> None:
        if is_tensor_parallel(model):
            raise ConfigError("asynchronous speculative decoding does not run tensor-parallel models yet; "
                              "load the target and the drafter on one device each")
        self.runner = LayerRunner(adapter_for(model))
        self.adapter = self.runner.adapter
        self.device = self.adapter.device
        self.num_layers = self.adapter.num_layers
        self.cost = CostLog()
        self.cache: KVCache | None = None

    def _reset(self, capacity: int) -> None:
        """A fresh cache of ``capacity`` positions and an empty cost log."""
        self.cache = self.runner.new_cache(capacity)
        self.cost = CostLog()

    def _chain(self, tokens: torch.Tensor, position: int) -> torch.Tensor:
        """Logits ``[S, V]`` after each of ``tokens`` ``[S]``, the first at ``position`` (writes their K/V)."""
        assert self.cache is not None
        positions = torch.arange(position, position + tokens.numel(), device=self.device)
        return self._run(tokens, position, self.runner.context(self.cache, positions, trusted=True))

    def _tree(self, tree: TreeLayout, nodes: torch.Tensor, valid: torch.Tensor, tokens: torch.Tensor,
              base: int) -> torch.Tensor:
        """Logits ``[S, V]`` after the tree nodes ``nodes`` (holding ``tokens`` ``[S]``) of ``tree``."""
        assert self.cache is not None
        return self._run(tokens, base, self.runner.tree_context(self.cache, tree, trusted=True).select(nodes, valid))

    def _run(self, tokens: torch.Tensor, position: int, ctx: LayerContext) -> torch.Tensor:
        self.cost.record(self.kind, tokens.numel(), position, 1, self.num_layers)
        return self.runner.logits(self.runner.forward(tokens.view(1, -1), ctx).final)[0]


class Chooser(Protocol):
    """Picks one token per row of logits ``[N, V]``: returns the tokens ``[N]`` and, when sampling, the
    distributions they were drawn from ``[N, V]`` (``None`` for deterministic choices)."""

    def __call__(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]: ...


@dataclass
class Spine:
    """The chain a round verifies: draft tokens and what the drafter chose them from.

    Attributes:
        tokens: Draft tokens ``[k]`` on the drafter's device.
        logits: Drafter logits ``[k, V]``; row ``i`` produced ``tokens[i]``.
        probs: Distributions the tokens were sampled from ``[k, V]``, ``None`` for greedy drafts.
    """

    tokens: torch.Tensor
    logits: torch.Tensor
    probs: torch.Tensor | None = None


@dataclass
class Speculation:
    """The drafter's preparation for the outcomes of one verification.

    An outcome is ``(a, y)``: the target accepts the first ``a`` drafts and appends ``y``. For every ``a`` in
    ``0..k`` the drafter guesses ``fanout`` candidates for ``y`` and drafts the next chain behind each of them.

    Attributes:
        base: Committed length when the round started (the spine's first draft sits at position ``base``).
        candidates: Guessed tokens ``y`` per outcome ``[k + 1, fanout]`` (device).
        tokens: The next chain behind every candidate ``[k + 1, fanout, k]``.
        logits: Its drafter logits ``[k + 1, fanout, k, V]``.
        probs: Its sampling distributions, or ``None`` for greedy drafts.
        ready: Host copy of ``candidates``; read it after ``event`` completed.
        event: Marks when ``ready`` holds the candidates (``None`` on the CPU).
    """

    base: int
    candidates: torch.Tensor
    tokens: torch.Tensor
    logits: torch.Tensor
    probs: torch.Tensor | None
    ready: torch.Tensor
    event: torch.cuda.Event | None = None

    def lookup(self, accepted: int, token: int) -> int | None:
        """Index of ``token`` among the candidates of outcome ``accepted``, or ``None`` (a miss)."""
        if self.event is not None:
            self.event.synchronize()
        row = self.ready[accepted].tolist()
        return row.index(token) if token in row else None

    def hit(self, accepted: int, token: int) -> Adoption | None:
        """The branch behind the outcome ``(accepted, token)``, or ``None`` if it was not speculated (a miss)."""
        index = self.lookup(accepted, token)
        if index is None:
            return None
        length = self.tokens.shape[-1]
        branch = accepted * self.candidates.shape[1] + index
        probs = None if self.probs is None else self.probs[accepted, index]
        spine = Spine(self.tokens[accepted, index], self.logits[accepted, index], probs)
        return Adoption(branch, self.base + length + branch * length, self.base + accepted, length, spine)


@dataclass(frozen=True)
class Adoption:
    """A speculated outcome that happened: the branch of the speculation tree (see :class:`TreeTemplate`) that
    becomes the committed sequence's continuation.

    Attributes:
        branch: Index of the branch, ``accepted * fanout + candidate``.
        source: Cache slot of the branch's first node.
        destination: Position of the appended token; the branch's nodes move to
            ``destination .. destination + length - 1``.
        length: Nodes of the branch (the draft length).
        spine: The chain drafted behind the appended token, the next round's spine.
    """

    branch: int
    source: int
    destination: int
    length: int
    spine: Spine

    def gather(self, cache: KVCache, device: torch.device) -> None:
        """Move the branch's K/V in ``cache`` from its tree slots to its sequence positions."""
        src = torch.arange(self.source, self.source + self.length, device=device)[None]
        dst = torch.arange(self.destination, self.destination + self.length, device=device)[None]
        cache.gather(torch.zeros(1, dtype=torch.long, device=device), src, dst)


class Drafts:
    """The steps of a chain, or the depths of a speculation tree, collected one at a time and stacked at the end.

    Every step adds the chosen tokens, the logits they were chosen from and their sampling distributions. The chooser
    returns a distribution for every step (sampling) or for none (greedy).
    """

    def __init__(self) -> None:
        self.tokens: list[torch.Tensor] = []
        self.logits: list[torch.Tensor] = []
        self.probs: list[torch.Tensor] = []

    def add(self, tokens: torch.Tensor, logits: torch.Tensor, probs: torch.Tensor | None) -> None:
        """One step: tokens ``[S]``, the logits ``[S, V]`` and the distributions ``[S, V]`` or ``None``."""
        self.tokens.append(tokens)
        self.logits.append(logits)
        if probs is not None:
            self.probs.append(probs)

    def spine(self) -> Spine:
        """The steps of a chain (one token each) as a :class:`Spine`."""
        return Spine(torch.cat(self.tokens), torch.cat(self.logits), torch.cat(self.probs) if self.probs else None)

    def speculation(self, base: int, candidates: torch.Tensor, ready: torch.Tensor,
                    event: torch.cuda.Event | None) -> Speculation:
        """The depths of a speculation tree behind ``candidates`` ``[k + 1, fanout]`` (one token per branch each) as
        a :class:`Speculation`; ``ready`` and ``event`` as returned by :func:`outcome_candidates`."""
        shape = (*candidates.shape, len(self.tokens))
        return Speculation(
            base, candidates,
            torch.stack(self.tokens, -1).view(shape),
            torch.stack(self.logits, 1).view(*shape, -1),
            torch.stack(self.probs, 1).view(*shape, -1) if self.probs else None,
            ready, event,
        )


def to_host(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.cuda.Event | None]:
    """An asynchronous host copy of ``tensor`` and the event that marks its arrival (``tensor`` itself and ``None``
    off CUDA)."""
    if tensor.device.type != "cuda":
        return tensor, None
    ready = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
    ready.copy_(tensor, non_blocking=True)
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(tensor.device))
    return ready, event


def outcome_candidates(spine: Spine, last: torch.Tensor,
                       fanout: int) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
    """The ``fanout`` likeliest appended tokens of every verification outcome of ``spine``.

    Outcome ``a < k`` rejects ``d_{a+1}`` and appends another token, ranked by the drafter's logits after ``d_a``.
    Outcome ``k`` keeps every draft and is ranked by ``last`` ``[1, V]``, the drafter's logits after ``d_k``.

    Returns:
        The candidates ``[k + 1, fanout]`` on the drafter's device, their host copy and its event (see
        :func:`to_host`).
    """
    k = spine.tokens.numel()
    scores = torch.cat([spine.logits, last]).float()
    rejected = torch.arange(k, device=scores.device)
    scores[rejected, spine.tokens] = float("-inf")  # outcome a < k appends a token other than d_{a+1}
    candidates = scores.topk(fanout, dim=-1).indices
    ready, event = to_host(candidates)
    return candidates, ready, event


class SpeculationWorker(ABC):
    """A drafter that drafts chains and speculates on verification outcomes.

    The worker keeps its own cache. Between rounds it holds the committed prefix up to the last committed token
    (the *root*, at position ``n - 1``) plus the spine's drafts ``d_1..d_{k-1}`` at positions ``n..n+k-2``.
    """

    #: Device the worker computes on.
    device: torch.device
    #: Every model call the worker made, in its own layer units.
    cost: CostLog
    #: Guessed tokens per verification outcome (read by the default :meth:`capacity_overhead`).
    fanout: int = 1

    @abstractmethod
    def begin(self, prompt: torch.Tensor, capacity: int) -> None:
        """Start a generation: a fresh cache of ``capacity`` positions and a pass over ``prompt`` ``[P]``."""

    @abstractmethod
    def draft(self, context: torch.Tensor, position: int, length: int, choose: Chooser) -> Spine:
        """Draft ``length`` tokens one at a time after ``context`` ``[S]``: committed tokens missing from the cache,
        the first at ``position``, the last the root."""

    @abstractmethod
    def speculate(self, spine: Spine, base: int, choose: Chooser) -> Speculation:
        """Prepare the next chain for the likeliest outcomes of verifying ``spine`` (first draft at ``base``).

        Must not wait for the device, so that it overlaps the target's verification.
        """

    @abstractmethod
    def adopt(self, speculation: Speculation, accepted: int, token: int) -> Spine | None:
        """The next spine if the outcome ``(accepted, token)`` was speculated (then the cache holds its tokens at
        their positions), else ``None``."""

    def capacity_overhead(self, length: int) -> int:
        """Cache positions beyond the generated sequence that one speculation of chains of ``length`` drafts needs.

        The default fits the layout of :class:`ModelWorker`: the spine plus ``(length + 1) * fanout`` chains of
        ``length`` nodes. A worker that lays out its speculation differently overrides it.
        """
        return length + (length + 1) * self.fanout * length + 2


@dataclass(frozen=True)
class TreeTemplate:
    """Node layout of a speculation tree (relative to its base), built once per ``(k, fanout)``.

    Nodes ``0..k-1`` are the spine ``d_1..d_k``. Branch ``b = a * fanout + c`` (outcome ``a``, candidate ``c``)
    holds ``k`` nodes from ``k + b * k`` on: the candidate ``y`` (child of ``d_a``, or of the root when ``a = 0``)
    and the next chain's drafts ``e_1..e_{k-1}`` below it.
    """

    length: int
    fanout: int
    offsets: torch.Tensor  # [1, T]
    depth: torch.Tensor  # [1, T]
    visible: torch.Tensor  # [1, T, T]
    zero: torch.Tensor  # [1]
    last: torch.Tensor  # [1, 1] the spine's last node
    levels: tuple[torch.Tensor, ...]  # [1, B] node j of every branch, j = 0..k-1
    valid: torch.Tensor  # [1, B] all true

    @classmethod
    def build(cls, length: int, fanout: int, device: torch.device) -> TreeTemplate:
        branches = (length + 1) * fanout
        parents = list(range(-1, length - 1))
        for b in range(branches):
            outcome = b // fanout
            start = length + b * length
            parents.append(outcome - 1)
            parents.extend(range(start, start + length - 1))
        table = torch.tensor([parents], device=device)
        reach = tree_ancestors(table)
        starts = length + torch.arange(branches, device=device) * length
        return cls(
            length, fanout,
            offsets=torch.arange(len(parents), device=device)[None],
            depth=reach.sum(-1) - 1,
            visible=reach,
            zero=torch.zeros(1, dtype=torch.long, device=device),
            last=torch.full((1, 1), length - 1, device=device),
            levels=tuple((starts + j)[None] for j in range(length)),
            valid=torch.ones(1, branches, dtype=torch.bool, device=device),
        )

    def layout(self, base: int) -> TreeLayout:
        """The tree with its first node at cache slot ``base``."""
        return TreeLayout(self.zero + base, self.offsets + base, self.depth + base, self.visible)


class TreeTemplates:
    """The :class:`TreeTemplate` of every draft length a worker speculates with, each built on first use."""

    def __init__(self, fanout: int, device: torch.device) -> None:
        self.fanout, self.device = fanout, device
        self._built: dict[int, TreeTemplate] = {}

    def __call__(self, length: int) -> TreeTemplate:
        template = self._built.get(length)
        if template is None:
            template = self._built[length] = TreeTemplate.build(length, self.fanout, self.device)
        return template


class ModelWorker(_CachedModel, SpeculationWorker):
    """A causal LM drafter on its own device (any model with an economize architecture adapter and attention layers).

    Args:
        model: The draft model, already on its device (e.g. ``cuda:1``) and in eval mode.
        fanout: Candidates for the appended token per outcome. Every speculation drafts ``(k + 1) * fanout``
            chains in parallel, one tree pass per depth.
    """

    kind = CostKind.DRAFT

    def __init__(self, model: nn.Module, fanout: int = 2) -> None:
        check_positive("fanout", fanout)
        super().__init__(model)
        if self.runner.has_recurrent_layers:
            raise ConfigError("speculation drafts trees; a drafter with recurrent layers cannot run them")
        self.fanout = fanout
        self._templates = TreeTemplates(fanout, self.device)

    # ---- SpeculationWorker ---------------------------------------------------------------------------------------
    @torch.no_grad()
    def begin(self, prompt: torch.Tensor, capacity: int) -> None:
        self._reset(capacity)
        self._chain(prompt.to(self.device), 0)

    @torch.no_grad()
    def draft(self, context: torch.Tensor, position: int, length: int, choose: Chooser) -> Spine:
        token = context.to(self.device).view(-1)
        drafts = Drafts()
        for _ in range(length):
            row = self._chain(token, position)[-1:]
            position += token.numel()
            token, dist = choose(row)
            drafts.add(token, row, dist)
        return drafts.spine()

    @torch.no_grad()
    def speculate(self, spine: Spine, base: int, choose: Chooser) -> Speculation:
        assert self.cache is not None
        template = self._templates(spine.tokens.numel())
        tree = template.layout(base)
        # Depth k of the spine: the distribution after the last draft (the outcome "all accepted").
        last = self._tree(tree, template.last, template.valid[:, :1], spine.tokens[-1:], base)
        candidates, ready, event = outcome_candidates(spine, last, self.fanout)
        drafts = Drafts()
        current = candidates.reshape(-1)
        for nodes in template.levels:
            rows = self._tree(tree, nodes, template.valid, current, base)
            current, dist = choose(rows)
            drafts.add(current, rows, dist)
        return drafts.speculation(base, candidates, ready, event)

    @torch.no_grad()
    def adopt(self, speculation: Speculation, accepted: int, token: int) -> Spine | None:
        assert self.cache is not None
        hit = speculation.hit(accepted, token)
        if hit is None:
            return None
        hit.gather(self.cache, self.device)
        return hit.spine


class TargetWorker(_CachedModel):
    """The target model on its device: one exact pass over the prompt, then one chain verification per round.

    Args:
        model: The target model, already on its device.
    """

    kind = CostKind.EXACT

    @torch.no_grad()
    def begin(self, prompt: torch.Tensor, capacity: int) -> torch.Tensor:
        """Fresh cache and a pass over ``prompt`` ``[P]``; returns the logits after the last prompt token ``[V]``."""
        self._reset(capacity)
        return self.verify(prompt.to(self.device), 0)[-1]

    @torch.no_grad()
    def verify(self, tokens: torch.Tensor, position: int) -> torch.Tensor:
        """Logits ``[S, V]`` after each of ``tokens`` ``[S]``, the first at ``position`` (writes their K/V)."""
        return self._chain(tokens, position)
