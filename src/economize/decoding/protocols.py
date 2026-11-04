"""Interfaces between the decoding loop and its pluggable parts.

The loop in :mod:`economize.decoding.loop` only talks to these interfaces:

* a :class:`Drafter` proposes tokens cheaply between exact passes (provided by a technique),
* a :class:`TokenSelector` turns logits into a token (greedy, sampling, Hugging Face logits processors),
* a :class:`Verifier` decides which drafts survive the exact pass,
* a :class:`DraftPolicy` decides when a row stops drafting within a cycle.

All of them work on a batch of ``B`` rows. Rows are ragged: each has its own prompt length, number of drafts and
stopping point.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import torch

from ..errors import ConfigError
from ..runtime.transfer import device_tensor
from .capabilities import selector_many
from .cost import LayerFootprint

if TYPE_CHECKING:
    from ..runtime import ExactPass, KVCache, LayerRunner
    from .config import DecodeConfig
    from .request import DraftRequest

__all__ = [
    "Draft",
    "DraftPolicy",
    "Drafter",
    "TokenSelector",
    "Verifier",
    "VerifyEvent",
    "VerifyObserver",
    "WrappableDrafter",
    "draft_footprint",
]


@dataclass
class Draft:
    """The tokens a drafter proposes for every row in one cycle.

    A chain draft lists the tokens that follow the last committed token in order. A tree draft lists nodes in
    topological order: ``parents[b, i]`` is the index of node ``i``'s parent, ``-1`` for a child of the last
    committed token.

    Attributes:
        tokens: Proposed tokens ``[B, T]`` on the model's device, left-aligned and padded with ``-1``.
        lengths: Number of proposed tokens (nodes) of every row; ``lengths[b] <= T``.
        parents: Parent node of every node ``[B, T]`` for a tree draft, ``None`` for a chain.
        probs: The distribution every token was drawn from ``[B, T, V]``, for verifiers that need it
            (:attr:`Verifier.needs_draft_probs`).
        confidence: The drafter's top probability at every token ``[B, T]`` (float32, ``nan`` where unknown), when
            :attr:`DecodeConfig.record_confidence` is set and the drafter drafts through
            :meth:`~economize.decoding.DraftRequest.extend`; ``None`` otherwise.
        logits: The drafter's float logits at every token ``[B, T, V]`` (float32), when
            :attr:`DecodeConfig.record_distributions` is set and the drafter drafts through
            :meth:`~economize.decoding.DraftRequest.extend`; ``None`` otherwise.
        candidates: For a tree draft, the number of leading nodes of every row the verifier may accept; the
            nodes after them (up to ``lengths[b]``) run in the exact pass for the drafter's own use (the Jacobi window
            of lookahead decoding, read back in :meth:`Drafter.observe_logits`) and are never accepted. They may
            hang anywhere in the tree; no candidate descends from one. ``None`` (the default) makes every node a
            candidate. See :meth:`verifiable`.
    """

    tokens: torch.Tensor
    lengths: list[int]
    parents: torch.Tensor | None = None
    probs: torch.Tensor | None = None
    confidence: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    candidates: list[int] | None = None

    @classmethod
    def empty(cls, batch_size: int, device: torch.device | str) -> Draft:
        """A draft without tokens."""
        return cls(torch.empty(batch_size, 0, dtype=torch.long, device=device), [0] * batch_size)

    @property
    def width(self) -> int:
        """``T``, the largest number of tokens of any row."""
        return self.tokens.shape[1]

    @property
    def is_tree(self) -> bool:
        return self.parents is not None

    def select(self, rows: Sequence[int]) -> Draft:
        """The drafts of the batch rows ``rows``, in this order."""
        index = device_tensor(list(rows), self.tokens.device)

        def pick(tensor: torch.Tensor | None) -> torch.Tensor | None:
            return None if tensor is None else tensor.index_select(0, index.to(tensor.device))

        lengths = [self.lengths[b] for b in rows]
        width = max(lengths, default=0)
        tokens = pick(self.tokens)
        assert tokens is not None
        candidates = None if self.candidates is None else [self.candidates[b] for b in rows]
        draft = Draft(tokens[:, :width], lengths, pick(self.parents), pick(self.probs), pick(self.confidence),
                      pick(self.logits), candidates)
        if draft.parents is not None:
            draft.parents = draft.parents[:, :width]
        if draft.probs is not None:
            draft.probs = draft.probs[:, :width]
        if draft.confidence is not None:
            draft.confidence = draft.confidence[:, :width]
        if draft.logits is not None:
            draft.logits = draft.logits[:, :width]
        return draft

    def verifiable(self) -> Draft:
        """The part of the draft the verifier checks: every row's first ``candidates[b]`` nodes (the whole draft
        when :attr:`candidates` is ``None``). Candidates come first in topological order, so the part is a tree of
        its own."""
        if self.candidates is None:
            return self
        width = max(self.candidates, default=0)

        def cut(tensor: torch.Tensor | None) -> torch.Tensor | None:
            return None if tensor is None else tensor[:, :width]

        tokens = cut(self.tokens)
        assert tokens is not None
        return Draft(tokens, list(self.candidates), cut(self.parents), cut(self.probs), cut(self.confidence),
                     cut(self.logits))


@dataclass(frozen=True)
class VerifyEvent:
    """What an observer registered with :meth:`economize.EfficientModel.add_verify_observer` receives after every exact
    pass, once per row.

    Attributes:
        row: Index of the row in the batch passed to ``decode``.
        start: Position of the first token of ``tokens``.
        tokens: The tokens of the exact pass that are now committed (pending tokens and accepted drafts), ``[n]``.
        next_token: The token the verifier chose after them (sampled from the exact logits); it is committed
            unless decoding stopped before it.
        drafted: Number of drafts this pass verified (``0`` for the prefill and for lossy decoding).
        accepted: Number of those drafts that were accepted.
        hidden: Exact hidden states of ``tokens`` at the drafter's layers, ``{layer: [n, d]}`` (``0`` is the
            embedding; empty without a drafter).
    """

    row: int
    start: int
    tokens: torch.Tensor
    next_token: int
    drafted: int
    accepted: int
    hidden: dict[int, torch.Tensor] = field(default_factory=dict)


#: Receives one :class:`VerifyEvent` per row after every exact pass.
VerifyObserver = Callable[[VerifyEvent], None]


@runtime_checkable
class TokenSelector(Protocol):
    """Chooses the next token from logits.

    Required:

    * ``greedy``: the choice is deterministic given the logits and the history,
    * ``__call__(logits[V], history[n]) -> int``.

    Optional capabilities (``select_many``, ``distribution``, ``distributions``, ``generator_for``) are protocols
    of their own in :mod:`economize.decoding.capabilities`; a selector gains one by defining the member. A plain
    ``generator`` attribute is still honoured when it lives on the logits' device.
    """

    greedy: bool

    def __call__(self, logits: torch.Tensor, history: torch.Tensor) -> int:
        """``logits`` ``[V]`` predict the token after ``history`` (all tokens so far, prompt included, ``[n]``)."""
        ...


class Drafter(ABC):
    """Proposes tokens between exact passes for a batch of rows.

    A drafter lives for one generation. The loop calls :meth:`begin` once, then, every cycle, :meth:`observe` with
    the exact pass, :meth:`observe_logits` with its logits and :meth:`propose` for new drafts. Rows that finish are
    dropped with :meth:`reorder`.

    Simple drafters implement :meth:`step` (one token per row) and inherit :meth:`propose`, which repeats it while
    the :class:`~economize.decoding.DraftRequest` wants more tokens. Drafters that make several tokens per call or
    trees override :meth:`propose`.

    Attributes:
        source_layers: Layers whose exact hidden states the drafter reads (``0`` is the token embedding).
        layer_span: First and last decoder layer one :meth:`step` runs, for cost accounting.
        footprint: The layers one :meth:`step` runs when :attr:`layer_span` cannot say it: layers that are not one
            range, or the layers of a separate network (``module``). ``None`` uses :attr:`layer_span`; read it
            through :func:`draft_footprint`.
        streams_prompt: Whether the drafter needs nothing of the prompt beyond what :meth:`observe` gives it for the
            last prompt token and the K/V it writes into the shared cache itself. Only such drafters allow a chunked
            prefill (:attr:`DecodeConfig.prefill_chunk <economize.decoding.DecodeConfig.prefill_chunk>`), whose
            exact pass holds the hidden states of the last chunk alone.
    """

    source_layers: tuple[int, ...] = ()
    layer_span: tuple[int, int] = (0, 0)
    footprint: LayerFootprint | None = None
    streams_prompt: bool = False

    @property
    def observed_layers(self) -> tuple[int, ...]:
        """Layers the exact pass captures for :meth:`observe` (``source_layers`` unless overridden)."""
        return tuple(self.source_layers)

    @abstractmethod
    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        """Called once per generation, before the first exact pass."""

    @abstractmethod
    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None:
        """Adopt the exact states of the last accepted token of every row.

        Args:
            exact: The latest exact pass (``exact.hidden`` holds :attr:`observed_layers`).
            columns: Column of each row's last accepted token in that pass, ``[B]`` on the model's device.
        """

    def step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        """Process ``tokens`` ``[B]`` at ``positions`` ``[B]`` and return float logits ``[B, V]`` for the next
        position. Rows with ``valid`` false (``[B]``, ``None`` when all are active) have stopped drafting; their
        output is ignored and they must not change the cache."""
        raise NotImplementedError(f"{type(self).__name__} must override step() or propose()")

    def propose(self, request: DraftRequest) -> Draft:
        """Draft tokens for every row, up to its budget in ``request``."""
        while request.drafting:
            request.extend(self.step(request.tokens, request.positions, request.valid))
        return request.draft()

    def observe_logits(self, logits: torch.Tensor) -> None:
        """Read the exact logits of the latest pass, after :meth:`observe` and before finished rows are dropped.

        ``logits`` is ``[B, T + 1, V]`` (float): for a tree pass entry ``0`` follows the root and entry ``i + 1``
        follows node ``i`` of the draft, non-candidate nodes included (see :attr:`Draft.candidates`). Lookahead
        decoding takes its next Jacobi window from it. The default ignores it.
        """

    def reorder(self, rows: torch.Tensor) -> None:
        """Keep the batch rows ``rows`` (a 1-D LongTensor) in this order. Drafters with per-row state override it."""

    def draft_capacity(self, cycle_length: int) -> int:
        """Most tokens (tree nodes) one :meth:`propose` call makes per row (the loop sizes its buffers with it)."""
        return cycle_length - 1

    def enable_graphs(self, wrap: Callable[[Callable[..., Any]], Callable[..., Any]]) -> bool:
        """Opt in to CUDA graphs (``DecodeConfig.cuda_graphs``); return whether the drafter did.

        The decoder calls this once, before the first generation. ``wrap(fn)`` turns a function of tensors into a
        callable that records ``fn`` as a CUDA graph per input signature and replays it (see
        :class:`~economize.runtime.StaticStep`). A drafter that opts in routes its draft steps through such a callable
        and keeps everything the step reads or writes besides its inputs at fixed addresses across calls: per-row
        state lives in persistent buffers that are updated in place, and the cache handed to :meth:`begin` is the
        same object in every generation. The default declines, and the drafter runs eagerly.
        """
        return False

    def drafts_trees(self, verifier: Verifier) -> bool:
        """Whether :meth:`propose` returns draft trees when ``verifier`` checks the drafts.

        The decoder rejects a tree drafter whose verifier cannot verify trees before it decodes, and wrappers that
        restrict chain drafts (FR-Spec) reject it before it drafts. The default drafts chains; tree drafters
        override it.
        """
        return False


def draft_footprint(drafter: Drafter | WrappableDrafter) -> LayerFootprint:
    """The layers one draft step of ``drafter`` runs: its :attr:`~Drafter.footprint`, else its
    :attr:`~Drafter.layer_span`."""
    footprint = getattr(drafter, "footprint", None)
    return footprint if footprint is not None else LayerFootprint(*drafter.layer_span)


@runtime_checkable
class WrappableDrafter(Protocol):
    """A drafter another drafter can wrap (FR-Spec's draft vocabulary, the fallback of hybrid drafting).

    Besides the members of :class:`Drafter`, which the wrapper forwards, a wrappable drafter promises two things
    the wrapper relies on. Every chain token it proposes goes through
    :meth:`~economize.decoding.DraftRequest.extend`, the only place a wrapper sees and can restrict the logits a token
    is chosen from; a drafter that writes tokens into the request's sequence itself (prompt lookup) cannot be
    wrapped. :meth:`drafts_trees` says truthfully whether it will return a tree, so a wrapper that handles chains
    only can refuse before drafting. FR-Spec checks both promises after every proposal and raises
    :class:`~economize.ConfigError` when one is broken. Every :class:`Drafter` has these members; the protocol names
    the contract a wrapper depends on.
    """

    source_layers: tuple[int, ...]
    layer_span: tuple[int, int]
    footprint: LayerFootprint | None

    @property
    def observed_layers(self) -> tuple[int, ...]: ...

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None: ...

    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None: ...

    def observe_logits(self, logits: torch.Tensor) -> None: ...

    def step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor: ...

    def propose(self, request: DraftRequest) -> Draft: ...

    def reorder(self, rows: torch.Tensor) -> None: ...

    def draft_capacity(self, cycle_length: int) -> int: ...

    def enable_graphs(self, wrap: Callable[[Callable[..., Any]], Callable[..., Any]]) -> bool: ...

    def drafts_trees(self, verifier: Verifier) -> bool: ...


class Verifier(ABC):
    """Turns one exact pass into accepted drafts plus one token chosen from the exact logits.

    Attributes:
        commits_drafts: Drafts become final as they are proposed (lossy decoding); the exact pass then sees no
            unverified drafts.
        requires_greedy: Verification is exact only for a deterministic selector.
        needs_draft_probs: :meth:`draft` returns the distribution each draft was drawn from, and :meth:`verify`
            reads it.
        supports_trees: :meth:`verify_tree` is implemented.
        default_options: The options the verifier takes from ``DecodeConfig.verifier_options``, with their
            defaults. :attr:`options` holds the merged values; an unknown option is a :class:`ConfigError`.
    """

    commits_drafts: ClassVar[bool] = False
    requires_greedy: ClassVar[bool] = False
    needs_draft_probs: ClassVar[bool] = False
    supports_trees: ClassVar[bool] = False
    default_options: ClassVar[Mapping[str, Any]] = {}

    def __init__(self, config: DecodeConfig) -> None:
        self.config = config
        unknown = set(config.verifier_options) - set(self.default_options)
        if unknown:
            raise ConfigError(
                f"unknown verifier_options {sorted(unknown)} for verifier {config.verifier_name!r}; "
                f"it takes {sorted(self.default_options)}"
            )
        self.options: dict[str, Any] = {**self.default_options, **config.verifier_options}

    def draft(
        self, logits: torch.Tensor, selector: TokenSelector, history: torch.Tensor
    ) -> tuple[int, torch.Tensor | None]:
        """The draft token after ``history`` given drafter logits ``[V]``, and the distribution it was drawn from
        (``None`` unless :attr:`needs_draft_probs`)."""
        return selector(logits, history), None

    def draft_many(
        self, logits: torch.Tensor, selector: TokenSelector
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        """:meth:`draft` for every row of ``logits`` ``[B, V]`` in one call: tokens ``[B]`` and their distributions
        ``[B, V]`` (or ``None``). Returns ``None`` when the selector needs the history; the caller then drafts row
        by row."""
        tokens = selector_many(selector, logits)
        return None if tokens is None else (tokens, None)

    @abstractmethod
    def verify(
        self,
        logits: torch.Tensor,
        drafts: list[int],
        probs: torch.Tensor | None,
        selector: TokenSelector,
        history: torch.Tensor,
        end: int,
    ) -> tuple[int, int]:
        """Check the ``k`` drafts of one row against exact ``logits`` ``[k + 1, V]``.

        Row ``i`` of ``logits`` predicts the token at position ``end + i``, whose history is ``history[:end + i]``
        (``history`` holds the drafts at their positions). ``probs`` ``[k, V]`` are the draft distributions when
        :attr:`needs_draft_probs`. Returns the number of accepted drafts ``a`` and the token that follows them.
        """

    def verify_batch(
        self,
        logits: torch.Tensor,
        draft: Draft,
        selector: TokenSelector,
        seq: torch.Tensor,
        ends: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """:meth:`verify` for a batch of chain drafts.

        Args:
            logits: ``[B, T + 1, V]``; row ``b`` uses its first ``draft.lengths[b] + 1`` entries.
            draft: The chain drafts.
            selector: The token selector.
            seq: All tokens so far ``[B, *]`` (drafts at their positions).
            ends: Number of committed tokens of every row.

        Returns:
            Accepted drafts ``[B]`` and the following token ``[B]`` (LongTensors on the logits' device). The default
            verifies row by row; subclasses vectorize it.
        """
        accepted, tokens = [], []
        drafts = draft.tokens.tolist()
        for b, k in enumerate(draft.lengths):
            probs = draft.probs[b, :k] if draft.probs is not None else None
            a, token = self.verify(logits[b, : k + 1], drafts[b][:k], probs, selector, seq[b], ends[b])
            accepted.append(a)
            tokens.append(token)
        device = logits.device
        return device_tensor(accepted, device), device_tensor(tokens, device)

    def verify_tree(
        self, logits: torch.Tensor, draft: Draft, selector: TokenSelector
    ) -> tuple[list[list[int]], list[int]]:
        """Verify tree drafts. ``logits`` ``[B, T + 1, V]``: entry ``0`` follows the last committed token, entry
        ``i + 1`` follows node ``i``. Returns, per row, the accepted path (node indices, root first) and the token
        after it."""
        raise NotImplementedError(f"verifier {type(self).__name__} does not verify trees")


class DraftPolicy(ABC):
    """Decides, after each draft step, which rows stop drafting.

    Attributes:
        may_stop: :meth:`stops` can return rows to stop. ``False`` lets the loop skip the call.
    """

    may_stop: ClassVar[bool] = True

    def __init__(self, config: DecodeConfig) -> None:
        self.config = config

    @abstractmethod
    def stops(self, index: int, logits: torch.Tensor) -> torch.Tensor | None:
        """Rows that stop after draft step ``index`` (counted from 0 within the cycle), given that step's logits
        ``[B, V]``: a bool tensor ``[B]``, or ``None`` for no row. A stop discards the step's proposal."""

    def wrap(self, drafter: Drafter, verifier: Verifier) -> Drafter:
        """The drafter the decoder drafts with under this policy, given the technique's ``drafter``.

        The default returns ``drafter`` unchanged. A policy that changes the shape of the draft itself (a dynamic
        draft tree, for example) returns a wrapper and raises :class:`~economize.ConfigError` for a drafter or
        ``verifier`` it cannot work with. The decoder calls it once, before it checks the verifier and the drafter
        against each other."""
        return drafter
