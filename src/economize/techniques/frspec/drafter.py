"""The FR-Spec drafter: any chain drafter whose draft tokens are restricted to a draft vocabulary."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from ...decoding.protocols import Draft, Drafter
from ...errors import ConfigError
from .._drafting import DelegatingDrafter

if TYPE_CHECKING:
    from ...decoding.request import DraftRequest

__all__ = ["FRSpecDrafter", "restrict_logits", "widen_logits"]


def widen_logits(logits: torch.Tensor, token_ids: torch.Tensor, size: int) -> torch.Tensor:
    """Logits ``[..., size]`` with ``logits`` ``[..., K]`` at ``token_ids`` ``[K]`` and ``-inf`` elsewhere."""
    full = logits.new_full((*logits.shape[:-1], size), float("-inf"))
    return full.index_copy_(-1, token_ids.to(logits.device), logits)


def restrict_logits(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """``logits`` ``[..., V]`` with ``-inf`` outside ``allowed`` (a bool mask ``[V']``, ``V' <= V``; ids at or above
    ``V'`` are outside)."""
    size = logits.shape[-1]
    keep = allowed[:size].to(logits.device)
    if keep.numel() < size:
        keep = torch.cat([keep, keep.new_zeros(size - keep.numel())])
    return logits.masked_fill(~keep, float("-inf"))


class _RestrictedRequest:
    """A view of a :class:`~economize.decoding.DraftRequest` whose :meth:`extend` restricts the logits first
    (``restrict`` turns the inner drafter's logits into full-vocabulary logits with ``-inf`` outside the draft
    vocabulary).

    It counts the tokens every row gains through :meth:`extend` in :attr:`extended`, so the wrapper can tell
    whether the inner drafter added tokens some other way. With ``sliced`` (the inner head computes the draft ids'
    logits only, so its arg-max is a draft id) it also lets the inner drafter hand whole greedy chains to
    :meth:`extend_tokens`; otherwise :meth:`graphed_chain_steps` is ``None`` and every step goes through
    :meth:`extend`."""

    def __init__(self, request: DraftRequest, restrict: Callable[[torch.Tensor], torch.Tensor],
                 sliced: bool = False) -> None:
        self._request, self._restrict, self._sliced = request, restrict, sliced
        self.initial = list(request.lengths)
        self.extended = [0] * len(request.lengths)

    def extend(self, logits: torch.Tensor) -> None:
        before = list(self._request.lengths)
        self._request.extend(self._restrict(logits))
        for b, (n, m) in enumerate(zip(self._request.lengths, before)):
            self.extended[b] += n - m

    def graphed_chain_steps(self) -> int | None:
        return self._request.graphed_chain_steps() if self._sliced else None

    def extend_tokens(self, tokens: torch.Tensor, confidence: torch.Tensor | None = None) -> None:
        if not self._sliced:
            raise ConfigError("extend_tokens needs a sliced head: the tokens could lie outside the draft vocabulary")
        before = list(self._request.lengths)
        self._request.extend_tokens(tokens, confidence)
        for b, (n, m) in enumerate(zip(self._request.lengths, before)):
            self.extended[b] += n - m

    def bypassed(self, draft: Draft) -> list[int]:
        """Rows whose drafts hold tokens that did not go through :meth:`extend`."""
        lengths = self._request.lengths
        return [b for b, (n, start, through) in enumerate(zip(lengths, self.initial, self.extended))
                if n - start != through or draft.lengths[b] > n]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._request, name)


class FRSpecDrafter(DelegatingDrafter):
    """Wraps the drafter of another technique and restricts every draft step's logits to the draft vocabulary.

    With ``token_ids`` the inner drafter computes the logits of the draft ids only
    (:meth:`~economize.techniques._drafting.HeadSliceable.slice_head`), and the wrapper places them in a
    full-vocabulary row of ``-inf``. Without, the inner drafter computes logits over the full vocabulary and the
    wrapper sets those of the other ids to ``-inf``. Either way the token is chosen from the same logits. Greedy drafts
    are then the argmax over the draft vocabulary, sampled drafts follow the softmax renormalized over it (the
    distribution the ``rejection`` verifier divides by), and draft policies see the restricted confidence. Everything
    else (the layers the inner drafter reads and runs, its cost footprint, its cache and graphs) is the inner drafter's
    (:class:`DelegatingDrafter`).

    Only chain drafts whose tokens go through :meth:`~economize.decoding.DraftRequest.extend` (or, with a sliced
    head, through :meth:`~economize.decoding.DraftRequest.extend_tokens`) can be restricted this
    way (the :class:`~economize.decoding.WrappableDrafter` contract). A drafter that drafts trees for the request's
    verifier (:meth:`~economize.decoding.Drafter.drafts_trees`) is refused before it drafts, and a proposal with tokens
    that bypassed ``extend``, or a tree the drafter did not declare, raises :class:`~economize.ConfigError` after it.

    Args:
        inner: The drafter of the wrapped technique.
        allowed: Bool mask over token ids ``[V]``, true for the draft vocabulary.
        token_ids: The draft ids ``[K]`` when the inner drafter's head was sliced to them, else ``None``.

    Raises:
        ConfigError: ``inner`` is not a :class:`~economize.decoding.WrappableDrafter`.
    """

    def __init__(self, inner: Drafter, allowed: torch.Tensor, token_ids: torch.Tensor | None = None) -> None:
        super().__init__(inner, "frspec")
        self.allowed = allowed
        self.token_ids = token_ids

    @property
    def sliced(self) -> bool:
        """Whether the inner drafter computes the logits of the draft ids only."""
        return self.token_ids is not None

    def restrict(self, logits: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary logits with ``-inf`` outside the draft vocabulary, from the inner drafter's logits."""
        if self.token_ids is None:
            return restrict_logits(logits, self.allowed)
        return widen_logits(logits, self.token_ids, self.allowed.numel())

    def step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        return self.restrict(self.inner.step(tokens, positions, valid))

    # ---- dynamic trees (economize.decoding.dynamic_tree): the inner drafter's tree steps, restricted ------------
    def supports_tree_steps(self) -> bool:
        supports = getattr(self.inner, "supports_tree_steps", None)
        return callable(supports) and bool(supports())

    def tree_begin(self, request: DraftRequest) -> None:
        self.inner.tree_begin(request)  # type: ignore[attr-defined]

    def tree_state(self) -> tuple[torch.Tensor, ...]:
        return self.inner.tree_state()  # type: ignore[attr-defined, no-any-return]

    def tree_step(self, tokens: torch.Tensor, parents: torch.Tensor, base: torch.Tensor,
                  *states: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        logits, out = self.inner.tree_step(tokens, parents, base, *states)  # type: ignore[attr-defined]
        return self.restrict(logits), out

    def propose(self, request: DraftRequest) -> Draft:
        name = type(self.inner).__name__
        if self.inner.drafts_trees(request.verifier):
            raise ConfigError(f"frspec restricts chain drafts only; {name} drafts trees for verifier "
                              f"{type(request.verifier).__name__}")
        restricted = _RestrictedRequest(request, self.restrict, self.sliced)
        draft = self.inner.propose(restricted)  # type: ignore[arg-type]
        if draft.is_tree:
            raise ConfigError(f"frspec restricts chain drafts only; {name} drafted a tree")
        if rows := restricted.bypassed(draft):
            raise ConfigError(f"frspec restricts tokens drafted through DraftRequest.extend only; {name} drafted "
                              f"tokens for rows {rows} without it, so they could lie outside the draft vocabulary")
        return draft
