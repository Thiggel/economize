"""Built-in verifiers.

A verifier decides which drafts survive an exact pass and chooses the token that follows them. Register a new one
with ``@VERIFIERS.register("name")`` and select it with ``DecodeConfig(verifier="name")``.

==================  ===================================================================================================
Name                Behaviour
==================  ===================================================================================================
``"greedy"``        Lossless for deterministic selectors: the longest prefix of drafts the selector reproduces on the
                    exact logits. Supports draft trees.
``"rejection"``     Lossless for sampling: speculative sampling (Leviathan et al., 2023; Chen et al., 2023) with the
                    selector's own distribution, and recursive rejection sampling over draft trees (SpecInfer, Miao
                    et al., 2024). The output has the same distribution as sampling without drafts.
``"typical"``       Lossy: typical acceptance (Medusa, Cai et al., 2024). A draft is kept when the exact model gives
                    it a probability above an entropy-dependent threshold. Supports draft trees. Reduces to
                    ``"greedy"`` for a deterministic selector.
``"accept_all"``    Lossy: drafts are final when proposed; the exact pass refreshes the cache and emits one token.
==================  ===================================================================================================
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from ..errors import ConfigError
from ..registry import Registry
from ..runtime.transfer import device_tensor
from .capabilities import selector_distribution, selector_distributions, selector_generator, selector_many
from .protocols import Draft, TokenSelector, Verifier

if TYPE_CHECKING:
    from .config import DecodeConfig

__all__ = ["VERIFIERS", "AcceptAllVerifier", "GreedyVerifier", "RejectionVerifier", "TypicalVerifier"]

VERIFIERS: Registry[type[Verifier]] = Registry("verifier", "economize.verifiers")


def _accepted_prefix(keep: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """The number of leading drafts kept in every row: ``keep`` ``[B, W]`` marks the drafts that pass, ``lengths``
    ``[B]`` how many of the ``W`` columns are real drafts. Returns ``[B]`` long."""
    keep = keep & (torch.arange(keep.shape[1], device=keep.device) < lengths[:, None])
    return keep.long().cumprod(1).sum(1)


def _accepts(u: float | torch.Tensor, p: torch.Tensor, q: torch.Tensor | None, x: int) -> bool:
    """Whether the draft ``x`` drawn from ``q`` (a one-hot on ``x`` when ``None``) survives the coin ``u`` against
    the exact distribution ``p``: ``u < p(x) / q(x)``."""
    q_x = 1.0 if q is None else q[x]
    return bool(u * q_x < p[x])


def _residual(p: torch.Tensor, q: torch.Tensor | None, x: int) -> torch.Tensor:
    """The unnormalized distribution to resample from after rejecting ``x``: ``max(p - q, 0)``, which for a one-hot
    ``q`` (``None``) is ``p`` with ``x`` removed. Falls back to ``p`` when nothing is left."""
    if q is None:
        residual = p.clone()
        residual[x] = 0.0
    else:
        residual = (p - q).clamp_min(0)
    return residual if bool(residual.sum() > 0) else p


def _children(parents: list[int], count: int) -> dict[int, list[int]]:
    """Children of every node of one tree row (``-1`` is the last committed token), in node order."""
    children: dict[int, list[int]] = {}
    for i in range(count):
        children.setdefault(parents[i], []).append(i)
    return children


@VERIFIERS.register("accept_all")
class AcceptAllVerifier(Verifier):
    """Lossy decoding: drafts are final, the exact pass refreshes the cache and emits one exact token."""

    commits_drafts = True

    def verify(
        self,
        logits: torch.Tensor,
        drafts: list[int],
        probs: torch.Tensor | None,
        selector: TokenSelector,
        history: torch.Tensor,
        end: int,
    ) -> tuple[int, int]:
        k = len(drafts)
        return k, selector(logits[k], history[: end + k])

    def verify_batch(
        self, logits: torch.Tensor, draft: Draft, selector: TokenSelector, seq: torch.Tensor, ends: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = device_tensor(draft.lengths, logits.device)
        rows = torch.arange(logits.shape[0], device=logits.device)
        tokens = selector_many(selector, logits[rows, lengths])
        if tokens is None:
            return super().verify_batch(logits, draft, selector, seq, ends)
        return lengths, tokens


@VERIFIERS.register("greedy")
class GreedyVerifier(Verifier):
    """Lossless speculative decoding for deterministic selectors: accept the longest prefix of drafts that the
    selector reproduces on the exact logits, then append the selector's own token."""

    requires_greedy = True
    supports_trees = True

    def verify(
        self,
        logits: torch.Tensor,
        drafts: list[int],
        probs: torch.Tensor | None,
        selector: TokenSelector,
        history: torch.Tensor,
        end: int,
    ) -> tuple[int, int]:
        for i, draft in enumerate(drafts):
            token = selector(logits[i], history[: end + i])
            if token != draft:
                return i, token
        k = len(drafts)
        return k, selector(logits[k], history[: end + k])

    def verify_batch(
        self, logits: torch.Tensor, draft: Draft, selector: TokenSelector, seq: torch.Tensor, ends: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        choice = selector_many(selector, logits)
        if choice is None:
            return super().verify_batch(logits, draft, selector, seq, ends)
        lengths = device_tensor(draft.lengths, logits.device)
        accepted = _accepted_prefix(choice[:, : draft.width] == draft.tokens.to(logits.device), lengths)
        return accepted, choice.gather(1, accepted[:, None])[:, 0]

    def verify_tree(
        self, logits: torch.Tensor, draft: Draft, selector: TokenSelector
    ) -> tuple[list[list[int]], list[int]]:
        choice = selector_many(selector, logits)
        if choice is None or draft.parents is None:
            raise ConfigError("tree verification needs a selector with select_many() (one that ignores the history)")
        choices, parents, tokens = choice.tolist(), draft.parents.tolist(), draft.tokens.tolist()
        paths, nexts = [], []
        for b, k in enumerate(draft.lengths):
            children: dict[tuple[int, int], int] = {}
            for i in range(k):
                children.setdefault((parents[b][i], tokens[b][i]), i)
            path, node = [], -1
            while (child := children.get((node, choices[b][node + 1]))) is not None:
                path.append(child)
                node = child
            paths.append(path)
            nexts.append(choices[b][node + 1])
        return paths, nexts


@VERIFIERS.register("rejection")
class RejectionVerifier(Verifier):
    """Lossless speculative sampling.

    Every draft ``x`` is drawn from the selector's distribution ``q`` on the drafter's logits. The exact pass gives
    the selector's distribution ``p``; ``x`` is accepted with probability ``min(1, p(x) / q(x))``. The first rejected
    draft is replaced by a sample from the normalized residual ``max(p - q, 0)``. When every draft is accepted the
    selector samples one more token from the exact logits. The tokens then follow the distribution of sampling from
    the model without drafts.

    A draft without a distribution (``probs`` is ``None``: a deterministic drafter such as n-gram lookup, or a
    drafter that takes its argmax) is treated as drawn from a one-hot ``q`` on itself: it is accepted with
    probability ``p(x)``, and a rejection samples from ``p`` with ``x`` removed. This is the same rule and stays
    lossless.

    The selector needs ``distribution(logits, history)`` (and ``distributions(logits)`` for the batched path); its
    generator for the logits' device (:func:`~economize.decoding.capabilities.selector_generator`) drives every draw.
    """

    needs_draft_probs = True
    supports_trees = True

    def draft(
        self, logits: torch.Tensor, selector: TokenSelector, history: torch.Tensor
    ) -> tuple[int, torch.Tensor | None]:
        dist = selector_distribution(selector, logits, history, "rejection")
        return int(torch.multinomial(dist, 1, generator=selector_generator(selector, dist.device))), dist

    def verify(
        self,
        logits: torch.Tensor,
        drafts: list[int],
        probs: torch.Tensor | None,
        selector: TokenSelector,
        history: torch.Tensor,
        end: int,
    ) -> tuple[int, int]:
        generator = selector_generator(selector, logits.device)
        for i, draft in enumerate(drafts):
            p = selector_distribution(selector, logits[i], history[: end + i], "rejection")
            q = None if probs is None else probs[i].to(p.device, torch.float32)
            if _accepts(torch.rand((), generator=generator, device=p.device), p, q, draft):
                continue
            residual = _residual(p, q, draft)
            return i, int(torch.multinomial(residual / residual.sum(), 1, generator=generator))
        k = len(drafts)
        return k, selector(logits[k], history[: end + k])

    def draft_many(
        self, logits: torch.Tensor, selector: TokenSelector
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        dist = selector_distributions(selector, logits)
        if dist is None:
            return None
        return torch.multinomial(dist, 1, generator=selector_generator(selector, dist.device))[:, 0], dist

    def verify_batch(
        self, logits: torch.Tensor, draft: Draft, selector: TokenSelector, seq: torch.Tensor, ends: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p = selector_distributions(selector, logits)  # [B, T + 1, V]
        if p is None:
            return super().verify_batch(logits, draft, selector, seq, ends)
        generator = selector_generator(selector, p.device)
        batch, width, device = p.shape[0], draft.width, p.device
        lengths = device_tensor(draft.lengths, device)
        rows = torch.arange(batch, device=device)
        if width == 0:
            accepted = torch.zeros(batch, dtype=torch.long, device=device)
            residual = p[:, 0]
        else:
            tokens = draft.tokens.to(device).clamp_min(0)
            p_x = p[:, :width].gather(2, tokens[:, :, None])[..., 0]
            u = torch.rand(batch, width, generator=generator, device=device)
            if draft.probs is None:  # one-hot q on every draft
                accepted = _accepted_prefix(u < p_x, lengths)
                p_next = p[rows, accepted]
                rejected = torch.cat([tokens, tokens.new_zeros(batch, 1)], 1)[rows, accepted]
                drop = torch.zeros_like(p_next).scatter_(1, rejected[:, None], 1.0) * (accepted < lengths)[:, None]
                residual = p_next * (1 - drop)
            else:
                q = draft.probs.to(device, torch.float32)
                accepted = _accepted_prefix(u * q.gather(2, tokens[:, :, None])[..., 0] < p_x, lengths)
                p_next = p[rows, accepted]
                q_next = torch.cat([q, q.new_zeros(batch, 1, q.shape[-1])], 1)[rows, accepted]
                residual = (p_next - q_next * (accepted < lengths)[:, None]).clamp_min(0)
            residual = torch.where(residual.sum(-1, keepdim=True) > 0, residual, p_next)
        following = torch.multinomial(residual / residual.sum(-1, keepdim=True), 1, generator=generator)[:, 0]
        return accepted.to(logits.device), following.to(logits.device)

    def verify_tree(
        self, logits: torch.Tensor, draft: Draft, selector: TokenSelector
    ) -> tuple[list[list[int]], list[int]]:
        """Recursive rejection sampling down the tree (SpecInfer, Miao et al., 2024).

        At every node the children are tried in order. Child ``i`` (token ``x``, drawn from ``q_i``) is accepted
        with probability ``min(1, p(x) / q_i(x))``; after a rejection ``p`` becomes the normalized residual
        ``max(p - q_i, 0)``. ``q_i`` is ``draft.probs[b, i]``, the distribution the drafter drew the child from
        given its earlier siblings (a one-hot on ``x`` when ``draft.probs`` is ``None``: deterministic drafts). When
        every child is rejected the next token is drawn from the residual; an accepted leaf draws it from its own
        exact distribution. The output follows the selector's distribution exactly, whatever the tree.
        """
        p_all = selector_distributions(selector, logits)  # [B, T + 1, V]
        if p_all is None or draft.parents is None:
            raise ConfigError(
                "the 'rejection' verifier needs a selector with distributions() (one that ignores the history) "
                "for draft trees"
            )
        device = p_all.device
        generator = selector_generator(selector, device)
        parents, tokens = draft.parents.tolist(), draft.tokens.tolist()
        uniforms = torch.rand(draft.tokens.shape, generator=generator, device=device).tolist()
        paths, finals = [], []
        for b, count in enumerate(draft.lengths):
            children = _children(parents[b], count)
            path, node = [], -1
            while True:
                p = p_all[b, node + 1]
                chosen = None
                for i in children.get(node, []):
                    x = tokens[b][i]
                    q = None if draft.probs is None else draft.probs[b, i].to(device, torch.float32)
                    if _accepts(uniforms[b][i], p, q, x):
                        chosen = i
                        break
                    residual = _residual(p, q, x)
                    p = residual / residual.sum()
                if chosen is None:
                    break
                path.append(chosen)
                node = chosen
            paths.append(path)
            finals.append(p)
        following = torch.multinomial(torch.stack(finals), 1, generator=generator)[:, 0].tolist()
        return paths, following


@VERIFIERS.register("typical")
class TypicalVerifier(Verifier):
    """Typical acceptance (Medusa, Cai et al., 2024), a lossy verifier for sampling.

    With ``p`` the selector's distribution on the exact logits and ``H(p) = -sum p log(p + 1e-5)`` its entropy, a
    draft ``x`` is accepted when ``p(x) > min(threshold, alpha * exp(-H(p)))``. The drafts are the drafter's most
    likely tokens. The token after the accepted drafts is sampled from ``p`` restricted to the tokens above the same
    threshold. Low-entropy (confident) positions accept only likely tokens, high-entropy positions accept more.
    A deterministic selector gives a one-hot ``p``, which makes this verifier identical to ``"greedy"``.

    For a draft tree the accepted path is the deepest node whose path is accepted throughout (the first such node
    on a tie), as in Medusa's choice of the longest accepted candidate.

    Options (``DecodeConfig.verifier_options``): ``threshold`` (Medusa's ``posterior_threshold``, default 0.09)
    and ``alpha`` (``posterior_alpha``, default 0.3).
    """

    supports_trees = True
    default_options: ClassVar[Mapping[str, Any]] = {"threshold": 0.09, "alpha": 0.3}

    def __init__(self, config: DecodeConfig) -> None:
        super().__init__(config)
        self.threshold, self.alpha = float(self.options["threshold"]), float(self.options["alpha"])

    # ---- drafts: the drafter's most likely tokens --------------------------------------------------------------
    def draft(
        self, logits: torch.Tensor, selector: TokenSelector, history: torch.Tensor
    ) -> tuple[int, torch.Tensor | None]:
        return int(logits.argmax(-1)), None

    def draft_many(
        self, logits: torch.Tensor, selector: TokenSelector
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        return logits.argmax(-1), None

    # ---- verification ------------------------------------------------------------------------------------------
    def cutoff(self, p: torch.Tensor) -> torch.Tensor:
        """The acceptance threshold of every distribution ``p`` ``[..., V]``, ``[...]``."""
        entropy = -(p * torch.log(p + 1e-5)).sum(-1)
        return torch.clamp(self.alpha * torch.exp(-entropy), max=self.threshold)

    def verify(
        self,
        logits: torch.Tensor,
        drafts: list[int],
        probs: torch.Tensor | None,
        selector: TokenSelector,
        history: torch.Tensor,
        end: int,
    ) -> tuple[int, int]:
        p = torch.stack(
            [selector_distribution(selector, logits[i], history[: end + i], "typical") for i in range(len(drafts) + 1)]
        )
        cut = self.cutoff(p)
        accepted = 0
        for i, x in enumerate(drafts):
            if not bool(p[i, x] > cut[i]):
                break
            accepted += 1
        return accepted, int(self._sample(p[accepted : accepted + 1], cut[accepted : accepted + 1], selector)[0])

    def verify_batch(
        self, logits: torch.Tensor, draft: Draft, selector: TokenSelector, seq: torch.Tensor, ends: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p = selector_distributions(selector, logits)
        if p is None:
            return super().verify_batch(logits, draft, selector, seq, ends)
        cut = self.cutoff(p)  # [B, T + 1]
        width, device = draft.width, p.device
        rows = torch.arange(p.shape[0], device=device)
        accepted = torch.zeros(p.shape[0], dtype=torch.long, device=device)
        if width > 0:
            tokens = draft.tokens.to(device).clamp_min(0)[:, :, None]
            lengths = device_tensor(draft.lengths, device)
            accepted = _accepted_prefix(p[:, :width].gather(2, tokens)[..., 0] > cut[:, :width], lengths)
        following = self._sample(p[rows, accepted], cut[rows, accepted], selector)
        return accepted.to(logits.device), following.to(logits.device)

    def verify_tree(
        self, logits: torch.Tensor, draft: Draft, selector: TokenSelector
    ) -> tuple[list[list[int]], list[int]]:
        p = selector_distributions(selector, logits)
        if p is None or draft.parents is None:
            raise ConfigError("the 'typical' verifier needs a selector with distributions() for draft trees")
        cut = self.cutoff(p)
        device = p.device
        tokens = draft.tokens.to(device).clamp_min(0)
        parents = draft.parents.to(device)
        # node i is kept when its exact probability under its parent's distribution exceeds the parent's cutoff
        parent_rows = (parents + 1).clamp_min(0)
        p_parent = p.gather(1, parent_rows[:, :, None].expand(-1, -1, p.shape[-1]))
        kept = (p_parent.gather(2, tokens[:, :, None])[..., 0] > cut.gather(1, parent_rows)).tolist()
        parent_list = draft.parents.tolist()
        paths, nodes = [], []
        for b, count in enumerate(draft.lengths):
            best: list[int] = []
            ok: dict[int, list[int]] = {-1: []}
            for i in range(count):  # topological order: a parent comes before its children
                parent = parent_list[b][i]
                if parent in ok and kept[b][i]:
                    ok[i] = ok[parent] + [i]
                    if len(ok[i]) > len(best):
                        best = ok[i]
            paths.append(best)
            nodes.append(best[-1] + 1 if best else 0)
        rows = torch.arange(p.shape[0], device=device)
        index = device_tensor(nodes, device)
        return paths, self._sample(p[rows, index], cut[rows, index], selector).tolist()

    # ---- internals -------------------------------------------------------------------------------------------
    @staticmethod
    def _sample(p: torch.Tensor, cut: torch.Tensor, selector: TokenSelector) -> torch.Tensor:
        """One token per row of ``p`` ``[B, V]`` from ``p`` restricted to the tokens at or above ``cut`` ``[B]``."""
        if selector.greedy:
            return p.argmax(-1)
        typical = p * (p >= cut[:, None])
        typical = torch.where(typical.sum(-1, keepdim=True) > 0, typical, p)
        generator = selector_generator(selector, p.device)
        return torch.multinomial(typical / typical.sum(-1, keepdim=True), 1, generator=generator)[:, 0]
