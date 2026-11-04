"""Verification and commits after an exact pass.

:func:`verify_and_commit` lets the verifier compare the drafts of a :class:`~.passes.Pass` with its exact logits,
commits the accepted drafts and one further token per row, moves the K/V of accepted tree paths into place
(:func:`compact`), notifies the observers (:func:`notify`) and re-runs committed tokens of recurrent models whose
pass went further than the commits (:func:`replay`).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

from ..errors import ConfigError
from ..runtime import device_tensor, phase
from .capabilities import selector_distributions
from .cost import CostKind
from .protocols import Draft, TokenSelector, VerifyEvent

if TYPE_CHECKING:
    from .generation import Generation
    from .passes import Pass

__all__ = ["compact", "distribution_match", "notify", "replay", "verify_and_commit", "verify_chain", "verify_tree"]


def verify_and_commit(gen: Generation, current: Pass) -> tuple[list[bool], torch.Tensor]:
    """Verify the pass, commit tokens and return which rows finished and the column of every row's last committed
    token that the pass processed (where the drafter continues)."""
    draft = current.draft
    if draft.is_tree:
        draft = draft.verifiable()
        paths, nexts = verify_tree(gen, current)
        accepted = [len(path) for path in paths]
        nodes = draft.tokens.tolist()
        drafts = [[nodes[b][i] for i in path] for b, path in enumerate(paths)]
        columns = [[0] + [1 + i for i in path] for path in paths]
        compact(gen, current.starts, paths, drafts)
    else:
        if draft.candidates is not None:
            raise ConfigError("Draft.candidates marks nodes of a draft tree; a chain draft verifies every token")
        accepted_t, next_t = verify_chain(gen, current)
        table = torch.cat([accepted_t[:, None], next_t[:, None], draft.tokens.to(accepted_t.device)], 1).tolist()
        accepted, nexts = [row[0] for row in table], [row[1] for row in table]
        drafts = [row[2 : 2 + a] for row, a in zip(table, accepted)]
        columns = [list(range(f, f + n + a)) for f, n, a in zip(current.first, current.pending, accepted)]
    match = _match(gen, current) if gen.config.record_distributions else None
    bonus = device_tensor([end + a for end, a in zip(gen.ends, accepted)], gen.device)
    gen.seq.scatter_(1, bonus[:, None], device_tensor(nexts, gen.device)[:, None])

    finished, committed = [], []
    for b in range(len(gen.rows)):
        k, a = draft.lengths[b], accepted[b]
        if k > 0:
            gen.acceptance[gen.rows[b]].append((k, a))
            if gen.config.record_confidence:  # a device view, read back when the row finishes
                confidence = None if draft.is_tree else draft.confidence
                if confidence is None:
                    gen.confidence[gen.rows[b]].append(None)
                else:
                    gen.confidence[gen.rows[b]].append(confidence[b, :k])
            if gen.config.record_distributions:
                gen.match[gen.rows[b]].append(None if match is None else match[b, :k])
        done, count = False, current.pending[b]
        for token in drafts[b]:
            count += 1
            if done := gen.advance(b, token):
                break
        if not done:
            done = gen.advance(b, nexts[b])
        finished.append(done)
        committed.append(count)
        gen.starts[b] = gen.ends[b] - 1
    if gen.observers:
        notify(gen, current, columns, committed, nexts, accepted, draft.lengths)
    if gen.snapshot is not None:
        replay(gen, current, committed, finished)
    last = [cols[n - 1] for cols, n in zip(columns, committed)]
    return finished, device_tensor(last, current.exact.final.device)


def verify_chain(gen: Generation, current: Pass) -> tuple[torch.Tensor, torch.Tensor]:
    """Accepted drafts ``[B]`` and the next token ``[B]`` of every row of a chain pass, per selector group."""
    groups = gen.groups()
    if len(groups) == 1:
        return gen.verifier.verify_batch(current.logits, current.draft, groups[0][0], gen.seq, gen.ends)
    accepted = torch.zeros(len(gen.rows), dtype=torch.long, device=gen.device)
    nexts = torch.zeros_like(accepted)
    for selector, members, part, logits in _group_parts(groups, current.draft, current.logits):
        index = device_tensor(members, gen.device)
        ends = [gen.ends[b] for b in members]
        a, n = gen.verifier.verify_batch(logits, part, selector, gen.seq.index_select(0, index), ends)
        accepted[index] = a.to(gen.device, torch.long)
        nexts[index] = n.to(gen.device, torch.long)
    return accepted, nexts


def verify_tree(gen: Generation, current: Pass) -> tuple[list[list[int]], list[int]]:
    """The accepted path (node indices) and the next token of every row of a tree pass, per selector group. Only
    the candidate nodes (:meth:`~.protocols.Draft.verifiable`) are verified."""
    groups = gen.groups()
    draft = current.draft.verifiable()
    if len(groups) == 1:
        logits = current.logits if draft is current.draft else current.logits[:, : draft.width + 1]
        return gen.verifier.verify_tree(logits, draft, groups[0][0])
    paths: list[list[int]] = [[] for _ in gen.rows]
    nexts = [0] * len(gen.rows)
    for selector, members, part, logits in _group_parts(groups, draft, current.logits):
        part_paths, part_nexts = gen.verifier.verify_tree(logits, part, selector)
        for b, path, token in zip(members, part_paths, part_nexts):
            paths[b], nexts[b] = list(path), int(token)
    return paths, nexts


def _match(gen: Generation, current: Pass) -> torch.Tensor | None:
    """``[B, T, 3]`` :func:`distribution_match` of a chain pass whose drafter logged its logits, else ``None``."""
    draft = current.draft
    if draft.is_tree or draft.logits is None or draft.width == 0:
        return None
    groups = gen.groups()
    selector = groups[0][0] if len(groups) == 1 else None
    target = current.logits[:, : draft.width]
    return distribution_match(target, draft.logits.to(target.device), selector, draft.probs)


def distribution_match(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    selector: TokenSelector | None = None,
    draft_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """``[..., 3]``: top-1 agreement, ``KL(p || q)`` and ``TV(p, q)`` of the verifier's distribution ``p`` and the
    drafter's ``q`` at every position of ``target_logits`` and ``draft_logits`` ``[..., V]``.

    ``p`` and ``q`` are the distributions of a sampling selector that exposes them (``distributions``), the softmax
    for greedy selectors (whose one-hot distributions would make KL and TV degenerate) and for others;
    ``draft_probs`` replaces ``q`` (the distribution a sampling drafter drew from). Top-1 compares the
    argmaxes of ``p`` and ``q``. ``1 - TV`` is the expected acceptance of speculative sampling at the position.
    Where ``q`` is zero and ``p`` is not (truncated samplers, restricted draft vocabularies) the KL uses
    ``log q = log(1e-30)``, so it stays finite.
    """
    vocab = min(target_logits.shape[-1], draft_logits.shape[-1])

    def dist(logits: torch.Tensor) -> torch.Tensor:
        logits = logits[..., :vocab].float()
        sampling = selector is not None and not getattr(selector, "greedy", False)
        probs = selector_distributions(selector, logits) if sampling else None
        return torch.softmax(logits, -1) if probs is None else probs.float()

    p = dist(target_logits)
    q = draft_probs[..., :vocab].float().to(p.device) if draft_probs is not None else dist(draft_logits)
    top1 = (p.argmax(-1) == q.argmax(-1)).float()
    kl = (torch.xlogy(p, p) - p * q.clamp_min(1e-30).log()).sum(-1)
    tv = 0.5 * (p - q).abs().sum(-1)
    return torch.stack([top1, kl, tv], -1)


def _group_parts(
    groups: list[tuple[TokenSelector, list[int]]], draft: Draft, logits: torch.Tensor
) -> Iterator[tuple[TokenSelector, list[int], Draft, torch.Tensor]]:
    """For every selector group: the selector, its rows, their drafts and their logits ``[G, T_g + 1, V]``."""
    for selector, members in groups:
        part = draft.select(members)
        index = device_tensor(members, logits.device)
        yield selector, members, part, logits.index_select(0, index)[:, : part.width + 1]


def compact(gen: Generation, starts: list[int], paths: list[list[int]], tokens: list[list[int]]) -> None:
    """Move the K/V of every accepted tree path to the slots that follow the root, and its tokens into ``seq``."""
    rows = [b for b, path in enumerate(paths) if path]
    if not rows:
        return
    depth = max(len(paths[b]) for b in rows)
    trash = gen.cache.trash_slot
    table: list[list[int]] = []  # per row: the row, its source slots, destination slots, tokens and their columns
    for b in rows:
        pad = depth - len(paths[b])
        dst = [starts[b] + 1 + j for j in range(len(paths[b]))]
        table.append(
            [b]
            + [starts[b] + 1 + i for i in paths[b]] + [trash] * pad
            + dst + [trash] * pad
            + tokens[b] + [tokens[b][-1]] * pad  # padding rewrites the last token in its own column
            + dst + [dst[-1]] * pad
        )
    packed = device_tensor(table, gen.device)  # one copy to the device
    rows_t, (src, dst_t, path_tokens, columns) = packed[:, 0], packed[:, 1:].view(len(rows), 4, depth).unbind(1)
    gen.cache.gather(rows_t, src, dst_t)
    gen.seq.index_put_((rows_t[:, None], columns), path_tokens.to(gen.seq.dtype))


def notify(
    gen: Generation, current: Pass, columns: list[list[int]], committed: list[int], nexts: list[int],
    accepted: list[int], drafted: list[int] | None = None,
) -> None:
    """Send one :class:`VerifyEvent` per row to every observer. ``drafted`` is the number of verified drafts of
    every row (default: the pass's draft lengths)."""
    drafted = current.draft.lengths if drafted is None else drafted
    for b, row in enumerate(gen.rows):
        count, start = committed[b], current.starts[b]
        picked = device_tensor(columns[b][:count], current.exact.final.device)
        hidden = {layer: h[b, picked.to(h.device)] for layer, h in current.exact.hidden.items()}
        event = VerifyEvent(
            row=row,
            start=start,
            tokens=gen.seq[b, start : start + count].clone(),
            next_token=nexts[b],
            drafted=drafted[b],
            accepted=accepted[b],
            hidden=hidden,
        )
        for observer in gen.observers:
            observer(event)


def replay(gen: Generation, current: Pass, committed: list[int], finished: list[bool]) -> None:
    """Re-run the committed tokens of rows whose pass advanced the recurrent states too far.

    A chain pass advanced the states of every row by its width. A tree pass left them where they were (its recurrent
    layers run branched copies, see :meth:`~economize.runtime.LayerRunner.tree_context`), so every unfinished row
    runs its committed path; the extra column keeps a fully accepted path from counting as already applied.
    """
    snapshot, gen.snapshot = gen.snapshot, None
    assert snapshot is not None
    width = current.width + (1 if current.draft.is_tree else 0)
    lengths = [width if done else n for n, done in zip(committed, finished)]
    if all(n == width for n in lengths):
        return
    starts = device_tensor(current.starts, gen.device)
    positions = starts[:, None] + torch.arange(width, device=gen.device)
    tokens = gen.seq.gather(1, positions.clamp(max=gen.scratch))
    with phase(gen.phase):
        gen.runner.replay(gen.cache, snapshot, tokens, starts, device_tensor(lengths, gen.device))
    for b, n in enumerate(lengths):
        if n < width:
            gen.costs[gen.rows[b]].record(CostKind.REPLAY, n, current.starts[b], 1, gen.num_layers)
