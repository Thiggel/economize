"""The exact passes of the decoding loop: the prefill, chain passes and draft-tree passes.

Every pass runs the unmodified model (with the decoder's layer policy) over the tokens of every row that no exact
pass has seen yet, records its cost and returns a :class:`Pass` that says where each row's tokens sit in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..errors import ConfigError
from ..runtime import ExactPass, TreeLayout, device_tensor, phase
from .protocols import Draft

if TYPE_CHECKING:
    from .generation import Generation

__all__ = ["Pass", "chain_pass", "chunked_prefill", "prefill", "single_token_pass", "tree_pass"]


@dataclass
class Pass:
    """One exact pass and where every row's tokens sit in it.

    Attributes:
        exact: The pass's hidden states and context.
        logits: ``[B, T + 1, V]`` float logits; entry ``j`` predicts the token after the row's ``j``-th draft.
        draft: The drafts the pass verifies.
        starts: Position of every row's first token in the pass.
        first: Column of that token.
        pending: Tokens of every row before its drafts.
        width: Columns of the pass.
    """

    exact: ExactPass
    logits: torch.Tensor
    draft: Draft
    starts: list[int]
    first: list[int]
    pending: list[int]
    width: int

    def select(self, keep: list[int], device: torch.device) -> Pass:
        """The pass restricted to the rows ``keep`` (logits and bookkeeping; ``exact`` stays whole), without drafts."""
        index = device_tensor(keep, self.logits.device)
        return Pass(
            self.exact,
            self.logits.index_select(0, index),
            Draft.empty(len(keep), device),
            [self.starts[b] for b in keep],
            [self.first[b] for b in keep],
            [self.pending[b] for b in keep],
            self.width,
        )


def prefill(gen: Generation) -> Pass:
    """The left-padded pass over every row's prompt; its logits predict each row's first new token.

    With :attr:`DecodeConfig.prefill_chunk <economize.decoding.DecodeConfig.prefill_chunk>` a longer prompt runs as
    consecutive chunks (:func:`chunked_prefill`).
    """
    lengths = [gen.prompt_lengths[row] for row in gen.rows]
    width = max(lengths)
    chunk = gen.config.prefill_chunk
    if chunk is not None and width > chunk:
        return chunked_prefill(gen, chunk)
    offsets = device_tensor([width - n for n in lengths], gen.device)
    positions = torch.arange(width, device=gen.device)[None] - offsets[:, None]
    valid = None if len(set(lengths)) == 1 else positions >= 0
    tokens = gen.seq.gather(1, positions.clamp_min(0))
    ctx = gen.runner.context(gen.cache, positions, valid, trusted=True)
    with phase("prefill"):
        exact = gen.runner.forward(tokens, ctx, capture=gen.capture, policy=gen.layer_policy)
    gen.prefilled = True
    logits = gen.runner.logits(exact.final[:, -1:]).float()
    gen.record_exact(lengths, [0] * len(lengths))
    batch = len(lengths)
    return Pass(exact, logits, Draft.empty(batch, gen.device), [0] * batch, [width - n for n in lengths],
                list(lengths), width)


def chunked_prefill(gen: Generation, chunk: int) -> Pass:
    """The prefill as consecutive passes of ``chunk`` tokens over prompts of equal length. Each chunk writes its
    K/V before the next one attends to them, so the logits are those of one pass; the returned pass holds the last
    chunk, which ends with the last prompt token (the token the drafter and the first verification read).

    The chunks run inside one ``writing_at`` block over the whole prompt, so a KV-cache technique
    (:class:`~economize.techniques._kv.CacheWrapper`) sees a single exact pass: its pass boundaries run once,
    around all chunks, as in an unchunked prefill."""
    lengths = [gen.prompt_lengths[row] for row in gen.rows]
    if len(set(lengths)) > 1:
        raise ConfigError(f"prefill_chunk needs prompts of equal length, got lengths {sorted(set(lengths))}")
    width, batch = lengths[0], len(lengths)
    positions = torch.arange(width, device=gen.device)[None].expand(batch, -1)
    with phase("prefill"), gen.cache.writing_at(positions):
        for begin in range(0, width, chunk):
            end = min(begin + chunk, width)
            ctx = gen.runner.context(gen.cache, positions[:, begin:end], None, trusted=True, consecutive=True)
            exact = gen.runner.forward(gen.seq[:, begin:end], ctx, capture=gen.capture)
    gen.prefilled = True
    logits = gen.runner.logits(exact.final[:, -1:]).float()
    gen.record_exact(lengths, [0] * batch)
    last = width - begin
    return Pass(exact, logits, Draft.empty(batch, gen.device), [begin] * batch, [0] * batch, [last] * batch, last)


def chain_pass(gen: Generation, draft: Draft) -> Pass:
    """The right-padded pass over every row's pending tokens followed by its chain of drafts."""
    starts = list(gen.starts)
    pending = [end - start for end, start in zip(gen.ends, starts)]
    widths = [n + k for n, k in zip(pending, draft.lengths)]
    width = max(widths)
    columns = torch.arange(width, device=gen.device)
    positions = device_tensor(starts, gen.device)[:, None] + columns
    valid = None
    if len(set(widths)) > 1:
        valid = columns < device_tensor(widths, gen.device)[:, None]
    index = positions if valid is None else torch.where(valid, positions, torch.full_like(positions, gen.scratch))
    tokens = gen.seq.gather(1, index)
    last = device_tensor([n - 1 for n in pending], gen.device)
    picks = (last[:, None] + torch.arange(draft.width + 1, device=gen.device)).clamp(max=width - 1)
    if gen.graphs is not None:  # one graph per (batch size, width): the mask is always given
        if valid is None:
            valid = torch.ones_like(positions, dtype=torch.bool)
        exact, logits = gen.graphs.exact(tokens, positions, valid, picks, starts)
    else:
        ctx = gen.runner.context(gen.cache, positions, valid, trusted=True, consecutive=True)
        with phase(gen.phase):
            exact = gen.runner.forward(tokens, ctx, capture=gen.capture, policy=gen.layer_policy)
        index = picks[:, :, None].to(exact.final.device).expand(-1, -1, exact.final.shape[-1])
        logits = gen.runner.logits(exact.final.gather(1, index)).float()
    gen.record_exact(widths, starts)
    return Pass(exact, logits, draft, starts, [0] * len(starts), pending, width)


def single_token_pass(
    gen: Generation, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor, picks: torch.Tensor
) -> Pass:
    """A graphed chain pass over one pending token per row from inputs already on the device, all ``[B, 1]``:
    ``tokens``, their ``positions``, ``valid`` (all true) and ``picks`` (all zero). It is the pass :func:`chain_pass`
    builds without drafts, without host copies."""
    assert gen.graphs is not None
    exact, logits = gen.graphs.exact(tokens, positions, valid, picks, list(gen.starts))
    gen.record_exact([1] * len(gen.rows), gen.starts)
    batch = len(gen.rows)
    return Pass(exact, logits, Draft.empty(batch, gen.device), list(gen.starts), [0] * batch, [1] * batch, 1)


def tree_pass(gen: Generation, draft: Draft) -> Pass:
    """The pass over every row's last committed token (the root) and the nodes of its draft tree."""
    if not gen.verifier.supports_trees:  # a drafter that drafts trees without declaring it in drafts_trees()
        raise ConfigError(f"verifier {type(gen.verifier).__name__} cannot verify draft trees; use a verifier that "
                          "verifies trees or a chain drafter")
    assert draft.parents is not None
    starts = list(gen.starts)
    batch, width = len(starts), draft.width + 1
    starts_t = device_tensor(starts, gen.device)
    root = gen.seq.gather(1, starts_t[:, None])
    tokens = torch.cat([root, draft.tokens.clamp_min(0).to(gen.device)], 1)
    root_parent = torch.full((batch, 1), -1, dtype=torch.long, device=gen.device)
    parents = torch.cat([root_parent, draft.parents.to(gen.device) + 1], 1)
    valid = torch.arange(width, device=gen.device) < device_tensor(draft.lengths, gen.device)[:, None] + 1
    if gen.graphs is not None:  # one graph per (batch size, tree width)
        exact, logits = gen.graphs.tree(tokens, parents, starts_t, valid, starts)
    else:
        tree = TreeLayout.build(parents, starts_t, valid, trash_slot=gen.cache.trash_slot)
        ctx = gen.runner.tree_context(gen.cache, tree, valid, trusted=True)
        with phase(gen.phase):
            exact = gen.runner.forward(tokens, ctx, capture=gen.capture, policy=gen.layer_policy)
        logits = gen.runner.logits(exact.final).float()
    gen.record_exact([k + 1 for k in draft.lengths], starts)
    return Pass(exact, logits, draft, starts, [0] * batch, [1] * batch, width)
