"""The drafting step of the decoding loop: the drafter proposes the next cycle's tokens for every row.

Recurrent states are saved before drafting and restored after it (draft steps advance them). With a verifier that
commits drafts (lossy decoding) the drafts are committed right away, rows that finish while drafting leave the batch
with the cost events of their later draft steps dropped, and the next exact pass sees no drafts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..errors import ConfigError
from ..runtime import phase
from .protocols import Draft, draft_footprint
from .request import DraftRequest

if TYPE_CHECKING:
    from .generation import Generation

__all__ = ["draft"]


def draft(gen: Generation) -> Draft:
    """Ask the drafter for every row's drafts; return what the next exact pass verifies."""
    drafter = gen.drafter
    assert drafter is not None
    n_draft = gen.config.cycle_length - 1
    commits = gen.verifier.commits_drafts
    budgets = [min(n_draft, gen.remaining(b)) if commits else n_draft for b in range(len(gen.rows))]
    costs = [gen.costs[row] for row in gen.rows]
    marks = [len(cost) for cost in costs]
    if gen.runner.has_recurrent_layers:
        gen.snapshot = gen.cache.snapshot()
    request = DraftRequest(
        gen.seq,
        gen.ends,
        budgets,
        selector=gen.selectors[gen.rows[0]] if gen.uniform else gen.selector,
        verifier=gen.verifier,
        policy=gen.policy,
        costs=costs,
        footprint=draft_footprint(drafter),
        selectors=None if gen.uniform else [gen.selectors[row] for row in gen.rows],
        record_confidence=gen.config.record_confidence,
        record_distributions=gen.config.record_distributions,
        draft_loop_graph=gen.config.draft_loop_graph,
    )
    with phase("draft"):
        proposal = drafter.propose(request)
    if gen.snapshot is not None:
        gen.cache.restore(gen.snapshot)
    if not commits:
        return proposal
    if proposal.is_tree:  # a drafter that drafts trees without declaring it in drafts_trees()
        raise ConfigError(f"{type(drafter).__name__} returned draft trees, which verifier "
                          f"{type(gen.verifier).__name__} commits unverified; it handles chains only")
    tokens = proposal.tokens.tolist()
    finished = []
    for b, (k, cost) in enumerate(zip(proposal.lengths, costs)):
        done = False
        for i in range(k):
            if done := gen.advance(b, tokens[b][i]):
                cost.truncate_after_step(i, marks[b])  # later steps (and their catch-up calls) were wasted
                break
        finished.append(done)
    gen.drop(finished)
    return Draft.empty(len(gen.rows), gen.device)
