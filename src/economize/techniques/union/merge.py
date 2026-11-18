"""Merging the drafts of several drafters into one draft tree, and reading which drafter an accepted path came from."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ...decoding.protocols import Draft
from ...errors import ConfigError
from ..lookup.proposal import one_hot_probs

__all__ = ["MergedTree", "TreeBuilder", "accepted_path", "build_tree", "leading_counts", "merge_drafts", "path_source"]

#: The source of a path that accepted no draft.
NONE = "none"
#: The source of a path whose deepest node more than one member proposed.
SHARED = "shared"


@dataclass(frozen=True)
class MergedTree:
    """A draft tree with the members behind every node.

    Attributes:
        draft: The tree as the verifier sees it (``parents`` set, ``-1`` padding).
        parents: Parent of every node of every row (``-1``: a child of the root), host lists.
        masks: Bit mask of the members that proposed every node (bit ``i``: member ``i``), host lists.
        depth: Depth of the deepest node of any row (``0`` for an empty tree).
    """

    draft: Draft
    parents: list[list[int]]
    masks: list[list[int]]
    depth: int


class TreeBuilder:
    """Collects the nodes of one row's draft tree in topological order.

    Args:
        dedupe: Merge a node into an existing child of the same parent with the same token (prefix merging). Without
            it every :meth:`add` makes a new node, as a verifier that needs every node's own draft distribution
            requires.
    """

    def __init__(self, dedupe: bool) -> None:
        self.dedupe = dedupe
        self.tokens: list[int] = []
        self.parents: list[int] = []
        self.masks: list[int] = []
        self.depths: list[int] = []
        #: (member, column in that member's draft) of every node, for its draft distribution.
        self.origins: list[tuple[int, int]] = []
        self._index: dict[tuple[int, int], int] = {}

    def add(self, parent: int, token: int, member: int, column: int = -1) -> int:
        """The node of ``token`` below ``parent`` (``-1``: the root) proposed by ``member``; ``column`` is the
        node's column in that member's draft (``-1`` when it has none)."""
        key = (parent, token)
        if self.dedupe and key in self._index:
            node = self._index[key]
            self.masks[node] |= 1 << member
            return node
        node = len(self.tokens)
        self.tokens.append(token)
        self.parents.append(parent)
        self.masks.append(1 << member)
        self.depths.append(1 if parent < 0 else self.depths[parent] + 1)
        self.origins.append((member, column))
        self._index.setdefault(key, node)
        return node


def build_tree(rows: Sequence[TreeBuilder], device: torch.device | str,
               member_probs: Sequence[torch.Tensor | None] | None = None,
               vocab_size: int | None = None) -> MergedTree:
    """The draft tree of the rows ``rows``.

    Args:
        rows: One builder per batch row.
        device: Device of the draft's tensors.
        member_probs: When given, the draft distribution of every member's draft ``[B, T_i, V]`` (``None`` for a
            member without one: its nodes get point masses), copied to every node from its origin.
        vocab_size: Vocabulary size (for point masses).
    """
    batch = len(rows)
    lengths = [len(row.tokens) for row in rows]
    width = max(lengths, default=0)
    parents = [list(row.parents) for row in rows]
    masks = [list(row.masks) for row in rows]
    depth = max((max(row.depths, default=0) for row in rows), default=0)
    if width == 0:
        return MergedTree(Draft.empty(batch, device), parents, masks, 0)
    tokens = torch.tensor([row.tokens + [-1] * (width - len(row.tokens)) for row in rows], dtype=torch.long,
                          device=device)
    parent_t = torch.tensor([row.parents + [-1] * (width - len(row.parents)) for row in rows], dtype=torch.long,
                            device=device)
    probs = None
    if member_probs is not None:
        if vocab_size is None:
            raise ConfigError("build_tree needs vocab_size for draft distributions")
        probs = one_hot_probs(tokens, vocab_size).to(device)
        for member, source in enumerate(member_probs):
            if source is None:
                continue
            picks = [(b, node, column) for b, row in enumerate(rows)
                     for node, (owner, column) in enumerate(row.origins) if owner == member and column >= 0]
            if not picks:
                continue
            b_index, nodes, columns = (torch.tensor(part, dtype=torch.long, device=device) for part in zip(*picks))
            probs[b_index, nodes] = source.to(device, probs.dtype)[b_index.to(source.device), columns.to(source.device)]
    return MergedTree(Draft(tokens, lengths, parent_t, probs), parents, masks, depth)


def merge_drafts(drafts: Sequence[Draft], *, dedupe: bool, keep_probs: bool = False,
                 vocab_size: int | None = None) -> MergedTree:
    """The drafts of several members merged into one tree per row, member ``i`` being ``drafts[i]``.

    Every member's chain or tree hangs below the root. With ``dedupe`` a node that another member already proposed
    (same parent, same token) is shared, so the tree is the prefix-merged trie of all drafts and verifying it accepts
    the longest path any member drafted (greedy verification). Without ``dedupe`` the members' drafts stay side by
    side as siblings, each node with its own draft distribution (``keep_probs``), as recursive rejection sampling
    needs: every member drew its draft independently of the others.

    Raises:
        ConfigError: A draft marks some nodes as non-candidates (:attr:`Draft.candidates`), or the drafts disagree
            on the batch size.
    """
    if not drafts:
        raise ConfigError("merge_drafts needs at least one draft")
    batch = len(drafts[0].lengths)
    if any(len(d.lengths) != batch for d in drafts):
        raise ConfigError("the merged drafts must describe the same rows")
    if any(d.candidates is not None for d in drafts):
        raise ConfigError("a merged draft cannot hold non-candidate nodes (Draft.candidates)")
    device = next((d.tokens.device for d in drafts if d.width), drafts[0].tokens.device)
    rows = [TreeBuilder(dedupe) for _ in range(batch)]
    for member, d in enumerate(drafts):
        if not d.width:
            continue
        tokens = d.tokens.tolist()
        parents = d.parents.tolist() if d.parents is not None else None
        for b, count in enumerate(d.lengths):
            local: list[int] = []
            for j in range(count):
                up = (j - 1) if parents is None else parents[b][j]
                local.append(rows[b].add(local[up] if up >= 0 else -1, tokens[b][j], member, j))
    member_probs = [d.probs for d in drafts] if keep_probs else None
    return build_tree(rows, device, member_probs, vocab_size)


def accepted_path(parents: Sequence[int], node: int) -> list[int]:
    """The nodes from the root's child down to ``node`` (``-1``: the empty path)."""
    path = []
    while node >= 0:
        path.append(node)
        node = parents[node]
    return path[::-1]


def leading_counts(masks: Sequence[int], path: Sequence[int], members: int) -> list[int]:
    """For every member, the number of leading nodes of ``path`` it proposed (its accepted length had it drafted
    alone, under greedy verification of chain drafts)."""
    counts = []
    for member in range(members):
        bit, n = 1 << member, 0
        for node in path:
            if not masks[node] & bit:
                break
            n += 1
        counts.append(n)
    return counts


def path_source(masks: Sequence[int], path: Sequence[int]) -> str:
    """Which member the accepted ``path`` came from: the index of the only member that proposed its deepest node,
    ``"shared"`` when several did, ``"none"`` for the empty path."""
    if not path:
        return NONE
    mask = masks[path[-1]]
    if mask & (mask - 1):
        return SHARED
    return str(mask.bit_length() - 1)
