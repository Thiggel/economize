"""Static sparse draft trees for Medusa heads.

A tree is a tuple of paths. The path ``(r_1, ..., r_j)`` is the node whose token is head ``j``'s rank-``r_j``
candidate, child of the node ``(r_1, ..., r_{j-1})`` (the empty path is the last committed token). Trees are
registered in :data:`MEDUSA_TREES` by name (open for extension)::

    from economize.techniques.medusa import MEDUSA_TREES

    MEDUSA_TREES.register("my_tree", ((0,), (1,), (0, 0)))
    config = MedusaConfig(tree="my_tree")
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from ...errors import ConfigError
from ...registry import Registry

__all__ = ["MEDUSA_TREES", "Paths", "TreePlan", "normalize_paths"]

Paths = tuple[tuple[int, ...], ...]

#: Named draft trees.
MEDUSA_TREES: Registry[Paths] = Registry("Medusa tree", "economize.medusa_trees")

# ``mc_sim_7b_63`` of FasterDecoding/Medusa (medusa/model/medusa_choices.py): 63 nodes, depth 4, ranks below 10,
# calibrated on Vicuna-7B. Copied verbatim.
_MC_SIM_7B_63 = [
    [0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1],
    [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0],
    [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6],
    [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2],
    [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1],
    [1, 6], [0, 7, 0],
]


def normalize_paths(paths: Sequence[Sequence[int]]) -> Paths:
    """Validate a tree and return it as a tuple of int tuples, in its given order.

    Raises:
        ConfigError: A path is empty, has a negative rank, is repeated or lacks its parent.
    """
    out = tuple(tuple(int(r) for r in path) for path in paths)
    if not out:
        raise ConfigError("a Medusa tree needs at least one path")
    seen = set(out)
    if len(seen) != len(out):
        raise ConfigError("a Medusa tree lists a path twice")
    for path in out:
        if not path or min(path) < 0:
            raise ConfigError(f"Medusa tree paths are non-empty tuples of ranks >= 0, got {path}")
        if len(path) > 1 and path[:-1] not in seen:
            raise ConfigError(f"Medusa tree path {path} has no parent {path[:-1]} in the tree")
    return out


MEDUSA_TREES.register("mc_sim_7b_63", normalize_paths(_MC_SIM_7B_63))
#: A chain of every head's top-1 token plus the runner-up of the first two heads (a small tree for tests and CPUs).
MEDUSA_TREES.register("small", normalize_paths([[0], [1], [0, 0], [0, 1], [0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0]]))


@dataclass
class TreePlan:
    """A tree laid out for drafting: nodes in topological order, with their head, rank and parent.

    Attributes:
        paths: The tree in Medusa's node order: shorter paths first, then lexicographic (parents before children).
        top_k: Candidates needed per head (largest rank + 1).
        depth: Longest path (heads needed).
    """

    paths: Paths
    top_k: int = field(init=False)
    depth: int = field(init=False)
    _layouts: dict[int, tuple[list[int], list[int], list[int]]] = field(
        init=False, default_factory=dict, compare=False, repr=False)
    _indices: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        init=False, default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        self.paths = tuple(sorted(self.paths, key=lambda p: (len(p), p)))
        self.top_k = 1 + max(max(p) for p in self.paths)
        self.depth = max(len(p) for p in self.paths)

    def size(self, depth: int) -> int:
        """Nodes of the tree cut at ``depth`` levels."""
        return sum(1 for p in self.paths if len(p) <= depth)

    def layout(self, depth: int) -> tuple[list[int], list[int], list[int]]:
        """``(heads, ranks, parents)`` of the nodes of the tree cut at ``depth``: node ``i`` is the rank-``ranks[i]``
        candidate of head ``heads[i]`` (0-based) and child of node ``parents[i]`` (``-1``: the committed token)."""
        if depth not in self._layouts:
            kept = [p for p in self.paths if len(p) <= depth]
            index = {p: i for i, p in enumerate(kept)}
            self._layouts[depth] = (
                [len(p) - 1 for p in kept],
                [p[-1] for p in kept],
                [index[p[:-1]] if len(p) > 1 else -1 for p in kept],
            )
        return self._layouts[depth]

    def indices(self, depth: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """:meth:`layout` as long tensors on ``device``, built once per depth and device (no transfer per draft)."""
        key = (depth, torch.device(device))
        if key not in self._indices:
            self._indices[key] = tuple(torch.tensor(v, dtype=torch.long, device=device)  # type: ignore[assignment]
                                       for v in self.layout(depth))
        return self._indices[key]

    def tokens(self, candidates: torch.Tensor, depth: int) -> torch.Tensor:
        """Node tokens ``[B, n]`` from every head's top candidates ``[B, heads, top_k]``."""
        heads, ranks, _ = self.indices(depth, candidates.device)
        return candidates[:, heads, ranks]


def resolve_tree(tree: str | Paths) -> Paths:
    """The paths of a registered tree name, or ``tree`` itself."""
    return MEDUSA_TREES.get(tree) if isinstance(tree, str) else tree
