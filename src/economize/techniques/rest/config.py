"""Configuration of REST, retrieval-based speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final, Literal

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...validation import check_choice, check_positive

__all__ = ["OCCURRENCE_RULES", "OccurrenceRule", "RestConfig"]

OccurrenceRule = Literal["first", "spread"]
OCCURRENCE_RULES: Final[tuple[str, ...]] = ("first", "spread")


@dataclass(frozen=True)
class RestConfig(TechniqueConfig):
    """Hyperparameters of REST (He et al., 2024, "REST: Retrieval-Based Speculative Decoding").

    Every cycle, the last ``n`` committed tokens of a row are searched in the datastore for
    ``n = max_span, ..., min_span``, longest first, and the first ``n`` with an occurrence is used. The
    ``max_depth`` tokens that follow each of up to ``max_occurrences`` occurrences (within one datastore record)
    form a trie in which every node counts the continuations through it. The ``tree_size`` most frequent nodes (a
    prefix-closed subtree; ties go to the shallower node, then to the lower token) are the tree draft, which the
    verifier checks in one exact pass. In chain mode the draft is the trie's most frequent path (at every node the
    child with the largest count). A row without any occurrence drafts nothing in that cycle.

    The defaults are the paper's (and its ``draftretriever`` library's): spans 16 down to 2, ``k = 5000``
    occurrences, continuations of 10 tokens and draft trees of 64 nodes. A tree path never exceeds the cycle's
    budget, so ``cycle_length = max_depth + 1`` lets every path be accepted whole.

    Attributes:
        datastore: Folder of the :class:`~economize.techniques.retrieval.SuffixIndex` of the datastore (build it
            with ``python -m economize.techniques.retrieval build``).
        max_span: Longest suffix of the context that is searched.
        min_span: Shortest suffix of the context that is searched.
        max_occurrences: Most occurrences whose continuations enter the trie (the paper's ``k``).
        max_depth: Most tokens of every continuation (the trie's depth, ``draftretriever``'s ``long``).
        tree: Draft the trie's top nodes as a tree (with a verifier that verifies trees); otherwise the most frequent
            path as a chain.
        tree_size: Most nodes of a tree draft (the paper's 64 "choices").
        occurrences: Which occurrences are read when there are more than ``max_occurrences``: ``"first"`` (the first
            ones in suffix-array order, as ``draftretriever``) or ``"spread"`` (evenly spaced over all of them).
    """

    technique: ClassVar[str] = "rest"

    datastore: str
    max_span: int = 16
    min_span: int = 2
    max_occurrences: int = 5000
    max_depth: int = 10
    tree: bool = True
    tree_size: int = 64
    occurrences: OccurrenceRule = "first"

    def __post_init__(self) -> None:
        if not self.datastore:
            raise ConfigError("RestConfig needs a datastore folder")
        if not 1 <= self.min_span <= self.max_span:
            raise ConfigError(f"need 1 <= min_span <= max_span, got min_span={self.min_span}, "
                              f"max_span={self.max_span}")
        check_positive("max_occurrences", self.max_occurrences)
        check_positive("max_depth", self.max_depth)
        check_positive("tree_size", self.tree_size)
        check_choice("occurrences", self.occurrences, OCCURRENCE_RULES)
