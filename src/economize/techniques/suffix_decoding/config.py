"""Configuration of SuffixDecoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...validation import check_number, check_positive, check_unit_interval

__all__ = ["SuffixDecodingConfig"]


@dataclass(frozen=True)
class SuffixDecodingConfig(TechniqueConfig):
    """Hyperparameters of SuffixDecoding (Oliaro et al., NeurIPS 2025, "SuffixDecoding: Extreme Speculative Decoding
    for Emerging AI Applications"; the reference implementation is Arctic Inference's suffix cache).

    Two suffix structures are searched every cycle: a global one over a datastore of earlier responses (a
    :class:`~economize.techniques.retrieval.SuffixIndex` folder) and a local one over the row's own prompt and
    output. For every match length ``m = 1, 2, ...`` of the context's suffix (at most ``max_depth - 1``, stopping
    at the first length without an occurrence), the matched node is expanded by the empirical continuation
    probability ``count(child) / count(node)``, multiplied along the path. Path mode follows the most frequent
    child; tree mode expands the most probable frontier node first. A token enters the draft only while its
    probability is at least ``min_token_prob``, and a match of ``m`` tokens drafts at most
    ``min(max_spec_tokens, floor(max_spec_factor * m + max_spec_offset))`` tokens (the adaptive speculation length).
    The draft with the largest score (the sum of its token probabilities; ties go to the longer match) wins, and the
    local draft wins a tie with the global one.

    The defaults are Arctic Inference's: depth 64, factor 1, offset 0, minimum probability 0.1, path mode.

    Attributes:
        datastore: Folder of the global suffix index (``None``: the local structure only). Build it from earlier
            responses with ``python -m economize.techniques.retrieval build``.
        local: Also match the row's own prompt and output.
        max_depth: Longest suffix the structures hold: match plus continuation never exceed it.
        max_spec_factor: Draft tokens per matched token.
        max_spec_offset: Draft tokens added to the adaptive length.
        min_token_prob: Smallest probability of a drafted token.
        tree: Draft a tree (with a verifier that verifies trees); otherwise a chain.
        max_spec_tokens: Most draft tokens (tree nodes in tree mode). ``None``: the cycle's budget,
            ``cycle_length - 1``.
    """

    technique: ClassVar[str] = "suffix_decoding"

    datastore: str | None = None
    local: bool = True
    max_depth: int = 64
    max_spec_factor: float = 1.0
    max_spec_offset: float = 0.0
    min_token_prob: float = 0.1
    tree: bool = False
    max_spec_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.datastore and not self.local:
            raise ConfigError("SuffixDecodingConfig needs a datastore, local matching or both")
        if self.max_depth < 2:
            raise ConfigError(f"max_depth must be at least 2, got {self.max_depth}")
        check_number("max_spec_factor", self.max_spec_factor, low=0.0)
        check_number("max_spec_offset", self.max_spec_offset)
        check_unit_interval("min_token_prob", self.min_token_prob, closed_high=True)
        if self.max_spec_tokens is not None:
            check_positive("max_spec_tokens", self.max_spec_tokens)
