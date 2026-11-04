"""Configuration of prompt-lookup (n-gram) drafting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...validation import check_choice, check_positive
from .matching import MATCH_RULES, MatchRule

__all__ = ["LookupConfig", "check_match_rule", "check_ngram_range"]


@dataclass(frozen=True)
class LookupConfig(TechniqueConfig):
    """Hyperparameters of prompt-lookup drafting (Saxena, 2023; vLLM's ``"ngram"`` method).

    Every cycle, the last ``n`` committed tokens of a row are searched in all its earlier tokens (prompt and
    output), for ``n = max_ngram, ..., min_ngram``, longest first. The tokens that followed the earliest (or, with
    ``match="latest"``, the most recent) earlier occurrence become the draft. A row without a match drafts nothing in
    that cycle (the exact pass then decodes one token, as plain decoding would).

    Attributes:
        max_ngram: Longest suffix that is matched.
        min_ngram: Shortest suffix that is matched.
        max_proposal: Most tokens one match proposes (default: the cycle's budget, ``cycle_length - 1``).
        match: Which occurrence is copied: ``"earliest"`` (Saxena's prompt lookup, Hugging Face and vLLM) or
            ``"latest"`` (the most recent occurrence).
    """

    technique: ClassVar[str] = "lookup"

    max_ngram: int = 3
    min_ngram: int = 1
    max_proposal: int | None = None
    match: MatchRule = "earliest"

    def __post_init__(self) -> None:
        check_ngram_range(self.max_ngram, self.min_ngram)
        check_match_rule(self.match)
        if self.max_proposal is not None:
            check_positive("max_proposal", self.max_proposal)


def check_ngram_range(max_ngram: int, min_ngram: int) -> None:
    """Raise :class:`ConfigError` unless ``1 <= min_ngram <= max_ngram``."""
    if not 1 <= min_ngram <= max_ngram:
        raise ConfigError(f"need 1 <= min_ngram <= max_ngram, got min_ngram={min_ngram}, max_ngram={max_ngram}")


def check_match_rule(match: str) -> None:
    """Raise :class:`ConfigError` unless ``match`` is ``"earliest"`` or ``"latest"``."""
    check_choice("match", match, MATCH_RULES)
