"""Configuration of hybrid drafting: prompt lookup first, another drafter where the lookup finds nothing."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...validation import check_positive
from .._drafting import WrappingConfig, drafting_class
from ..lookup.config import LookupConfig, check_match_rule, check_ngram_range
from ..lookup.matching import MatchRule

__all__ = ["HybridConfig"]


@dataclass(frozen=True)
class HybridConfig(WrappingConfig):
    """Prompt lookup with a fallback drafter.

    Every cycle, each row first tries a prompt lookup (see :class:`~economize.techniques.lookup.LookupConfig`). Rows
    whose last n-gram occurred earlier draft the copied tokens; the other rows draft with ``fallback`` in the same
    cycle. Lookups cost no model call, so the hybrid spends the fallback's draft steps only where copying fails.

    The lookup is the default primary drafter. ``primary`` replaces it with any technique whose drafter is a
    :class:`~economize.techniques._drafting.PartialDrafter` (it serves some rows without a model call); the n-gram
    fields then keep their defaults.

    Attributes:
        fallback: Configuration of the drafting technique used where the primary drafts nothing (e.g. an
            :class:`~economize.techniques.eagle3.Eagle3Config`). Its parameters are saved with the hybrid under
            ``fallback.``.
        max_ngram: Longest suffix the lookup matches.
        min_ngram: Shortest suffix the lookup matches.
        max_proposal: Most tokens one lookup proposes (default: the cycle's budget).
        max_drafts: Most tokens the fallback drafts per cycle (default: the cycle's budget).
        match: Which earlier occurrence the lookup copies from (``"earliest"`` or ``"latest"``, see
            :class:`~economize.techniques.lookup.LookupConfig`).
        primary: Configuration of a primary drafter other than the built-in lookup (``None``: the lookup).
    """

    technique: ClassVar[str] = "hybrid"
    nested_fields: ClassVar[tuple[str, ...]] = ("fallback", "primary")

    fallback: TechniqueConfig
    max_ngram: int = 3
    min_ngram: int = 1
    max_proposal: int | None = None
    max_drafts: int | None = None
    match: MatchRule = "earliest"
    primary: TechniqueConfig | None = None

    def __post_init__(self) -> None:
        self.coerce_nested()
        drafting_class(self.fallback, "a hybrid", "fallback")
        if self.primary is not None:
            drafting_class(self.primary, "a hybrid", "primary", wrappable=False)
            changed = [f.name for f in fields(LookupConfig) if getattr(self, f.name) != f.default]
            if changed:
                raise ConfigError(f"the n-gram fields {changed} configure the built-in lookup; a hybrid with a "
                                  "primary drafter keeps them at their defaults")
        check_ngram_range(self.max_ngram, self.min_ngram)
        check_match_rule(self.match)
        for name in ("max_proposal", "max_drafts"):
            value = getattr(self, name)
            if value is not None:
                check_positive(name, value)

    def lookup_config(self) -> LookupConfig:
        """The configuration of the built-in lookup (from the n-gram fields)."""
        return LookupConfig(self.max_ngram, self.min_ngram, self.max_proposal, self.match)
