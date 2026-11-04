"""Configuration of FR-Spec: draft over the most frequent tokens only, verify over the full vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...validation import check_indices, check_positive
from .._drafting.wrapping import WrappingConfig, drafting_class

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["FRSpecConfig"]


@dataclass(frozen=True)
class FRSpecConfig(WrappingConfig):
    """Frequency-ranked draft vocabulary (FR-Spec, Zhao et al., 2025) around another drafter.

    The inner drafter proposes the argmax (or a sample) of its logits restricted to ``vocab_size`` token ids;
    verification keeps the model's full head, so decoding stays lossless. The ids are, in this order of precedence,
    ``token_ids``, a saved state, or the ``vocab_size`` most frequent tokens of the calibration data passed to
    :func:`economize.apply` (``calibration=``), counted with :func:`~economize.techniques.frspec.frequency_order`.

    Attributes:
        drafter: Configuration of the wrapped drafting technique (e.g. an `EarlyExitConfig`). Its
            parameters are saved with the wrapper under ``inner.``. Techniques that draft without logits (marked
            :class:`~economize.techniques._drafting.NotWrappable`: ``lookup``, ``hybrid``) and FR-Spec itself are
            refused.
        vocab_size: Number of draft token ids ``K``.
        token_ids: The draft ids, most frequent first (``None``: from calibration data or a saved state).
    """

    technique: ClassVar[str] = "frspec"
    nested_fields: ClassVar[tuple[str, ...]] = ("drafter",)

    drafter: TechniqueConfig
    vocab_size: int
    token_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        self.coerce_nested()
        if self.drafter is None:
            raise ConfigError("frspec needs a drafter configuration")
        if self.drafter.technique == self.technique:
            raise ConfigError("frspec cannot wrap another frspec drafter (restrict the vocabulary once)")
        drafting_class(self.drafter, "frspec", "drafter")
        check_positive("vocab_size", self.vocab_size)
        if self.token_ids is not None:
            ids = tuple(int(i) for i in self.token_ids)
            object.__setattr__(self, "token_ids", ids)
            if len(ids) != self.vocab_size:
                raise ConfigError(f"token_ids holds {len(ids)} ids, vocab_size is {self.vocab_size}")
            check_indices("token_ids", ids, distinct=True)

    def validate(self, adapter: ArchAdapter) -> None:
        super().validate(adapter)
        if self.vocab_size > adapter.vocab_size:
            raise ConfigError(f"vocab_size {self.vocab_size} exceeds the model's vocabulary ({adapter.vocab_size})")
        if self.token_ids is not None and max(self.token_ids) >= adapter.vocab_size:
            raise ConfigError(f"token_ids holds ids outside the model's vocabulary ({adapter.vocab_size})")
