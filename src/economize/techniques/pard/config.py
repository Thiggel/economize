"""Configuration of PARD, the parallel draft LM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ...errors import ConfigError
from ..draft_model.config import DraftModelConfig

__all__ = ["PardConfig"]


@dataclass(frozen=True)
class PardConfig(DraftModelConfig):
    """A :class:`~economize.techniques.draft_model.DraftModelConfig` whose draft LM drafts in parallel (PARD).

    One draft-LM call per cycle reads the owed committed tokens followed by ``drafts - 1`` copies of the mask token
    at the positions after them and yields all drafts of the cycle at once. A cycle drafts at most
    ``max_parallel`` tokens (``DecodeConfig.cycle_length - 1`` larger than that is cut to it).

    Attributes:
        mask_token_id: Token id of the mask (the same id at every depth, as in the reference code). ``None`` reads
            ``pard_token`` from the draft LM's config, which the reference training and ``scripts/pard31_train.py``
            write.
        max_parallel: The most drafts of one call: the number of blocks the draft LM was trained with (the
            reference's ``para_num``, 8 for the published checkpoints).
    """

    technique: ClassVar[str] = "pard"

    mask_token_id: int | None = None
    max_parallel: int = 8

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mask_token_id is not None and (isinstance(self.mask_token_id, bool) or self.mask_token_id < 0):
            raise ConfigError(f"PardConfig.mask_token_id must be a token id >= 0, got {self.mask_token_id!r}")
        if isinstance(self.max_parallel, bool) or not isinstance(self.max_parallel, int) or self.max_parallel < 1:
            raise ConfigError(f"PardConfig.max_parallel must be an int >= 1, got {self.max_parallel!r}")
