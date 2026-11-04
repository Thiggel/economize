"""Configuration of two-model speculative decoding with the draft LM on a device of its own."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from ...errors import ConfigError
from ..draft_model.config import DraftModelConfig

__all__ = ["PlacedDraftModelConfig"]


@dataclass(frozen=True)
class PlacedDraftModelConfig(DraftModelConfig):
    """A :class:`~economize.techniques.draft_model.DraftModelConfig` whose draft LM lives on ``device``.

    Attributes:
        device: Device of the draft LM and its KV cache (``"cuda:1"``: the second GPU). The model stays where it
            is; per draft step the draft token goes to ``device`` and the draft's float logits ``[B, V]`` come back.
    """

    technique: ClassVar[str] = "placed_draft_model"

    device: str = "cuda:1"

    def __post_init__(self) -> None:
        super().__post_init__()
        try:
            torch.device(self.device)
        except (RuntimeError, TypeError) as error:
            raise ConfigError(f"PlacedDraftModelConfig.device {self.device!r} is not a torch device: {error}") from None
