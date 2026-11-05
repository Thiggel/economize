"""The PARD technique: loads the parallel draft LM and builds its drafter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import register_technique
from ...errors import ConfigError
from ..draft_model.technique import DraftModelTechnique
from .config import PardConfig
from .drafter import PardDrafter

if TYPE_CHECKING:
    from ...runtime import LayerRunner

__all__ = ["PardTechnique"]


@register_technique
class PardTechnique(DraftModelTechnique):
    """PARD: a separate draft model that drafts several tokens in one parallel forward pass.

    :class:`~economize.techniques.draft_model.DraftModelTechnique` with the parallel drafter of PARD.

    The cost log is the one of the two-model technique (events of the network ``"draft_model"``): the owed tokens
    beyond the first as a ``"draft_catchup"`` event and one ``"draft"`` event per draft, so the logged tokens are
    the tokens of the cycle's single draft-LM call (each mask stands for the draft it yields).
    """

    name = PardConfig.technique
    config_class = PardConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True)
    config: PardConfig

    @property
    def mask_token_id(self) -> int:
        """The mask token id of the configuration, or ``pard_token`` of the draft LM's config."""
        mask = self.config.mask_token_id
        if mask is None:
            mask = getattr(self.draft_model.config, "pard_token", None)
        if mask is None:
            raise ConfigError(f"PardConfig(model={self.config.model!r}) needs mask_token_id: the draft LM's config "
                              "has no pard_token")
        vocab = self.draft_runner.adapter.vocab_size
        if not 0 <= int(mask) < vocab:
            raise ConfigError(f"PARD mask token {mask} lies outside the draft LM's vocabulary of {vocab} tokens")
        return int(mask)

    def drafter(self, runner: LayerRunner) -> PardDrafter:
        return PardDrafter(self.draft_runner, self.adapter.vocab_size, self.mask_token_id, self.config.max_parallel)

    def extra_repr(self) -> str:
        return f"model={self.config.model!r}, max_parallel={self.config.max_parallel}"
