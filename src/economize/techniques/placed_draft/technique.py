"""The placed two-model technique: loads the draft LM on its own device and builds its drafter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import AttachContext, register_technique
from ...errors import ConfigError
from ..draft_model.technique import DraftModelTechnique, check_vocabularies, load_draft_model, load_local_tokenizer
from .config import PlacedDraftModelConfig
from .drafter import PlacedDraftModelDrafter

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...runtime import LayerRunner

__all__ = ["PlacedDraftModelTechnique"]


@register_technique
class PlacedDraftModelTechnique(DraftModelTechnique):
    """:class:`~economize.techniques.draft_model.DraftModelTechnique` with the draft LM on
    :attr:`PlacedDraftModelConfig.device`.

    The cost log is the same as the one-device technique's (events of the network ``"draft_model"``); the device
    count belongs to the hardware description of a benchmark.
    """

    name = PlacedDraftModelConfig.technique
    config_class = PlacedDraftModelConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True)
    config: PlacedDraftModelConfig

    @property
    def draft_device(self) -> torch.device:
        return torch.device(self.config.device)

    def attach(self, model: nn.Module, adapter: ArchAdapter, context: AttachContext | None = None) -> None:
        """Load the draft LM on the draft device and check its vocabulary against the model's."""
        if not self._draft:
            self._draft.append(load_draft_model(self.config, self.draft_device, adapter.dtype))
            self._runner = None
            source = getattr(getattr(model, "config", None), "_name_or_path", "") or ""
            try:
                check_vocabularies(load_local_tokenizer(str(source)), load_local_tokenizer(self.config.model),
                                   adapter.vocab_size, self.draft_runner.adapter.vocab_size,
                                   draft=repr(self.config.model))
            except ConfigError:
                self.detach_draft()
                raise

    def drafter(self, runner: LayerRunner) -> PlacedDraftModelDrafter:
        return PlacedDraftModelDrafter(self.draft_runner, self.adapter.vocab_size, home=self.adapter.device)

    def extra_repr(self) -> str:
        return f"model={self.config.model!r}, device={self.config.device!r}"
