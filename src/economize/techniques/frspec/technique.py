"""The FR-Spec technique: a draft vocabulary of frequent tokens around another drafter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import nn

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import AttachContext, Capability, Technique, register_technique
from ...errors import ConfigError
from .._drafting import HeadSliceable, WrappingTechnique
from .config import FRSpecConfig
from .vocab import frequency_order

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...runtime import LayerRunner
    from .drafter import FRSpecDrafter

__all__ = ["FRSpecTechnique"]


@register_technique
class FRSpecTechnique(WrappingTechnique):
    """Restricts the drafts of another drafting technique to a draft vocabulary (FR-Spec, Zhao et al., 2025).

    The wrapped technique is built from ``config.drafter`` and held as the submodule :attr:`inner`; attaching,
    detaching and refreshing the wrapper does the same to it. Train the inner drafter on its own (FR-Spec needs no
    training) and load its weights with ``frspec.inner.load_state_dict(...)``, or save a model that already holds
    the trained drafter inside the wrapper. The draft ids are the buffer :attr:`draft_ids` (most frequent first),
    saved with the technique. When the inner drafter can restrict its LM head
    (:class:`~economize.techniques._drafting.HeadSliceable`), its draft steps compute the logits of the draft ids
    only, which is FR-Spec's saving; otherwise the full logits are masked.

    Example::

        model = economize.apply(base, FRSpecConfig(drafter=EarlyExitConfig(exit_layer=12),
                                                vocab_size=32768), calibration=corpus_ids)
    """

    name = FRSpecConfig.technique
    config_class = FRSpecConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True, trainable=("inner",))
    capabilities = Capability.DRAFTER
    wrapped: ClassVar[dict[str, str]] = {"inner": "drafter"}
    config: FRSpecConfig
    inner: Technique
    draft_ids: torch.Tensor

    def __init__(self, config: FRSpecConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)
        ids = config.token_ids if config.token_ids is not None else (-1,) * config.vocab_size
        self.register_buffer("draft_ids", torch.tensor(ids, dtype=torch.long))

    # ---- draft vocabulary --------------------------------------------------------------------------------------
    @property
    def has_vocab(self) -> bool:
        """Whether the draft ids are set (from the config, a saved state, calibration or :meth:`set_vocab`)."""
        return bool((self.draft_ids >= 0).all())

    def set_vocab(self, token_ids: torch.Tensor | Sequence[int]) -> None:
        """Use ``token_ids`` (``vocab_size`` distinct ids, most frequent first) as the draft vocabulary."""
        ids = torch.as_tensor(token_ids, dtype=torch.long).reshape(-1)
        if ids.numel() != self.config.vocab_size:
            raise ConfigError(f"expected {self.config.vocab_size} token ids, got {ids.numel()}")
        if ids.unique().numel() != ids.numel() or int(ids.min()) < 0 or int(ids.max()) >= self.adapter.vocab_size:
            raise ConfigError(f"draft ids must be distinct ids in [0, {self.adapter.vocab_size})")
        self.draft_ids.copy_(ids.to(self.draft_ids.device))

    def allowed(self, device: torch.device | str | None = None) -> torch.Tensor:
        """Bool mask over the model's vocabulary, true for the draft ids."""
        if not self.has_vocab:
            raise ConfigError("the frspec draft vocabulary is not set: pass token_ids, calibration data to "
                              "economize.apply, or call set_vocab")
        mask = torch.zeros(self.adapter.vocab_size, dtype=torch.bool, device=self.draft_ids.device)
        mask[self.draft_ids] = True
        return mask.to(device) if device is not None else mask

    # ---- technique interface ----------------------------------------------------------------------------------
    def attach(self, model: nn.Module, adapter: ArchAdapter, context: AttachContext | None = None) -> None:
        super().attach(model, adapter, context)
        if not self.has_vocab:
            calibration = None if context is None else context.calibration
            if calibration is None:
                raise ConfigError("frspec needs calibration data (economize.apply(..., calibration=ids)) to rank "
                                  "tokens by frequency, or token_ids in its config")
            self.set_vocab(frequency_order(calibration, self.adapter.vocab_size)[: self.config.vocab_size])

    def drafter(self, runner: LayerRunner) -> FRSpecDrafter:
        from .drafter import FRSpecDrafter

        inner = self.inner.drafter(runner)
        allowed = self.allowed(self.adapter.head_device)
        ids = self.draft_ids.to(self.adapter.head_device)
        sliced = isinstance(inner, HeadSliceable) and inner.slice_head(ids)
        return FRSpecDrafter(inner, allowed, ids if sliced else None)

    def extra_repr(self) -> str:
        return f"drafter={self.config.drafter.technique}, vocab_size={self.config.vocab_size}"
