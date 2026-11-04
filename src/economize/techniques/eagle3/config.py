"""Configuration of EAGLE-3 drafting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ...errors import ConfigError
from ...validation import check_choice, check_indices, check_positive
from .._draft_layer import DraftLayerConfig

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["EAGLE3_HEADS", "Eagle3Config", "default_aux_layers"]

#: How the draft layer's output becomes logits: the model's final norm and LM head, or the drafter's own.
EAGLE3_HEADS = ("own", "shared")


def default_aux_layers(num_layers: int) -> tuple[int, ...]:
    """EAGLE-3's low, middle and high feature layers ``(2, L // 2, L - 3)`` (outputs of 1-indexed blocks), clamped
    to ``0..L`` and without repeats for very shallow models."""
    return tuple(sorted({min(max(layer, 0), num_layers) for layer in (2, num_layers // 2, num_layers - 3)}))


@dataclass(frozen=True)
class Eagle3Config(DraftLayerConfig):
    """Hyperparameters of an EAGLE-3 draft head (Li et al., 2025, arXiv 2503.01840).

    The drafter fuses the model's hidden states at ``aux_layers`` (the outputs of those blocks, ``0`` being the
    token embedding) of token ``t - 1`` with ``fc`` into one vector ``g``, and runs one decoder layer with its own
    attention and KV cache on ``[norm(E[x_t]); norm(g)]`` at position ``t``. The layer's output drafts token
    ``t + 1`` and replaces ``g`` at the next draft step. The draft layer's shape, rotary table and training options
    are those of :class:`~economize.techniques._draft_layer.DraftLayerConfig`.

    Attributes:
        aux_layers: Layers whose states are fused (``None``: ``(2, L // 2, L - 3)``, the paper's choice).
        head: ``"own"`` (EAGLE-3: own final norm and LM head, initialized from the model's) or ``"shared"`` (the
            model's frozen final norm and LM head).
        draft_vocab: Size of the own head's vocabulary (``None``: the model's). Draft token ``i`` is model token
            ``i + d2t[i]``; :meth:`Eagle3Technique.set_draft_vocab` chooses the tokens.
        norm_before_residual: The residual of the attention block is the normalized ``g`` (SpecForge option).
    """

    technique: ClassVar[str] = "eagle3"

    aux_layers: tuple[int, ...] | None = None
    head: str = "own"
    draft_vocab: int | None = None
    norm_before_residual: bool = False

    def __post_init__(self) -> None:
        if self.aux_layers is not None:
            layers = tuple(int(layer) for layer in self.aux_layers)
            check_indices("aux_layers", layers, allow_empty=False)
            object.__setattr__(self, "aux_layers", layers)
        check_choice("head", self.head, EAGLE3_HEADS)
        if self.draft_vocab is not None:
            check_positive("draft_vocab", self.draft_vocab)
            if self.head != "own":
                raise ConfigError("draft_vocab needs head='own' (restrict a shared head with the frspec technique)")
        super().__post_init__()

    def layers(self, num_layers: int) -> tuple[int, ...]:
        """The fused layers for a model with ``num_layers`` blocks."""
        return self.aux_layers if self.aux_layers is not None else default_aux_layers(num_layers)

    def validate(self, adapter: ArchAdapter) -> None:
        layers = self.layers(adapter.num_layers)
        if max(layers) > adapter.num_layers:
            raise ConfigError(f"aux_layers {layers} exceed the model's {adapter.num_layers} layers")
        if self.draft_vocab is not None and self.draft_vocab > adapter.vocab_size:
            raise ConfigError(f"draft_vocab {self.draft_vocab} exceeds the model's vocabulary ({adapter.vocab_size})")
        super().validate(adapter)
