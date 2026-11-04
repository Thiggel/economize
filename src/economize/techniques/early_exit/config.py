"""Configuration of self-speculative early-exit drafting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...modules import COPY_DTYPES
from ...validation import check_choice, check_positive

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["EXIT_ADAPTATIONS", "EXIT_HEADS", "EarlyExitConfig"]

#: How the exit reads logits: the model's own final norm and LM head, or a trained readout adapter before them.
EXIT_HEADS = ("shared", "readout")
#: How layers ``1..r`` are adapted on the draft path: not at all, with LoRA, or as trainable copies.
EXIT_ADAPTATIONS = ("frozen", "lora", "full")


@dataclass(frozen=True)
class EarlyExitConfig(TechniqueConfig):
    """Hyperparameters of an early-exit drafter.

    A draft step runs the model's first ``exit_layer`` (``r``) layers on the last token and reads the next-token
    logits there. ``head="shared"`` with ``adaptation="frozen"`` is the self-speculative decoding of LayerSkip
    (Elhoushi et al., 2024), whose base model is trained with the ``layerskip`` objective so that its intermediate
    layers can exit. ``head="readout"`` adds a trainable readout adapter before the model's norm and head
    (Kangaroo's adapter plays the same role).

    Attributes:
        exit_layer: ``r``, the last layer of a draft step (``1 <= r < L``).
        head: ``"shared"`` (final norm and LM head of the model) or ``"readout"`` (a readout adapter first).
        adaptation: ``"frozen"`` (the draft path runs the model's layers), ``"lora"`` (LoRA updates active only on
            the draft path) or ``"full"`` (trainable copies of layers ``1..r``).
        lora_rank: Rank of the LoRA updates.
        lora_alpha: LoRA scaling numerator (default ``2 * lora_rank``).
        copy_dtype: Storage dtype of the trainable copies (``adaptation="full"``): ``"float32"``, or ``"model"``
            for the model's dtype, which halves their memory in a bf16 model and is meant for training with
            float32 master weights (``TrainConfig(offload_optimizer=True)``).
    """

    technique: ClassVar[str] = "early_exit"

    exit_layer: int
    head: str = "readout"
    adaptation: str = "frozen"
    lora_rank: int = 16
    lora_alpha: float | None = None
    copy_dtype: str = "float32"

    def __post_init__(self) -> None:
        check_choice("copy_dtype", self.copy_dtype, COPY_DTYPES)
        check_positive("exit_layer", self.exit_layer)
        check_choice("head", self.head, EXIT_HEADS)
        check_choice("adaptation", self.adaptation, EXIT_ADAPTATIONS)
        check_positive("lora_rank", self.lora_rank)

    @property
    def resolved_lora_alpha(self) -> float:
        return float(2 * self.lora_rank if self.lora_alpha is None else self.lora_alpha)

    def validate(self, adapter: ArchAdapter) -> None:
        if self.exit_layer >= adapter.num_layers:
            raise ConfigError(
                f"exit_layer must be below the model's {adapter.num_layers} layers, got {self.exit_layer}"
            )
