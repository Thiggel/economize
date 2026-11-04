"""Configuration of KnapSpec (Cha et al., ICML 2026): self-speculative drafting with a sublayer skip set chosen by a
knapsack search over measured latencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError, UnsupportedModelError
from ...validation import check_indices, check_non_negative, check_number, check_positive, check_unit_interval

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["KnapSpecConfig"]


@dataclass(frozen=True)
class KnapSpecConfig(TechniqueConfig):
    """Hyperparameters of a KnapSpec self-drafter.

    A draft step runs the model with the attention of some layers and the MLP of others skipped. Every
    ``interval`` decoded tokens the drafter searches a new skip set and draft length on the hidden states of its
    recent exact passes: a dynamic program over the ``2 L`` sublayers and their skipped latency keeps, per budget,
    the candidate closest (cosine) to the full model, and the pair of skip set and draft length with the highest
    modelled tokens per unit time wins. The defaults are the paper's.

    Attributes:
        min_similarity: Candidates whose mean cosine similarity to the full model's state falls below it are
            pruned (the paper's ``tau = 0.5``).
        history: Number of recent exact passes whose committed tokens the search runs on (the paper's ``m = 5``
            speculation steps).
        max_draft: Largest draft length the search may choose (the paper's ``D = 10``). The decoding loop's
            ``cycle_length - 1`` caps it further; the confidence exit of the paper (``tau_conf = 0.7``) is the
            loop's ``min_confidence``.
        interval: Decoded tokens between two searches (the paper's ``T``: 64 below 10B parameters, 128 above).
        prompt_tokens: Most committed tokens of one exact pass the search reads: the prompt's tail for the first
            search (the paper does not say what the first search runs on), all of a later pass at the usual cycle
            lengths.
        max_skip_fraction: Largest share of the model's latency weight a skip set may remove (the paper's
            ``K / 2``).
        keep_first: The first ``keep_first`` layers are never skipped (neither sublayer). The paper skips any.
        keep_last: The last ``keep_last`` layers are never skipped.
        latency: Measured sublayer latencies ``(t_mlp, attention_slope, attention_base)`` in seconds, with
            ``t_attn(n) = attention_slope * n + attention_base`` at context length ``n``. ``None``: measured on the
            model's device the first time the drafter starts.
        calibration_lengths: Context lengths of the latency measurement.
        calibration_repeats: Timed calls per context length.
        max_graphs: CUDA graphs of draft steps kept at once, one per skip set (least recently used dropped).
    """

    technique: ClassVar[str] = "knapspec"

    min_similarity: float = 0.5
    history: int = 5
    max_draft: int = 10
    interval: int = 128
    prompt_tokens: int = 16
    max_skip_fraction: float = 0.5
    keep_first: int = 0
    keep_last: int = 0
    latency: tuple[float, float, float] | None = None
    calibration_lengths: tuple[int, ...] = (64, 512, 2048, 8192)
    calibration_repeats: int = 20
    max_graphs: int = 8

    def __post_init__(self) -> None:
        check_number("min_similarity", self.min_similarity, low=-1, high=1)
        check_positive("history", self.history)
        check_positive("max_draft", self.max_draft)
        check_positive("interval", self.interval)
        check_positive("prompt_tokens", self.prompt_tokens)
        check_unit_interval("max_skip_fraction", self.max_skip_fraction, closed_low=False, closed_high=True)
        check_non_negative("keep_first", self.keep_first)
        check_non_negative("keep_last", self.keep_last)
        if self.latency is not None:
            if len(self.latency) != 3:
                raise ConfigError(f"latency needs (t_mlp, attention_slope, attention_base), got {self.latency}")
            mlp, slope, base = (float(v) for v in self.latency)
            if mlp <= 0 or slope < 0 or base <= 0:
                raise ConfigError(f"latency needs t_mlp > 0, attention_slope >= 0, attention_base > 0, "
                                  f"got {self.latency}")
        if len(self.calibration_lengths) < 2:
            raise ConfigError("calibration_lengths needs at least two context lengths to fit t_attn(n)")
        check_indices("calibration_lengths", self.calibration_lengths, low=1, distinct=True)
        check_positive("calibration_repeats", self.calibration_repeats)
        check_positive("max_graphs", self.max_graphs)

    def candidates(self, num_layers: int) -> tuple[int, ...]:
        """Layers whose sublayers the search may skip."""
        return tuple(range(self.keep_first + 1, num_layers - self.keep_last + 1))

    def validate(self, adapter: ArchAdapter) -> None:
        from ...arch.spec import StateKind

        L = adapter.num_layers
        if self.keep_first + self.keep_last > L:
            raise ConfigError(f"keep_first + keep_last exceeds the model's {L} layers")
        if any(adapter.state_kind(i) is StateKind.RECURRENT for i in range(1, L + 1)):
            # The search runs every layer on many copies of the recent tokens, which would advance recurrent states.
            raise UnsupportedModelError("knapspec supports attention-only models (this one has recurrent layers)")
