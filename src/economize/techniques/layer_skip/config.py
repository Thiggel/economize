"""Configuration of self-speculative whole-layer skipping (a static-set approximation of Draft & Verify, SWIFT and
CLaSp)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...validation import check_choice, check_indices, check_non_negative, check_positive, check_registered
from .metrics import SKIP_METRICS

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["SKIP_METRICS", "LayerSkipConfig"]


@dataclass(frozen=True)
class LayerSkipConfig(TechniqueConfig):
    """Hyperparameters of a whole-layer greedy-skip self-drafter.

    A draft step runs every layer of the model except the skip set and reads the next-token logits with the model's
    own head. The skip set is given (``skip_layers``) or searched when the technique attaches, on the calibration
    data passed to :func:`economize.apply` (``num_skip`` layers, chosen by the registered search ``search``).

    Attributes:
        skip_layers: Layers (1-indexed) the draft path skips. Empty means: search ``num_skip`` layers.
        num_skip: Size of the searched skip set.
        search: Name of a registered skip-set search (:data:`SKIP_SEARCHES`): ``"greedy"`` adds, one at a time, the
            layer whose skipping hurts the metric least; ``"uniform"`` spreads the skipped layers evenly and needs
            no data.
        metric: Name of a registered score minimized by data-driven searches (:data:`SKIP_METRICS`):
            ``"kl"`` is KL(full || skipped) of the next-token distributions, ``"mismatch"`` the share of positions
            where the skipped model's greedy token differs from the full model's.
        keep_first: The first ``keep_first`` layers are never skipped by a search.
        keep_last: The last ``keep_last`` layers are never skipped by a search.
    """

    technique: ClassVar[str] = "layer_skip"

    skip_layers: tuple[int, ...] = ()
    num_skip: int | None = None
    search: str = "greedy"
    metric: str = "kl"
    keep_first: int = 1
    keep_last: int = 1

    def __post_init__(self) -> None:
        if not self.skip_layers and self.num_skip is None:
            raise ConfigError("give the layers to skip (skip_layers) or the size of a searched skip set (num_skip)")
        if self.skip_layers and self.num_skip is not None and self.num_skip != len(set(self.skip_layers)):
            raise ConfigError(f"num_skip={self.num_skip} contradicts skip_layers={self.skip_layers}")
        check_indices("skip_layers", self.skip_layers, low=1, distinct=True)
        if self.num_skip is not None:
            check_positive("num_skip", self.num_skip)
        check_choice("metric", self.metric, SKIP_METRICS.names())
        check_non_negative("keep_first", self.keep_first)
        check_non_negative("keep_last", self.keep_last)

    @property
    def size(self) -> int:
        """Number of skipped layers."""
        return len(self.skip_layers) if self.skip_layers else int(self.num_skip or 0)

    def candidates(self, num_layers: int) -> tuple[int, ...]:
        """Layers a search may skip."""
        return tuple(range(self.keep_first + 1, num_layers - self.keep_last + 1))

    def validate(self, adapter: ArchAdapter) -> None:
        from .search import SKIP_SEARCHES

        L = adapter.num_layers
        if any(layer > L for layer in self.skip_layers):
            raise ConfigError(f"skip_layers {self.skip_layers} name layers beyond the model's {L}")
        if self.size >= L:
            raise ConfigError(f"the draft path must keep at least one of the model's {L} layers")
        if not self.skip_layers:
            check_registered("skip-set search", self.search, SKIP_SEARCHES)
            if self.size > len(self.candidates(L)):
                raise ConfigError(
                    f"num_skip={self.num_skip} exceeds the {len(self.candidates(L))} layers a search may skip "
                    f"(keep_first={self.keep_first}, keep_last={self.keep_last})"
                )
