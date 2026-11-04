"""Registered pruning criteria. A new criterion is one :class:`Pruner` subclass registered in :data:`PRUNE_METHODS`::

    from economize.techniques.prune import PRUNE_METHODS, Pruner

    @PRUNE_METHODS.register("my_criterion")
    class MyPruner(Pruner):
        options = frozenset({"percdamp"})  # the PruneConfig knobs it reads besides the generic ones

        def observer(self, module): ...  # an Observer of the module's inputs, or None (data-free)
        def prune(self, weight, stats, pattern, config): ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from torch import Tensor, nn

from ...core.calibration import GramObserver, Observer, SquaredNormObserver
from ...errors import ConfigError
from ...registry import Registry
from .masks import SparsityPattern
from .sparsegpt import sparsegpt_prune
from .wanda import magnitude_mask, wanda_mask

if TYPE_CHECKING:
    from .config import PruneConfig

__all__ = ["PRUNE_METHODS", "MagnitudePruner", "Pruner", "SparseGPTPruner", "WandaPruner"]


class Pruner(ABC):
    """One pruning criterion applied to one linear map at a time."""

    #: Whether :meth:`prune` needs input statistics from calibration data.
    needs_calibration: ClassVar[bool] = True
    #: The method-specific :class:`PruneConfig` fields (``PruneConfig.method_options``) that :meth:`prune` reads;
    #: the config rejects a non-default value of any other.
    options: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def check_config(cls, config: PruneConfig) -> None:
        """Reject settings the criterion cannot run (called by ``PruneConfig.__post_init__``)."""

    @abstractmethod
    def observer(self, module: nn.Linear) -> Observer | None:
        """The statistic of the module's inputs that :meth:`prune` consumes (``None`` for a data-free criterion)."""

    @abstractmethod
    def prune(self, weight: Tensor, stats: Observer | None, pattern: SparsityPattern, config: PruneConfig) -> Tensor:
        """Return the pruned weight (same shape, dtype and device)."""


PRUNE_METHODS: Registry[type[Pruner]] = Registry("pruning method", "economize.prune_methods")


@PRUNE_METHODS.register("magnitude")
class MagnitudePruner(Pruner):
    """Lowest ``|W|`` (Han et al., 2015); the baseline of the Wanda and SparseGPT papers."""

    needs_calibration = False

    def observer(self, module: nn.Linear) -> None:
        return None

    def prune(self, weight: Tensor, stats: Observer | None, pattern: SparsityPattern, config: PruneConfig) -> Tensor:
        return weight.masked_fill(magnitude_mask(weight, pattern), 0)


@PRUNE_METHODS.register("wanda")
class WandaPruner(Pruner):
    """Wanda (Sun et al., ICLR 2024): lowest ``|W| * ||X||`` per output row, no weight update."""

    def observer(self, module: nn.Linear) -> SquaredNormObserver:
        return SquaredNormObserver(module)

    def prune(self, weight: Tensor, stats: Observer | None, pattern: SparsityPattern, config: PruneConfig) -> Tensor:
        assert isinstance(stats, SquaredNormObserver)
        return weight.masked_fill(wanda_mask(weight, stats.sum, pattern), 0)


@PRUNE_METHODS.register("sparsegpt")
class SparseGPTPruner(Pruner):
    """SparseGPT (Frantar & Alistarh, ICML 2023): OBS saliency with error compensation from ``H = X X^T``."""

    options = frozenset({"blocksize", "percdamp"})

    @classmethod
    def check_config(cls, config: PruneConfig) -> None:
        m = config.pattern.m
        if config.pattern.structured and config.blocksize % (m or 1):
            raise ConfigError(f"blocksize {config.blocksize} must be a multiple of M={m}")

    def observer(self, module: nn.Linear) -> GramObserver:
        return GramObserver(module)

    def prune(self, weight: Tensor, stats: Observer | None, pattern: SparsityPattern, config: PruneConfig) -> Tensor:
        assert isinstance(stats, GramObserver)
        return sparsegpt_prune(weight, stats.sum, pattern, blocksize=config.blocksize, percdamp=config.percdamp)
