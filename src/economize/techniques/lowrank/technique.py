"""The registered ``lowrank`` technique (a thin wrapper over :func:`.model.lowrank_model`)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from torch import Tensor

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import Capability, DraftCopies, Technique, register_technique
from ...registry import Registry
from .._weights.technique import CalibratedWeightTransform
from ..prune.config import PruneConfig
from .config import LowRankConfig
from .factorize import FACTORIZERS, Factorizer
from .model import lowrank_model

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["LowRankTechnique"]


@register_technique
class LowRankTechnique(CalibratedWeightTransform):
    """SVD-LLM / truncated-SVD factorization of linear maps (see :class:`LowRankConfig`).

    Runs after pruning and before quantization (``order`` 20); the factors are plain ``nn.Linear`` modules, so a
    following ``quant`` quantizes them.
    """

    name: ClassVar[str] = LowRankConfig.technique
    config_class: ClassVar[type[LowRankConfig]] = LowRankConfig
    info = TechniqueInfo(family=Family.WEIGHT_COMPRESSION, saves=Saving.WEIGHT_MEMORY | Saving.COMPUTE, lossless=False)
    capabilities: ClassVar[Capability] = Capability.WEIGHTS
    order: ClassVar[int] = 20
    draft_copies: ClassVar[DraftCopies | None] = DraftCopies.OWNED
    methods: ClassVar[Registry[type[Factorizer]]] = FACTORIZERS
    config: LowRankConfig

    def plain_save_issue(self) -> str:
        return "factorized maps are module pairs whose weights the model's architecture has no keys for"

    def check_compatible(self, others: Sequence[Technique]) -> None:
        """Also warn when weight pruning runs before this factorization: the SVD factors of a pruned weight are
        dense, so the pruned zeros no longer save memory or compute in the factorized layers."""
        super().check_compatible(others)
        if any(isinstance(other.config, PruneConfig) for other in others):
            warnings.warn(
                "lowrank factorizes the weights that prune has just sparsified; the factors are dense, so the "
                "sparsity is lost in every layer both target (restrict one of them with targets/exclude to keep it)",
                UserWarning,
                stacklevel=2,
            )

    def transform(self, adapter: ArchAdapter, batches: Sequence[Tensor] | None) -> dict[str, object]:
        return lowrank_model(adapter, self.config, batches)
