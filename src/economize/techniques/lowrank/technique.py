"""The registered ``lowrank`` technique (a thin wrapper over :func:`.model.lowrank_model`)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from torch import Tensor

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import Capability, register_technique
from ...registry import Registry
from .._weights.technique import CalibratedWeightTransform
from .config import LowRankConfig
from .factorize import FACTORIZERS
from .model import lowrank_model

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
    methods: ClassVar[Registry[Any]] = FACTORIZERS
    config: LowRankConfig

    def transform(self, adapter: Any, batches: Sequence[Tensor] | None) -> dict[str, Any]:
        return lowrank_model(adapter, self.config, batches)
