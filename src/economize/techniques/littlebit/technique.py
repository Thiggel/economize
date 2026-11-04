"""The registered ``littlebit`` technique (a thin wrapper over :func:`.model.littlebit_model`)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from torch import Tensor

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import Capability, register_technique
from ...registry import Registry
from .._weights.technique import CalibratedWeightTransform
from .config import LittleBitConfig
from .methods import LITTLEBIT_METHODS
from .model import littlebit_model

__all__ = ["LittleBitTechnique"]


@register_technique
class LittleBitTechnique(CalibratedWeightTransform):
    """LittleBit binary low-rank factorization of linear maps (see :class:`LittleBitConfig`).

    Runs after pruning and low rank and before quantization (``order`` 25); the packed maps are not
    ``nn.Linear``, so a following ``quant`` or ``lowrank`` leaves them alone.
    """

    name: ClassVar[str] = LittleBitConfig.technique
    config_class: ClassVar[type[LittleBitConfig]] = LittleBitConfig
    info = TechniqueInfo(family=Family.WEIGHT_COMPRESSION, saves=Saving.WEIGHT_MEMORY, lossless=False)
    capabilities: ClassVar[Capability] = Capability.WEIGHTS
    order: ClassVar[int] = 25
    methods: ClassVar[Registry[Any]] = LITTLEBIT_METHODS
    config: LittleBitConfig

    def needs_calibration(self) -> bool:
        return not self.config.factors and super().needs_calibration()

    def transform(self, adapter: Any, batches: Sequence[Tensor] | None) -> dict[str, Any]:
        return littlebit_model(adapter, self.config, batches)
