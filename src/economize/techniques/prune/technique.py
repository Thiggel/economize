"""The registered ``prune`` and ``depth_prune`` techniques (thin wrappers over :mod:`.model`)."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, ClassVar

from torch import Tensor

from ...core.config import TechniqueConfig
from ...core.metadata import Family, KernelChoice, Saving, Simulation, TechniqueInfo
from ...core.technique import Capability, Technique, register_technique
from ...registry import Registry
from .._weights.technique import CalibratedWeightTransform
from .config import DepthPruneConfig, PruneConfig
from .depth import remove_layers
from .methods import PRUNE_METHODS
from .model import depth_prune_model, prune_model
from .semi_kernels import SEMI_STRUCTURED_KERNELS

__all__ = ["DepthPruneTechnique", "PruneTechnique"]

_MASKED_DENSE = Simulation(Saving.WEIGHT_MEMORY | Saving.COMPUTE, "keeps the masked dense weights")


@register_technique
class PruneTechnique(CalibratedWeightTransform):
    """Wanda / SparseGPT / magnitude pruning, unstructured or N:M (see :class:`PruneConfig`)."""

    name: ClassVar[str] = PruneConfig.technique
    config_class: ClassVar[type[PruneConfig]] = PruneConfig
    info = TechniqueInfo(family=Family.WEIGHT_COMPRESSION, saves=Saving.WEIGHT_MEMORY | Saving.COMPUTE, lossless=False)
    capabilities: ClassVar[Capability] = Capability.WEIGHTS
    order: ClassVar[int] = 10
    methods: ClassVar[Registry[Any]] = PRUNE_METHODS
    config: PruneConfig

    def transform(self, adapter: Any, batches: Sequence[Tensor] | None) -> dict[str, Any]:
        return prune_model(adapter, self.config, batches)

    @classmethod
    def conflict(
        cls, config: TechniqueConfig | None, other: type[Technique], other_config: TechniqueConfig | None
    ) -> str | None:
        """Semi-structured storage leaves sparse tensors that a later weight transform cannot rewrite."""
        if (
            isinstance(config, PruneConfig)
            and config.semi_structured
            and Capability.WEIGHTS in other.capabilities
            and other.order > cls.order
        ):
            return (
                f"prune(semi_structured=True) stores sparse tensors that {other.name!r} cannot transform; "
                "use semi_structured=False when combining pruning with low rank or quantization"
            )
        return super().conflict(config, other, other_config)

    @classmethod
    def simulation_for(cls, config: TechniqueConfig | None = None) -> Simulation | None:
        """Unstructured and N:M pruning keep masked dense weights; 2:4 with ``semi_structured`` runs the
        :data:`~.semi_kernels.SEMI_STRUCTURED_KERNELS` entry the configuration names."""
        config = config if isinstance(config, PruneConfig) else PruneConfig()
        if not config.semi_structured:
            return _MASKED_DENSE
        kernel = KernelChoice("semi_structured_kernel", SEMI_STRUCTURED_KERNELS).simulation(config)
        return kernel.within(cls.info.saves) if kernel is not None else None


@register_technique
class DepthPruneTechnique(CalibratedWeightTransform):
    """ShortGPT / SLEB layer removal (see :class:`DepthPruneConfig`). Runs before every other weight transform."""

    name: ClassVar[str] = DepthPruneConfig.technique
    config_class: ClassVar[type[DepthPruneConfig]] = DepthPruneConfig
    info = TechniqueInfo(
        family=Family.WEIGHT_COMPRESSION,
        saves=Saving.WEIGHT_MEMORY | Saving.COMPUTE | Saving.LATENCY,
        lossless=False,
    )
    capabilities: ClassVar[Capability] = Capability.WEIGHTS
    order: ClassVar[int] = 5
    config: DepthPruneConfig

    def needs_calibration(self) -> bool:
        """The criterion reads calibration data unless the config names the layers to remove."""
        return not self.config.layers

    def transform(self, adapter: Any, batches: Sequence[Tensor] | None) -> dict[str, Any]:
        if self.config.layers:
            removed = sorted(self.config.layers)
            remove_layers(adapter, removed)
        else:
            assert batches is not None
            removed = depth_prune_model(adapter, self.config, batches)
            # record the choice, so that a saved checkpoint removes the same layers without calibration
            self.config = dataclasses.replace(self.config, layers=tuple(removed))
        return {"method": self.config.method, "removed_layers": removed, "num_layers": adapter.num_layers}
