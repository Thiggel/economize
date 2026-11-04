"""The registered ``quant`` and ``draft_quant`` techniques (thin wrappers over :mod:`.model` and :mod:`.draft`)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from torch import Tensor

from ...core.config import TechniqueConfig
from ...core.metadata import Family, KernelChoice, Saving, TechniqueInfo
from ...core.technique import Capability, Technique, register_technique
from ...errors import ConfigError, IncompatibleTechniquesError
from ...registry import Registry
from .._weights.technique import CalibratedWeightTransform
from .backends import QUANT_BACKENDS
from .config import DraftQuantConfig, QuantConfig
from .draft import DraftSnapshot, draft_modules, quantize_draft, restore_draft
from .methods import QUANT_METHODS
from .model import quantize_model

__all__ = ["DraftQuantTechnique", "QuantTechnique"]


@register_technique
class QuantTechnique(CalibratedWeightTransform):
    """Weight-only RTN / GPTQ / AWQ quantization to int2-8 or fp8 (see :class:`QuantConfig`).

    Runs after pruning and low-rank factorization (``order`` 30), so pruned zeros and low-rank factors are
    quantized as they are.
    """

    name: ClassVar[str] = QuantConfig.technique
    config_class: ClassVar[type[QuantConfig]] = QuantConfig
    info = TechniqueInfo(
        family=Family.WEIGHT_COMPRESSION,
        saves=Saving.WEIGHT_MEMORY | Saving.LATENCY,
        lossless=False,
        kernel=KernelChoice("backend", QUANT_BACKENDS),
    )
    capabilities: ClassVar[Capability] = Capability.WEIGHTS
    order: ClassVar[int] = 30
    methods: ClassVar[Registry[Any]] = QUANT_METHODS
    config: QuantConfig

    def transform(self, adapter: Any, batches: Sequence[Tensor] | None) -> dict[str, Any]:
        return quantize_model(adapter, self.config, batches)


@register_technique
class DraftQuantTechnique(Technique):
    """Round-to-nearest quantization of the drafter's own modules (see :class:`DraftQuantConfig`).

    Lossless, because the full-precision model verifies every drafted token. It needs exactly one drafter among the
    attached techniques and must attach after it.
    """

    name: ClassVar[str] = DraftQuantConfig.technique
    config_class: ClassVar[type[DraftQuantConfig]] = DraftQuantConfig
    info = TechniqueInfo(
        family=Family.DRAFT_COMPRESSION,
        saves=Saving.LATENCY,
        lossless=True,
        requires=(Family.DRAFTING,),
        kernel=KernelChoice("backend", QUANT_BACKENDS),
    )
    capabilities: ClassVar[Capability] = Capability.NONE
    order: ClassVar[int] = 100
    config: DraftQuantConfig

    @classmethod
    def conflict(
        cls, config: TechniqueConfig | None, other: type[Technique], other_config: TechniqueConfig | None
    ) -> str | None:
        """Also refuses a drafter with no modules of its own (:meth:`~Technique.trainable_for` empty): a
        retrieval drafter, one that runs the model's own layers, or a separate draft LM, which is kept outside the
        module tree."""
        reason = super().conflict(config, other, other_config)
        if reason is None and other.info.family is Family.DRAFTING and not other.trainable_for(other_config):
            reason = f"{cls.name} cannot be combined with {other.name}: {other.name} has no drafter modules to quantize"
        return reason

    def attach(self, model: Any, adapter: Any, context: Any = None) -> None:
        drafter = _the_drafter(getattr(context, "techniques", ()) if context is not None else ())
        modules = draft_modules(drafter, self.config, adapter.model.parameters())
        if not modules:
            raise ConfigError(f"draft_quant found no quantizable linear maps in drafter {drafter.name!r}")
        self._snapshot: DraftSnapshot | None = quantize_draft(modules, self.config)
        self.summary = self._snapshot.summary

    def detach(self) -> None:
        snapshot = getattr(self, "_snapshot", None)
        if snapshot is not None:
            restore_draft(snapshot)
            self._snapshot = None


def _the_drafter(techniques: Sequence[Technique]) -> Technique:
    drafters = [t for t in techniques if Capability.DRAFTER in t.capabilities]
    if len(drafters) != 1:
        raise IncompatibleTechniquesError(
            f"draft_quant quantizes the modules of exactly one drafter technique, found {len(drafters)}"
        )
    return drafters[0]
