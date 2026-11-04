"""The registered ``rotate`` technique (a thin wrapper over :func:`.model.rotate_model`)."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, ClassVar

from torch import Tensor

from ...core.config import TechniqueConfig
from ...core.metadata import Exclusion, Family, Saving, TechniqueInfo
from ...core.technique import Capability, Technique, register_technique
from ...registry import Registry
from .._weights.technique import CalibratedWeightTransform
from .config import RotateConfig
from .methods import ROTATIONS
from .model import rotate_model

__all__ = ["RotateTechnique"]


@register_technique
class RotateTechnique(CalibratedWeightTransform):
    """QuaRot's Hadamard rotations (see :class:`RotateConfig`), an enabler of a following weight quantization.

    The rotations leave the model's function unchanged (exact in exact arithmetic) and save nothing on their own,
    so the technique requires a weight-compression technique in the same recipe. It runs after pruning and low-rank
    factorization (whose sparse or factorized maps a rotation would densify, so those combinations are refused) and
    before ``quant`` (order 30), so GPTQ and HQQ fit the rotated weights.
    """

    name: ClassVar[str] = RotateConfig.technique
    config_class: ClassVar[type[RotateConfig]] = RotateConfig
    info = TechniqueInfo(
        family=Family.WEIGHT_COMPRESSION,
        saves=Saving.NONE,
        lossless=True,
        requires=(Family.WEIGHT_COMPRESSION,),
        excludes=(
            Exclusion(
                Family.ACTIVATION_SPARSITY,
                "rotate densifies the inputs whose small entries activation sparsity drops, and its thresholds are "
                "calibrated on the unrotated inputs",
            ),
        ),
    )
    capabilities: ClassVar[Capability] = Capability.WEIGHTS
    order: ClassVar[int] = 29
    methods: ClassVar[Registry[Any]] = ROTATIONS
    config: RotateConfig

    @classmethod
    def conflict(
        cls, config: TechniqueConfig | None, other: type[Technique], other_config: TechniqueConfig | None
    ) -> str | None:
        """Refuse the combinations a rotation breaks, then apply the metadata rules.

        * a weight technique with per-module ``targets`` that runs earlier (pruning, low-rank, LittleBit): the
          rotation would densify its sparse or factorized maps;
        * a ``quant`` method that folds per-input-channel scales (AWQ, SmoothQuant) while ``online`` rotates map
          inputs: such scales do not commute with the online Hadamard transform;
        * KV quantization with calibrated static scales (``scale="tensor"``) while ``values`` rotates the values:
          the scales are calibrated on the unrotated values.

        Args:
            config: This technique's config, ``None`` if unknown.
            other: The other technique's class.
            other_config: Its config, ``None`` if unknown.

        Returns:
            The reason the two cannot be combined, or ``None``.
        """
        if (
            Capability.WEIGHTS in other.capabilities
            and other.order < cls.order
            and any(field.name == "targets" for field in dataclasses.fields(other.config_class))
        ):
            return (f"rotate cannot follow {other.name}: the rotation would densify the maps {other.name} sparsified "
                    "or factorized")
        if isinstance(config, RotateConfig) and other_config is not None:
            reason = _config_conflict(config, other, other_config)
            if reason is not None:
                return reason
        return super().conflict(config, other, other_config)

    def transform(self, adapter: Any, batches: Sequence[Tensor] | None) -> dict[str, Any]:
        return rotate_model(adapter, self.config)


def _config_conflict(config: RotateConfig, other: type[Technique], other_config: TechniqueConfig) -> str | None:
    """The configuration-dependent rules of :meth:`RotateTechnique.conflict`."""
    if config.online:
        from ..quant import QUANT_METHODS, QuantConfig

        if (
            isinstance(other_config, QuantConfig)
            and other_config.method in QUANT_METHODS
            and getattr(QUANT_METHODS.get(other_config.method), "folds_input_scales", False)
        ):
            return (f"quant method {other_config.method!r} folds per-input-channel scales, which do not commute "
                    "with rotate's online Hadamard transforms; use method 'rtn', 'gptq' or 'hqq', or "
                    "rotate(online=())")
    static_scales = getattr(other_config, "scale", None) == "tensor"
    if config.values and other.info.family == Family.KV_QUANTIZATION and static_scales:
        return (f"{other.name} calibrates static value scales before rotate rotates the values; use scale='token' "
                "or rotate(values=False)")
    return None
