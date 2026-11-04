"""How the factors of one decoder layer's maps are found: the registered LittleBit methods.

A method gets the layer, its selected maps with their ranks and the layer's calibration step, and returns the
factors of every map; :func:`.model.littlebit_model` packs them. A new method is one :class:`LittleBitMethod`
subclass registered in :data:`LITTLEBIT_METHODS`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from torch import nn

from ...registry import Registry
from .._weights.transaction import replace_module
from .factorize import LittleBitFactors, littlebit_init
from .qat import LatentLittleBitLinear, train_block

if TYPE_CHECKING:
    from ...core.calibration import CalibrationStep
    from .config import LittleBitConfig

__all__ = ["LITTLEBIT_METHODS", "BlockwiseMethod", "DualSVIDMethod", "LittleBitMethod"]


class LittleBitMethod(ABC):
    """Finds the LittleBit factors of the selected maps of one decoder layer."""

    #: Whether :meth:`factorize` reads the layer's calibration step.
    needs_calibration: ClassVar[bool] = False

    @abstractmethod
    def factorize(
        self,
        layer: nn.Module,
        modules: Mapping[str, nn.Linear],
        ranks: Mapping[str, int],
        step: CalibrationStep | None,
        config: LittleBitConfig,
    ) -> tuple[dict[str, LittleBitFactors], dict[str, Any]]:
        """The factors of every map in ``modules`` (keyed like it) and per-layer statistics for the summary.

        The layer must be left as it was found: the caller swaps the packed maps in.
        """


LITTLEBIT_METHODS: Registry[type[LittleBitMethod]] = Registry("LittleBit method", "economize.littlebit_methods")


@LITTLEBIT_METHODS.register("dual_svid")
class DualSVIDMethod(LittleBitMethod):
    """The paper's data-free Dual-SVID initialization (with residual compensation when ``config.residual``)."""

    def factorize(
        self,
        layer: nn.Module,
        modules: Mapping[str, nn.Linear],
        ranks: Mapping[str, int],
        step: CalibrationStep | None,
        config: LittleBitConfig,
    ) -> tuple[dict[str, LittleBitFactors], dict[str, Any]]:
        factors = {
            name: littlebit_init(module.weight.data, ranks[name], residual=config.residual)
            for name, module in modules.items()
        }
        return factors, {}


@LITTLEBIT_METHODS.register("blockwise")
class BlockwiseMethod(LittleBitMethod):
    """Dual-SVID, then QAT of all the layer's latent factors jointly on the MSE to the original layer's outputs.

    The short-budget form of the paper's training (see :mod:`.qat`). The targets are the original layer's outputs
    on the step's inputs (:attr:`~economize.core.calibration.CalibrationStep.first_outputs`), which the calibration
    engine records before the step is handed over.
    """

    needs_calibration: ClassVar[bool] = True

    def factorize(
        self,
        layer: nn.Module,
        modules: Mapping[str, nn.Linear],
        ranks: Mapping[str, int],
        step: CalibrationStep | None,
        config: LittleBitConfig,
    ) -> tuple[dict[str, LittleBitFactors], dict[str, Any]]:
        assert step is not None, "blockwise LittleBit needs a calibration step"
        targets = step.first_outputs if step.first_outputs is not None else step.forward()
        latents = {
            name: LatentLittleBitLinear(
                littlebit_init(module.weight.data, ranks[name], residual=config.residual),
                module.bias,
                config.smooth_k,
            )
            for name, module in modules.items()
        }
        frozen = {p: p.requires_grad for p in layer.parameters()}
        for p in frozen:
            p.requires_grad_(False)
        for name, latent in latents.items():
            replace_module(layer, name, latent)
        try:
            stats = train_block(
                layer,
                latents,
                step.inputs,
                step.calls,
                targets,
                step.device,
                epochs=config.epochs,
                lr=config.lr,
                accumulate=config.accumulate,
                seed=step.index,
            )
        finally:
            for name, module in modules.items():
                replace_module(layer, name, module)
            for p, flag in frozen.items():
                p.requires_grad_(flag)
        return {name: latent.factors() for name, latent in latents.items()}, stats
