"""The registered ``teal`` technique (a thin wrapper over :mod:`.model`)."""

from __future__ import annotations

import math
from typing import Any, ClassVar, cast

import torch
from torch import nn

from ...core.calibration import calibration_source, token_batches
from ...core.metadata import Family, KernelChoice, Saving, TechniqueInfo
from ...core.technique import Capability, Technique, register_technique
from .backends import TEAL_BACKENDS, Installed
from .config import TealConfig
from .model import TealCalibration, calibrate_teal, install_teal, teal_modules
from .sparsify import block_sparsity

__all__ = ["TealTechnique"]


@register_technique
class TealTechnique(Technique):
    """Training-free magnitude activation sparsity (TEAL, see :class:`TealConfig`).

    The calibrated levels and thresholds are buffers (``sparsities``, ``thresholds``: one entry per target map in
    the order of :func:`.model.teal_modules`), so they are saved and loaded with the technique's state dict; a
    technique whose thresholds were loaded skips calibration at attach time. The base weights are never touched,
    and :meth:`detach` removes every hook.
    """

    name: ClassVar[str] = TealConfig.technique
    config_class: ClassVar[type[TealConfig]] = TealConfig
    info = TechniqueInfo(
        family=Family.ACTIVATION_SPARSITY,
        saves=Saving.COMPUTE | Saving.LATENCY,
        lossless=False,
        kernel=KernelChoice("backend", TEAL_BACKENDS),
    )
    capabilities: ClassVar[Capability] = Capability.ACTIVATIONS
    order: ClassVar[int] = 0
    config: TealConfig
    sparsities: torch.Tensor
    thresholds: torch.Tensor

    def __init__(self, config: TealConfig, adapter: Any) -> None:
        super().__init__(config, adapter)
        modules = teal_modules(adapter, config)
        self.keys = list(modules)
        self._sizes = {key: cast(torch.Tensor, module.weight).numel() for key, module in modules.items()}
        self.register_buffer("sparsities", torch.full((len(self.keys),), math.nan))
        self.register_buffer("thresholds", torch.full((len(self.keys),), math.nan))
        self.calibration: TealCalibration | None = None
        self._installed: Installed | None = None
        self._live: dict[str, torch.Tensor] = {}  # the installed threshold tensors, updated in place on a load
        self.register_load_state_dict_post_hook(_refresh_installed)

    @property
    def calibrated(self) -> bool:
        """Whether every map has a threshold (computed by :meth:`attach` or loaded from a checkpoint)."""
        return bool(torch.isfinite(self.thresholds).all())

    def attach(self, model: Any, adapter: Any, context: Any = None) -> None:
        self.config.validate(adapter)
        if not self.calibrated:
            batches = token_batches(calibration_source(context, self.name))
            self.calibration = calibrate_teal(adapter, self.config, batches)
            self.sparsities.copy_(torch.tensor([self.calibration.sparsities[k] for k in self.keys]))
            self.thresholds.copy_(torch.tensor([self.calibration.thresholds[k] for k in self.keys]))
        self.install()

    def install(self) -> None:
        """(Re)install the hooks with the current thresholds."""
        self.detach()
        self._installed, self._live = install_teal(self.adapter, self.threshold_map(), self.config)

    def detach(self) -> None:
        if self._installed is not None:
            self._installed.remove()
        self._installed, self._live = None, {}

    def threshold_map(self) -> dict[str, float]:
        """Threshold of every target map, keyed ``"<layer>.<module>"``."""
        return dict(zip(self.keys, self.thresholds.tolist()))

    def sparsity_map(self) -> dict[str, float]:
        """Calibrated sparsity level of every target map, keyed ``"<layer>.<module>"``."""
        return dict(zip(self.keys, self.sparsities.tolist()))

    @property
    def summary(self) -> dict[str, Any]:
        """Target and calibrated model-level sparsity (parameter-weighted), and the number of sparsified maps."""
        weighted = block_sparsity(self.sparsity_map(), self._sizes)
        return {
            "target": self.config.sparsity,
            "allocation": self.config.allocation,
            "model_sparsity": weighted,
            "maps": len(self.keys),
            "installed": self._installed is not None,
        }

    def refresh_thresholds(self) -> None:
        """Hand the current thresholds to the installed hooks (after a checkpoint load); a no-op without hooks.

        The installed threshold tensors are updated in place when the same maps stay sparsified; a map whose threshold
        turned zero or positive changes the hooked set, and the hooks are reinstalled.
        """
        if self._installed is None or not self.calibrated:
            return
        thresholds = self.threshold_map()
        if {key for key, value in thresholds.items() if value > 0} != set(self._live):
            self.install()
            return
        for key, live in self._live.items():
            live.fill_(thresholds[key])


def _refresh_installed(module: nn.Module, incompatible: Any) -> None:
    """Load-state-dict post hook: :meth:`TealTechnique.refresh_thresholds` (a checkpoint loaded after attach)."""
    assert isinstance(module, TealTechnique)
    module.refresh_thresholds()
