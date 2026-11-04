"""CATS (Lee et al., "CATS: Contextually-Aware Thresholding for Sparsity in Large Language Models", COLM 2024,
arXiv:2404.08763; reference code github.com/ScalingIntelligence/CATS).

CATS sparsifies the activation of every gated MLP: with ``g = act(gate(x))`` the MLP computes
``down((g * 1[|g| > t]) * up(x))`` (Equation 2 of the paper), so the up rows and the down columns of the dropped
channels are never read. Each layer's threshold ``t`` is the ``sparsity`` quantile of ``|g|`` on generic text
(Section 3.2, "cutoff threshold"). The paper also fine-tunes the thresholded model briefly; the training-free
variant is implemented here. Training calls of :mod:`economize.train` run the thresholded forward only with
``prefill="all"`` (they are multi-token calls outside decoding), and the mask then has no gradient.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any, ClassVar

import torch
from torch import Tensor, nn

from ...core.calibration import (
    LayerwiseCalibration,
    MagnitudeSampleObserver,
    calibration_source,
    observing,
    resolve_layers,
    token_batches,
)
from ...core.metadata import Family, KernelChoice, Saving, TechniqueInfo
from ...core.technique import Capability, Technique, register_technique
from ..sparsity.sparsify import magnitude_quantile, track_layer_tokens
from .backends import MLP_SPARSITY_BACKENDS, MlpOps
from .config import CatsConfig
from .mlp import GatedMlp, MlpPatch, dense_first_half, gated_mlps, masked

__all__ = ["CatsTechnique", "GateMagnitudeObserver", "calibrate_cats", "cats_mlp"]


def cats_mlp(mlp: GatedMlp, x: Tensor, ops: MlpOps, mode: str, threshold: Tensor) -> Tensor:
    """The CATS forward of one gated MLP (``mode``: ``"all"`` or ``"second_half"``)."""
    g = mlp.act(mlp.gate(x))
    keep = g.abs() > threshold
    if mode == "second_half":
        keep = dense_first_half(keep)
    h = masked(g, keep) * ops.rows(x, mlp.up, keep)
    return ops.cols(h, mlp.down)


class GateMagnitudeObserver:
    """Samples ``|act(gate(x))|`` of one gated MLP from its inputs ``x`` (for the CATS threshold)."""

    def __init__(self, mlp: GatedMlp, max_samples: int, seed: int) -> None:
        self.mlp = mlp
        self.samples = MagnitudeSampleObserver(max_samples=max_samples, seed=seed)

    def update(self, x: Tensor) -> None:
        self.samples.update(self.mlp.act(self.mlp.gate(x)))


def _gate_observer(_module: nn.Module, *, mlp: GatedMlp, max_samples: int, seed: int) -> GateMagnitudeObserver:
    return GateMagnitudeObserver(mlp, max_samples, seed)


@torch.no_grad()
def calibrate_cats(adapter: Any, config: CatsConfig, batches: list[Tensor]) -> dict[int, float]:
    """The threshold of every selected layer: the ``config.sparsity`` quantile of the dense model's ``|g|``.

    Every layer sees the dense model's inputs, so the thresholds do not depend on the order of the layers.
    """
    mlps = gated_mlps(adapter, config)
    thresholds: dict[int, float] = {}
    layers = resolve_layers(config.layers, adapter.num_layers)
    for step in LayerwiseCalibration(adapter, batches, propagate="original", layers=layers, offload=config.offload):
        mlp = mlps[step.index]
        factory = partial(_gate_observer, mlp=mlp, max_samples=config.max_samples, seed=step.index)
        with observing({"mlp": mlp.module}, factory) as observers:
            step.forward()
        magnitudes = observers["mlp"].samples.values().sort().values
        thresholds[step.index] = magnitude_quantile(magnitudes, config.sparsity)
    return thresholds


@register_technique
class CatsTechnique(Technique):
    """Training-free threshold sparsity of the gated-MLP activations (CATS, see :class:`CatsConfig`).

    The thresholds are a buffer (``thresholds``, one entry per selected layer in layer order), saved and loaded with
    the technique's state dict; a loaded technique skips calibration. The base weights are never changed by the
    dense backend, the Triton backend stores the down weights transposed while attached, and :meth:`detach`
    restores both.
    """

    name: ClassVar[str] = CatsConfig.technique
    config_class: ClassVar[type[CatsConfig]] = CatsConfig
    info = TechniqueInfo(
        family=Family.ACTIVATION_SPARSITY,
        saves=Saving.COMPUTE | Saving.LATENCY,
        lossless=False,
        kernel=KernelChoice("backend", MLP_SPARSITY_BACKENDS),
    )
    capabilities: ClassVar[Capability] = Capability.ACTIVATIONS
    order: ClassVar[int] = 1
    config: CatsConfig
    thresholds: torch.Tensor

    def __init__(self, config: CatsConfig, adapter: Any) -> None:
        super().__init__(config, adapter)
        self.layers = list(gated_mlps(adapter, config))
        self.register_buffer("thresholds", torch.full((len(self.layers),), math.nan))
        self._patches: list[Any] = []
        self._live: list[Tensor] = []  # the installed 0-dim thresholds, updated in place on a load
        self.register_load_state_dict_post_hook(_refresh_installed)

    @property
    def calibrated(self) -> bool:
        """Whether every layer has a threshold (computed by :meth:`attach` or loaded from a checkpoint)."""
        return bool(torch.isfinite(self.thresholds).all())

    def attach(self, model: nn.Module, adapter: Any, context: Any = None) -> None:
        self.config.validate(adapter)
        self.detach()  # calibration reads the dense MLPs
        if not self.calibrated:
            batches = token_batches(calibration_source(context, self.name))
            found = calibrate_cats(adapter, self.config, batches)
            self.thresholds.copy_(torch.tensor([found[i] for i in self.layers]))
        self.install()

    def install(self) -> None:
        """(Re)install the patched MLPs with the current thresholds."""
        self.detach()
        backend = MLP_SPARSITY_BACKENDS.get(self.config.backend)()
        policy = self.config.policy
        for index, value in zip(self.layers, self.thresholds.tolist()):
            mlp = GatedMlp(getattr(self.adapter.layer(index), self.config.mlp), self.config)
            device = next(mlp.module.parameters()).device
            live = torch.tensor(value, dtype=torch.float32, device=device)
            self._live.append(live)
            self._patches.append(MlpPatch(mlp, partial(_rule, threshold=live), backend.ops(mlp), policy))
        self._patches += track_layer_tokens(self.adapter.layers)

    def detach(self) -> None:
        for patch in self._patches:
            patch.remove()
        self._patches, self._live = [], []

    def threshold_map(self) -> dict[int, float]:
        """Threshold of every selected layer, keyed by the 1-indexed layer."""
        return dict(zip(self.layers, self.thresholds.tolist()))

    @property
    def summary(self) -> dict[str, Any]:
        """Target sparsity, number of sparsified MLPs and whether the patches are installed."""
        return {"target": self.config.sparsity, "layers": len(self.layers), "installed": bool(self._patches)}

    def refresh_thresholds(self) -> None:
        """Hand the current thresholds to the installed MLPs (after a checkpoint load); a no-op without patches."""
        if not self._live or not self.calibrated:
            return
        for live, value in zip(self._live, self.thresholds.tolist()):
            live.fill_(value)


def _rule(mlp: GatedMlp, x: Tensor, ops: MlpOps, mode: str, *, threshold: Tensor) -> Tensor:
    return cats_mlp(mlp, x, ops, mode, threshold)


def _refresh_installed(module: nn.Module, incompatible: Any) -> None:
    """Load-state-dict post hook: :meth:`CatsTechnique.refresh_thresholds` (a checkpoint loaded after attach)."""
    assert isinstance(module, CatsTechnique)
    module.refresh_thresholds()
