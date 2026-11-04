"""SmoothQuant: migrate activation outliers into the weights, then quantize weights and activations (W8A8).

Xiao, Lin, Seznec, Wu, Demouth & Han, "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large
Language Models", ICML 2023 (https://github.com/mit-han-lab/smoothquant). Port of ``smooth_ln_fcs``:

1. Calibration records the per-channel maximum ``max_t |x_tj|`` of the input of every norm-fed group of linear maps
   (``input_layernorm -> q/k/v``, ``post_attention_layernorm -> gate/up`` for Llama-style layers), on the original
   model's activations as in the reference.
2. Each channel gets the smoothing factor ``s_j = max|x_j|^alpha / max_i |W_ij|^(1 - alpha)`` (the weight maximum taken
   over all maps of the group, both factors clamped at 1e-5), ``1 / s`` is folded into the norm and ``s`` into the
   input columns of the maps, which leaves the float model unchanged.
3. The smoothed weights are rounded to nearest (int8 per channel for W8A8), and the inputs of the quantized maps are
   quantized per token at run time (``QuantConfig.activations``, see :mod:`.activations`).

As in the reference only norm-fed groups are smoothed: the inputs of ``o_proj`` and ``down_proj`` are quantized
unsmoothed. The groups are those AWQ uses (:data:`~.awq.AWQ_SCALE_GROUPS`, else derived per layer by
:func:`~.awq.derive_scale_groups`, which is how Gemma 4 gets them), restricted to groups whose previous op is not a
linear map.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, cast

import torch
from torch import Tensor, nn

from ...errors import ConfigError
from ...validation import check_unit_interval
from .._weights.layerwise import observe_step
from .awq import ScaleGroup, apply_scale, derive_scale_groups, registered_scale_groups
from .grid import QuantGrid, QuantizedWeight
from .methods import QUANT_METHODS, QuantMethod, RTNMethod

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...core.calibration import CalibrationStep, Propagation
    from .config import QuantConfig

__all__ = ["AbsMaxObserver", "SmoothQuantMethod", "smooth_groups", "smooth_layer", "smoothquant_scales"]


class AbsMaxObserver:
    """Per input feature ``max_t |x_t|`` over every observed token (SmoothQuant's ``act_scales``).

    Args:
        module: The linear map whose input is observed.
    """

    def __init__(self, module: nn.Module) -> None:
        weight = cast(Tensor, module.weight)
        self.max = torch.zeros(weight.shape[1], dtype=torch.float32, device=weight.device)
        self.tokens = 0

    def update(self, x: Tensor) -> None:
        """Fold one input batch ``[..., in_features]`` into the running maximum."""
        flat = x.reshape(-1, x.shape[-1])
        self.max = torch.maximum(self.max, flat.abs().amax(0).to(device=self.max.device, dtype=torch.float32))
        self.tokens += flat.shape[0]


def smoothquant_scales(act_max: Tensor, weights: Sequence[Tensor], alpha: float) -> Tensor:
    """The smoothing factors of one group: ``max|x|^alpha / max|W|^(1 - alpha)`` per input channel.

    Args:
        act_max: Per-channel maximum of the group's input, ``[in_features]``.
        weights: Weights ``[out_i, in_features]`` of the group's linear maps.
        alpha: Migration strength in ``[0, 1]`` (0: no smoothing of the weights' side, 1: all of it).

    Returns:
        float32 factors ``[in_features]``; the input is divided by them and the weight columns multiplied.
    """
    weight_max = torch.cat([w.abs().amax(dim=0, keepdim=True).float() for w in weights]).amax(dim=0).clamp(min=1e-5)
    act = act_max.float().to(weight_max.device)
    return (act.pow(alpha) / weight_max.pow(1 - alpha)).clamp(min=1e-5)


def smooth_groups(
    layer: nn.Module, groups: Sequence[ScaleGroup], targets: Mapping[str, nn.Linear]
) -> list[ScaleGroup]:
    """The groups SmoothQuant smooths: previous op not a linear map, every linear map among ``targets``.

    Args:
        layer: The decoder layer.
        groups: Its candidate scale groups.
        targets: The linear maps being quantized, by name inside the layer.

    Returns:
        The kept groups, in order.
    """
    kept = []
    for group in groups:
        if isinstance(layer.get_submodule(group.prev), nn.Linear):
            continue
        if all(name in targets for name in group.linears):
            kept.append(group)
    return kept


@QUANT_METHODS.register("smoothquant")
class SmoothQuantMethod(QuantMethod):
    """SmoothQuant (Xiao et al., ICML 2023): per-channel smoothing of the norm-fed linear maps, then round to
    nearest; the inputs are quantized per token by ``config.activations``. Calibrated on the original model."""

    propagate: ClassVar[Propagation] = "original"
    options = frozenset({"smooth_alpha"})
    folds_input_scales = True

    @classmethod
    def check_config(cls, config: QuantConfig) -> None:
        check_unit_interval("smooth_alpha", config.smooth_alpha, closed_low=True, closed_high=True)
        if config.activations is None:
            raise ConfigError(
                "smoothquant moves activation outliers into the weights for activation quantization: set "
                "activations='int8' (W8A8) or another registered activation scheme"
            )

    def quantize_layer(
        self,
        modules: Mapping[str, nn.Linear],
        step: CalibrationStep | None,
        grid: QuantGrid,
        config: QuantConfig,
        adapter: ArchAdapter,
    ) -> dict[str, QuantizedWeight]:
        assert step is not None
        smooth_layer(step, modules, config.smooth_alpha, adapter)
        return RTNMethod().quantize_layer(modules, None, grid, config, adapter)


@torch.no_grad()
def smooth_layer(
    step: CalibrationStep, modules: Mapping[str, nn.Linear], alpha: float, adapter: ArchAdapter
) -> dict[str, Tensor]:
    """Smooth the norm-fed groups of ``step.layer`` in place (the float model's outputs stay the same).

    Args:
        step: The layer's calibration step (its inputs from the original model).
        modules: The linear maps being quantized, by name inside the layer.
        alpha: Migration strength in ``[0, 1]``.
        adapter: The model's architecture adapter (for registered scale groups).

    Returns:
        The smoothing factors by the name of each smoothed group's previous op.

    Raises:
        ConfigError: The step saw no calibration tokens.
    """
    groups = registered_scale_groups(adapter)
    if groups is None:
        groups = derive_scale_groups(step)
    layer = step.layer
    kept = smooth_groups(layer, groups, modules)
    observed = {group.linears[0]: modules[group.linears[0]] for group in kept}
    stats = observe_step(step, observed, AbsMaxObserver) if observed else {}
    applied: dict[str, Tensor] = {}
    for group in kept:
        stat = stats[group.linears[0]]
        if stat.tokens == 0:
            raise ConfigError("SmoothQuant observed no calibration inputs at this layer: pass calibration data")
        fcs = [modules[name] for name in group.linears]
        scales = smoothquant_scales(stat.max, [fc.weight.data for fc in fcs], alpha)
        apply_scale(layer.get_submodule(group.prev), fcs, scales)
        applied[group.prev] = scales
    return applied
