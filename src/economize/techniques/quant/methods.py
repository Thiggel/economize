"""Registered weight-quantization methods. A new method is one :class:`QuantMethod` subclass registered in
:data:`QUANT_METHODS`::

    from economize.techniques.quant import QUANT_METHODS, QuantMethod

    @QUANT_METHODS.register("my_method")
    class MyMethod(QuantMethod):
        propagate = "original"
        options = frozenset({"percdamp"})  # the method-specific QuantConfig fields it reads

        def quantize_layer(self, modules, step, grid, config, adapter):
            ...  # return {name: QuantizedWeight} and write the dequantized weights into the modules

A method checks its own settings in :meth:`QuantMethod.check_config` (when the config is built) and
:meth:`QuantMethod.validate` (against the model), so :class:`~.config.QuantConfig` never names a method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import nn

from ...core.calibration import CalibrationStep, GramObserver, shared_input_groups
from ...errors import ConfigError
from ...registry import Registry
from .._weights.layerwise import observe_step
from .awq import ScaleGroup, awq_scale_layer, registered_scale_groups
from .gptq import gptq_quantize
from .grid import QuantGrid, QuantizedWeight, bind_shards, quantize_rtn

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...core.calibration import Propagation
    from .config import QuantConfig

__all__ = ["QUANT_METHODS", "AWQMethod", "GPTQMethod", "QuantMethod", "RTNMethod", "awq_scale_groups"]


class QuantMethod(ABC):
    """One way of choosing the codes of the linear maps of a decoder layer."""

    #: Whether :meth:`quantize_layer` needs calibration data (``step`` is ``None`` otherwise).
    needs_calibration: ClassVar[bool] = True
    #: Which activations feed the next layer during calibration (see ``LayerwiseCalibration``).
    propagate: ClassVar[Propagation] = "transformed"
    #: The method-specific fields of :class:`~.config.QuantConfig` (its ``method_options``) this method reads; the
    #: others must keep their defaults.
    options: ClassVar[frozenset[str]] = frozenset()
    #: Whether the method folds per-input-channel scales into the linear maps (AWQ, SmoothQuant). Such scales do not
    #: commute with an online rotation of the map's input, so ``rotate`` with online sites refuses these methods.
    folds_input_scales: ClassVar[bool] = False

    @classmethod
    def check_config(cls, config: QuantConfig) -> None:
        """Raise :class:`ConfigError` for settings the method cannot honour (runs when the config is built)."""

    @classmethod
    def float_zeros(cls, config: QuantConfig) -> bool:
        """Whether the method chooses real-valued zero points with ``config`` (only backends whose
        :meth:`~.backends.QuantBackend.stores_float_zeros` is true then accept it)."""
        return False

    @classmethod
    def validate(cls, config: QuantConfig, adapter: ArchAdapter) -> None:
        """Raise if the method cannot quantize the model of ``adapter`` (runs when the technique is applied)."""

    @abstractmethod
    def quantize_layer(
        self,
        modules: Mapping[str, nn.Linear],
        step: CalibrationStep | None,
        grid: QuantGrid,
        config: QuantConfig,
        adapter: ArchAdapter,
    ) -> dict[str, QuantizedWeight]:
        """Quantize ``modules`` (the targets of one decoder layer, keyed by their name inside the layer).

        Writes the dequantized weights into the modules (fake quantization) and returns the codes, which a
        kernel backend converts afterwards.
        """


QUANT_METHODS: Registry[type[QuantMethod]] = Registry("quantization method", "economize.quant_methods")


@torch.no_grad()
def write_fake(module: nn.Linear, qweight: QuantizedWeight) -> None:
    """Replace ``module.weight`` by the dequantized weight, in the module's dtype."""
    module.weight.data.copy_(qweight.dequantize(module.weight.dtype))


@QUANT_METHODS.register("rtn")
class RTNMethod(QuantMethod):
    """Round to nearest on the group-wise min-max grid (the baseline of GPTQ and AWQ). No calibration."""

    needs_calibration = False

    def quantize_layer(
        self,
        modules: Mapping[str, nn.Linear],
        step: CalibrationStep | None,
        grid: QuantGrid,
        config: QuantConfig,
        adapter: ArchAdapter,
    ) -> dict[str, QuantizedWeight]:
        out: dict[str, QuantizedWeight] = {}
        grids = bind_shards(modules, grid)
        for name, module in modules.items():
            out[name] = quantize_rtn(module.weight.data, grids[name], config.group_size)
            write_fake(module, out[name])
        return out


@QUANT_METHODS.register("gptq")
class GPTQMethod(QuantMethod):
    """GPTQ (Frantar et al., ICLR 2023), layer by layer on the quantized model's activations.

    With ``true_sequential`` (the reference default) the linear maps that read the same tensor are quantized
    together, in call order (``q/k/v``, ``o``, ``gate/up``, ``down`` for Llama-style layers), and the Hessians of
    later maps are collected after the earlier ones are quantized.
    """

    options = frozenset({"act_order", "static_groups", "true_sequential", "blocksize", "percdamp", "keep_zeros"})

    @classmethod
    def check_config(cls, config: QuantConfig) -> None:
        from .backends import QUANT_BACKENDS

        dynamic = config.act_order and not config.static_groups and config.group_size is not None
        if dynamic and not QUANT_BACKENDS.get(config.backend).supports_g_idx:
            raise ConfigError(
                f"act_order with static_groups=False reorders the groups (g_idx), which backend "
                f"{config.backend!r} cannot store: use static_groups=True or backend='fake'"
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
        groups = shared_input_groups(step, modules) if config.true_sequential else [list(modules)]
        grids = bind_shards(modules, grid)  # on the weights before any GPTQ update, as llm-compressor does
        out: dict[str, QuantizedWeight] = {}
        for names in groups:
            group = {name: modules[name] for name in names}
            stats = observe_step(step, group, GramObserver)
            for name, module in group.items():
                out[name] = gptq_quantize(
                    module.weight.data,
                    stats[name].sum,
                    grids[name],
                    config.group_size,
                    act_order=config.act_order,
                    blocksize=config.blocksize,
                    percdamp=config.percdamp,
                    static_groups=config.static_groups,
                    keep_zeros=config.keep_zeros,
                )
                write_fake(module, out[name])
        return out


def awq_scale_groups(adapter: ArchAdapter) -> tuple[ScaleGroup, ...] | None:
    """The AWQ scale groups registered for the adapter's ``model_type`` or ``layer_layout``
    (:func:`.awq.scale_groups_for`), or ``None``: then every layer derives its own
    (:func:`.awq.derive_scale_groups`)."""
    return registered_scale_groups(adapter)


@QUANT_METHODS.register("awq")
class AWQMethod(QuantMethod):
    """AWQ (Lin et al., MLSys 2024): activation-aware scales and clipping, then round to nearest.

    Calibrated on the original model's activations, as in the reference.
    """

    propagate = "original"
    options = frozenset({"awq_grid", "awq_clip"})
    folds_input_scales = True

    def quantize_layer(
        self,
        modules: Mapping[str, nn.Linear],
        step: CalibrationStep | None,
        grid: QuantGrid,
        config: QuantConfig,
        adapter: ArchAdapter,
    ) -> dict[str, QuantizedWeight]:
        assert step is not None
        awq_scale_layer(
            step,
            awq_scale_groups(adapter),
            modules,
            grid,
            config.group_size,
            n_grid=config.awq_grid,
            clip=config.awq_clip,
            offload=config.offload,
        )
        return RTNMethod().quantize_layer(modules, None, grid, config, adapter)
