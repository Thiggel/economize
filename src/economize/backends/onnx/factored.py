"""Factored linear maps in the exported graph: ``lowrank`` keeps its two factors as two ``MatMul`` nodes.

:func:`factor_linears` replaces each map that :attr:`ExportBuilder.factored
<economize.backends.runtime_export.ExportBuilder.factored>` names in the plain model by a :class:`FactoredLinear`
whose ``down`` and ``up`` are plain ``nn.Linear`` modules. The traced graph then holds the initializers
``<name>.down.weight`` and ``<name>.up.weight`` (and ``<name>.up.bias``), ``rank * (in + out)`` numbers in place
of ``in * out``, and a quantized factor still becomes a ``MatMulNBits`` node
(:func:`~economize.backends.onnx.graph.matmul_nbits` finds it by that name).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ...errors import ConfigError
from ..runtime_export import ExportBuilder

__all__ = ["FactoredLinear", "factor_linears"]


class FactoredLinear(nn.Module):
    """``y = up(down(x))``: a linear map stored as its two factors.

    Parameters
    ----------
    down : nn.Linear
        ``in -> rank`` (no bias).
    up : nn.Linear
        ``rank -> out`` (the map's bias).
    """

    def __init__(self, down: nn.Linear, up: nn.Linear) -> None:
        super().__init__()
        self.down, self.up = down, up

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.down(x))


def _factor_weight(builder: ExportBuilder, name: str, module: nn.Module) -> Tensor:
    """The dense weight of the factor ``name`` (dequantized when the quantizer recorded its codes)."""
    from ...integrations.compressed_tensors import quantized_modules

    qweight = quantized_modules(builder.model).get(name)
    if qweight is not None:
        return qweight.dequantize(torch.float32)
    weight = getattr(module, "weight", None)
    if not isinstance(weight, Tensor):
        raise ConfigError(f"the factor {name!r} holds neither a weight nor recorded quantized codes")
    from ..vllm.serving.weights import _plain

    return _plain(weight)


def _linear(weight: Tensor, bias: Tensor | None, dtype: torch.dtype) -> nn.Linear:
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, dtype=dtype)
    with torch.no_grad():
        linear.weight.copy_(weight)
        if bias is not None:
            linear.bias.copy_(bias.detach())
    return linear.requires_grad_(False)


def factor_linears(plain: nn.Module, builder: ExportBuilder, dtype: torch.dtype) -> None:
    """Replace the maps of ``builder.factored`` in ``plain`` by their two factors (in place).

    Parameters
    ----------
    plain : nn.Module
        The plain architecture with the dense state loaded (:func:`~economize.backends.onnx.engine.plain_model`).
    builder : ExportBuilder
        The export plan; its ``factored`` modules (names relative to the base) hold ``down`` and ``up``.
    dtype : torch.dtype
        The type of the factors' weights.

    Raises
    ------
    ConfigError
        A factored module has no ``down`` / ``up`` factor, or a factor has no weight.
    """
    from ..vllm.serving.weights import _bias_of

    for name, module in builder.factored.items():
        down, up = getattr(module, "down", None), getattr(module, "up", None)
        if not isinstance(down, nn.Module) or not isinstance(up, nn.Module):
            raise ConfigError(f"the factored map {name!r} has no down / up factors")
        factored = FactoredLinear(
            _linear(_factor_weight(builder, name + ".down", down), None, dtype),
            _linear(_factor_weight(builder, name + ".up", up), _bias_of(module), dtype),
        )
        owner, _, leaf = name.rpartition(".")
        setattr(plain.get_submodule(owner), leaf, factored)
