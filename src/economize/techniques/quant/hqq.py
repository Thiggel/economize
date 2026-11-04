"""HQQ, half-quadratic quantization (Badri and Shaji, 2023; https://github.com/mobiusml/hqq).

Calibration-free: the min-max grid of every group is refined by a few proximal steps that fit the zero points to a
sparsity-promoting ``l_p`` loss (``p < 1``) of the quantization error, with the scales held fixed. The steps follow
``hqq.core.optimize.optimize_weights_proximal_legacy`` (the default solver of hqq 0.2):

.. code-block:: text

    s = (2^b - 1) / (max - min)          # inverse scale, per row and group; min/max do not include 0
    z = -min * s
    repeat (at most iters times):
        q   = clamp(round(w * s + z), 0, 2^b - 1)
        w_r = (q - z) / s
        e   = shrink_p(w - w_r, beta)    # sign(x) * relu(|x| - |x|^(p-1) / beta)
        z   = mean(q - (w - e) * s)      # over the group
        stop when mean|w - w_r| over the whole tensor no longer decreases (z keeps its last update)
    codes = clamp(round(w * s + z), 0, 2^b - 1),  w ~= (codes - z) / s

The zero points are real numbers, so the codes run only on kernels that store float zeros (torchao's tinygemm int4,
or the fake backend); ``hqq_integer_zero=True`` rounds the final zero points (and re-rounds the codes) for the
integer-zero kernels of vLLM (Marlin, Machete, compressed-tensors), a deviation from the reference. The reference
computes in fp16 on CUDA and in fp32 on the CPU; this implementation always computes in fp32, so it matches the
reference's CPU path exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from ...errors import ConfigError
from .grid import IntGrid, QuantGrid, QuantizedWeight
from .methods import QUANT_METHODS, QuantMethod, write_fake

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...core.calibration import CalibrationStep
    from .config import QuantConfig

__all__ = ["HQQMethod", "hqq_quantize", "shrink_lp"]


def shrink_lp(x: Tensor, beta: float, p: float) -> Tensor:
    """The proximal operator of the ``l_p`` norm used by HQQ (``hqq.core.optimize.shrink_lp_op``).

    Args:
        x: The quantization error.
        beta: The penalty weight of the half-quadratic split (larger shrinks less).
        p: The norm exponent (``p = 1`` is soft thresholding).

    Returns:
        ``sign(x) * relu(|x| - |x|^(p-1) / beta)`` (``relu(|x| - 1 / beta)`` for ``p = 1``).
    """
    if p == 1:
        return torch.sign(x) * torch.relu(x.abs() - 1.0 / beta)
    return torch.sign(x) * torch.relu(x.abs() - (1.0 / beta) * torch.pow(x.abs(), p - 1))


@torch.no_grad()
def hqq_quantize(
    weight: Tensor,
    grid: IntGrid,
    group_size: int | None,
    *,
    iters: int = 20,
    lp_norm: float = 0.7,
    beta: float = 10.0,
    round_zero: bool = False,
    integer_zero: bool = False,
) -> QuantizedWeight:
    """HQQ quantization of ``weight [out, in]`` with one scale and zero point per row and group of input columns.

    Args:
        weight: The weight to quantize.
        grid: An asymmetric integer grid (its ``bits`` are used).
        group_size: Input columns per group (``None``: one group per output channel).
        iters: Maximal number of proximal steps (the reference stops earlier once the mean error rises).
        lp_norm: The exponent ``p`` of the error norm.
        beta: The half-quadratic penalty weight.
        round_zero: Round the initial zero points (the reference's ``round_zero``).
        integer_zero: Round the final zero points and re-round the codes, for integer-zero kernels.

    Returns:
        The codes, scales ``[out, groups]`` and zero points (float, integer-valued with ``integer_zero``).

    Raises:
        ConfigError: If ``group_size`` does not divide the input size.
    """
    out_features, in_features = weight.shape
    size = group_size or in_features
    if in_features % size:
        raise ConfigError(f"group size {size} does not divide {in_features} input features")
    w = weight.detach().float().reshape(-1, size)
    maxq = float(grid.maxq)
    low, high = w.min(1, keepdim=True).values, w.max(1, keepdim=True).values
    denom = high - low
    inv_scale = torch.where(denom.abs() <= 1e-4, torch.ones_like(denom), maxq / denom).clamp(max=2e4)
    zero = -low * inv_scale
    if round_zero:
        zero = torch.round(zero)
    best = torch.tensor(float("inf"), device=w.device)
    for _ in range(iters):
        q = torch.round(w * inv_scale + zero).clamp(0, maxq)
        w_r = (q - zero) / inv_scale
        error = shrink_lp(w - w_r, beta, lp_norm)
        zero = torch.mean(q - (w - error) * inv_scale, dim=1, keepdim=True)
        current = (w - w_r).abs().mean()
        if current < best:
            best = current
        else:
            break
    if integer_zero:
        zero = torch.round(zero).clamp(0, maxq)
    codes = torch.round(w * inv_scale + zero).clamp(0, maxq).to(torch.uint8)
    groups = in_features // size
    return QuantizedWeight(
        grid,
        codes.reshape(out_features, in_features),
        (1.0 / inv_scale).reshape(out_features, groups),
        zero.reshape(out_features, groups),
        size,
    )


@QUANT_METHODS.register("hqq")
class HQQMethod(QuantMethod):
    """HQQ (see :func:`hqq_quantize`): calibration-free, asymmetric integer grids with optimized float zero points.

    Reads the ``hqq_*`` fields of :class:`~.config.QuantConfig`. Needs ``symmetric=False`` and an integer format.
    """

    needs_calibration = False
    options = frozenset({"hqq_iters", "hqq_lp_norm", "hqq_beta", "hqq_round_zero", "hqq_integer_zero"})

    @classmethod
    def float_zeros(cls, config: QuantConfig) -> bool:
        return not config.hqq_integer_zero

    @classmethod
    def check_config(cls, config: QuantConfig) -> None:
        if not isinstance(config.grid, IntGrid):
            raise ConfigError(f"method 'hqq' quantizes to integer formats, got format {config.format!r}")
        if config.symmetric:
            raise ConfigError("method 'hqq' optimizes the zero points: use symmetric=False")
        if not 0 < config.hqq_lp_norm <= 1:
            raise ConfigError(f"hqq_lp_norm must be in (0, 1], got {config.hqq_lp_norm}")
        if config.hqq_iters < 0 or config.hqq_beta <= 0:
            raise ConfigError("hqq_iters must be non-negative and hqq_beta positive")

    def quantize_layer(
        self,
        modules: Mapping[str, nn.Linear],
        step: CalibrationStep | None,
        grid: QuantGrid,
        config: QuantConfig,
        adapter: ArchAdapter,
    ) -> dict[str, QuantizedWeight]:
        assert isinstance(grid, IntGrid)
        out: dict[str, QuantizedWeight] = {}
        for name, module in modules.items():
            out[name] = hqq_quantize(
                module.weight.data,
                grid,
                config.group_size,
                iters=config.hqq_iters,
                lp_norm=config.hqq_lp_norm,
                beta=config.hqq_beta,
                round_zero=config.hqq_round_zero,
                integer_zero=config.hqq_integer_zero,
            )
            write_fake(module, out[name])
        return out
