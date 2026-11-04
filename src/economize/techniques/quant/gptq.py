"""GPTQ.

Frantar, Ashkboos, Hoefler & Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained
Transformers", ICLR 2023 (https://github.com/IST-DASLab/gptq). The columns of a weight matrix are quantized left to
right; the rounding error of every column is compensated on the columns not yet quantized through the rows of the
upper Cholesky factor of the damped inverse Hessian ``(X X^T + lambda I)^{-1}``. Updates are applied lazily in blocks
of 128 columns. With ``act_order`` the columns are processed in order of decreasing ``diag(H)``.

This is a port of ``GPTQ.fasterquant`` that also returns the integer codes and group parameters. Two details follow
the reference on purpose:

* Without ``act_order`` the parameters of a group are fitted when its first column is reached, on the global weight
  matrix (its columns inside the current block do not yet contain the in-block updates).
* With ``act_order`` and groups, ``static_groups=True`` (the default) fits the groups up front on contiguous
  columns (the reference's ``static_groups``), so the result stays a contiguous group-wise format that standard
  int4 kernels can run. ``static_groups=False`` is the reference's plain ``act_order``: a group is fitted when its
  first column in the permuted order is reached, so a group holds the columns of similar Hessian diagonal and the
  weight needs a column-to-group map (``g_idx``) at inference time.

With ``keep_zeros``, entries that are exactly zero on entry (a pruned weight) stay zero, as in SparseGPT's joint
prune-and-quantize mode (Frantar & Alistarh, ICML 2023): the target of such an entry is 0 instead of its compensated
value, and its error is spread over the later columns like any other rounding error. Every grid contains zero, so
the codes decode to exact zeros. ``keep_zeros=None`` turns it on when at least :data:`KEEP_ZEROS_FRACTION` of the
weight is exactly zero, which a pruned weight reaches and the occasional exact zero of a dense weight does not.
Without it the algorithm is the reference unchanged.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ...errors import ConfigError
from .._weights.hessian import damped_inverse_cholesky, fix_dead_features
from .._weights.obs import ColumnRule, ObsBlock, obs_sweep
from .grid import QParams, QuantGrid, QuantizedWeight

__all__ = ["KEEP_ZEROS_FRACTION", "gptq_quantize"]

#: Fraction of exact zeros above which ``keep_zeros=None`` treats a weight as pruned.
KEEP_ZEROS_FRACTION = 0.05


@torch.no_grad()
def gptq_quantize(
    weight: Tensor,
    hessian: Tensor,
    grid: QuantGrid,
    group_size: int | None = None,
    *,
    act_order: bool = False,
    blocksize: int = 128,
    percdamp: float = 0.01,
    static_groups: bool = True,
    keep_zeros: bool | None = None,
) -> QuantizedWeight:
    """Quantize ``weight [out, in]`` with GPTQ.

    Args:
        weight: The weight (any float dtype).
        hessian: ``[in, in]`` sum of ``x x^T`` over the calibration inputs of this linear map (any positive scale).
        grid: Number format.
        group_size: Input columns per quantization group (``None``: one group per row).
        act_order: Quantize columns in order of decreasing Hessian diagonal ("desc_act").
        blocksize: Columns per lazy-update block.
        percdamp: Diagonal damping relative to ``mean(diag(H))``.
        static_groups: With ``act_order`` and groups: fit the groups on contiguous columns before the permutation
            (contiguous groups); ``False`` fits them in the permuted order and returns a ``g_idx``.
        keep_zeros: Keep exact zeros at zero (``None``: when at least :data:`KEEP_ZEROS_FRACTION` of the entries
            are zero).

    Returns:
        The quantized weight; ``dequantize()`` gives the reference's ``Q``.
    """
    W = weight.detach().float().clone()
    if grid.requires_binding:
        grid = grid.bind([W])
    pruned = W == 0
    if keep_zeros is None:
        keep_zeros = bool(pruned.float().mean() >= KEEP_ZEROS_FRACTION)
    rows, columns = W.shape
    size = group_size or columns
    if columns % size:
        raise ConfigError(f"group size {size} does not divide {columns} input features")
    groups = _GroupParams(rows, columns // size, W.device)

    channel_params = grid.fit(W) if group_size is None else None  # fitted before dead columns are zeroed (reference)
    H, _dead = fix_dead_features(hessian, W)
    static: list[QParams] | None = None
    if group_size is not None and act_order and static_groups:
        static = [grid.fit(W[:, g * size : (g + 1) * size]) for g in range(columns // size)]
    for g, params in enumerate([channel_params] if channel_params is not None else static or []):
        groups.set(g, params)

    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        pruned = pruned[:, perm]
    Hinv = damped_inverse_cholesky(H, percdamp)

    rule = _GPTQRule(grid, W, pruned, keep_zeros, size, groups, perm, static, channel_params, group_size is not None)
    obs_sweep(W, Hinv, rule, blocksize)
    g_idx = None
    if group_size is not None and act_order and not static_groups:
        assert perm is not None
        g_idx = torch.empty(columns, dtype=torch.long, device=W.device)
        g_idx[perm] = torch.arange(columns, device=W.device) // size
    return QuantizedWeight(grid, rule.codes, groups.scale, groups.zero if groups.has_zero else None, size, g_idx)


class _GroupParams:
    """The ``[rows, groups]`` scales and zero points of a GPTQ weight, filled in as the groups are fitted."""

    def __init__(self, rows: int, ngroups: int, device: torch.device) -> None:
        self.scale = torch.empty(rows, ngroups, device=device)
        self.zero = torch.empty(rows, ngroups, device=device)
        self.has_zero = False

    def set(self, group: int, params: QParams) -> None:
        self.scale[:, group] = params.scale[:, 0]
        if params.zero is not None:
            self.zero[:, group] = params.zero[:, 0]
            self.has_zero = True


class _GPTQRule(ColumnRule):
    """Round a column to the grid with the parameters of its group and record its codes.

    The parameters come from the per-row fit (no groups), the static groups, or a fit on the global (permuted)
    weight when a group's first column is reached; the global weight's columns inside the current block do not yet
    hold the in-block updates, as in the reference.
    """

    def __init__(
        self,
        grid: QuantGrid,
        weight: Tensor,
        pruned: Tensor,
        keep_zeros: bool,
        size: int,
        groups: _GroupParams,
        perm: Tensor | None,
        static: list[QParams] | None,
        params: QParams | None,
        grouped: bool,
    ) -> None:
        self.grid = grid
        self.weight = weight
        self.pruned = pruned
        self.keep_zeros = keep_zeros
        self.size = size
        self.groups = groups
        self.perm = perm
        self.static = static
        self.params = params
        self.grouped = grouped
        self.codes = torch.empty(weight.shape, dtype=grid.code_dtype, device=weight.device)

    def _params(self, column: int, original: int) -> QParams:
        if self.grouped:
            if self.static is not None:
                self.params = self.static[original // self.size]
            elif column % self.size == 0:
                block = self.weight[:, column : column + self.size]
                pruned = self.pruned[:, column : column + self.size]
                self.params = self.grid.fit(block.masked_fill(pruned, 0) if self.keep_zeros else block)
                self.groups.set(column // self.size, self.params)
        assert self.params is not None
        return self.params

    def replace(self, block: ObsBlock, i: int) -> Tensor:
        column = block.start + i
        original = int(self.perm[column]) if self.perm is not None else column
        params = self._params(column, original)
        w = block.weight[:, i]
        target = w.masked_fill(self.pruned[:, column], 0) if self.keep_zeros else w
        code = self.grid.encode(target.unsqueeze(1), params)
        self.codes[:, original] = code[:, 0]
        return self.grid.decode(code, params).flatten()
