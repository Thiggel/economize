"""The arithmetic of factored (low-rank) linear maps inside vLLM's linear layers, free of vLLM for CPU tests.

economize's ``lowrank`` replaces an HF map ``W`` by ``up(down(x))`` with ``down: in -> r`` and ``up: r -> out``.
vLLM fuses maps that read one input (``q_proj``, ``k_proj`` and ``v_proj`` into ``qkv_proj``), and ``lowrank`` keeps a
map dense when its factors would not be smaller, so a fused layer holds an ordered list of :class:`Unit`: one per
output shard, each dense or factored with its own rank. TEAL thresholds the input of every map it hooks, which on a
factored map means the input of ``down`` (threshold ``t``) and the input of ``up`` (threshold ``t_z``).

Storage of a layer, local to its tensor-parallel rank:

* ``weight`` ``[sum of the dense units' outputs, in]``: the dense units, in output order.
* ``lowrank_down`` ``[sum of the ranks, in]``: the ``down`` factors of the factored units, in output order.
* ``lowrank_up`` flat ``[sum of rank * output]``: each factored unit's ``up`` factor transposed (``[r, out]``,
  row-major), which is the ``W^T`` layout of TEAL's sparse GEMV.

:func:`factored_linear` computes the layer's output from them, unit by unit, and matches the HF runtime's
``up(down(x))`` with TEAL's masks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ...exporters import UnsupportedBackendError
from .teal import bounds, map_names, teal_linear

__all__ = ["Unit", "factored_linear", "layer_units", "up_factor"]


@dataclass(frozen=True)
class Unit:
    """One output shard of a vLLM linear layer as economize's linear method computes it.

    Attributes:
        name: The unit's module name in vLLM's naming (the HF map it stands for).
        rank: The rank of its factors, 0 for a dense unit.
        threshold: TEAL's threshold on the unit's input (0 keeps every nonzero input).
        factor_threshold: TEAL's threshold on the input of ``up`` (factored units only).
    """

    name: str
    rank: int = 0
    threshold: float = 0.0
    factor_threshold: float = 0.0

    @property
    def factored(self) -> bool:
        """Whether the unit is computed through its factors."""
        return self.rank > 0


def layer_units(prefix: str, packed: Mapping[str, Sequence[str]], ranks: Mapping[str, int],
                thresholds: Mapping[str, float], factor_thresholds: Mapping[str, float]) -> list[Unit] | None:
    """The units of vLLM's linear layer ``prefix``, or ``None`` when no table names it or one of its shards.

    Args:
        prefix: The vLLM module name of the layer, e.g. ``model.layers.0.self_attn.qkv_proj``.
        packed: The model's ``packed_modules_mapping`` (fused name -> the HF names of its shards, in output order).
        ranks: ``lowrank``'s ranks by module name, in vLLM's naming.
        thresholds: TEAL's input thresholds by module name.
        factor_thresholds: TEAL's thresholds on the input of ``up``, by the factored map's module name.
    """
    names = map_names(prefix, packed, ranks, thresholds, factor_thresholds)
    if names is None:
        return None
    return [Unit(name, max(int(ranks.get(name, 0)), 0), max(float(thresholds.get(name, 0.0)), 0.0),
                 max(float(factor_thresholds.get(name, 0.0)), 0.0) if ranks.get(name, 0) else 0.0)
            for name in names]


def up_factor(up: Tensor, offset: int, rank: int, size: int) -> Tensor:
    """A factored unit's ``up`` factor ``[size, rank]`` as a view of the flat storage (column-major)."""
    return up[offset:offset + rank * size].view(rank, size).t()


def factored_linear(x: Tensor, bias: Tensor | None, weight: Tensor | None, dense_thresholds: Tensor, down: Tensor,
                    down_thresholds: Tensor, up: Tensor, up_thresholds: Tensor, sizes: Sequence[int],
                    ranks: Sequence[int], rows: bool | Tensor) -> Tensor:
    """The output of a layer with factored units (and possibly dense ones), TEAL's masks on the selected rows.

    Args:
        x: Inputs ``[..., in]``.
        bias: The layer's bias ``[sum of sizes]`` or ``None``.
        weight: The dense units' weight (see the module docstring), ``None`` without dense units.
        dense_thresholds: float32, one per dense unit.
        down: The ``down`` factors ``[sum of ranks, in]``.
        down_thresholds: float32, one per factored unit (TEAL's threshold on its input).
        up: The flat ``up`` storage.
        up_thresholds: float32, one per factored unit (TEAL's threshold on the input of ``up``).
        sizes: Output size of every unit, in output order.
        ranks: Rank of every unit (0 for a dense unit).
        rows: ``True``, ``False`` or a row mask, as in :func:`~economize.backends.vllm.linear.teal.teal_linear`.

    Returns:
        ``[..., sum of sizes]`` in the dtype of ``x``.
    """
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    dense_sizes = [size for size, rank in zip(sizes, ranks, strict=True) if rank == 0]
    factor_ranks = [rank for rank in ranks if rank > 0]
    dense = None
    if dense_sizes:
        if weight is None:
            raise UnsupportedBackendError("a layer with dense units needs their weight")
        dense = teal_linear(flat, weight, None, dense_thresholds, dense_sizes, rows)
    latent = teal_linear(flat, down, None, down_thresholds, factor_ranks, rows)
    dense_bounds = iter(bounds(dense_sizes))
    latent_bounds = iter(bounds(factor_ranks))
    parts: list[Tensor] = []
    offset = 0
    factor = 0
    for size, rank in zip(sizes, ranks, strict=True):
        if rank == 0:
            start, end = next(dense_bounds)
            parts.append(dense[:, start:end])  # type: ignore[index]
            continue
        start, end = next(latent_bounds)
        parts.append(teal_linear(latent[:, start:end], up_factor(up, offset, rank, size), None,
                                 up_thresholds[factor:factor + 1], [size], rows))
        offset += rank * size
        factor += 1
    out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
    if bias is not None:
        out = out + bias
    return out.reshape(*shape[:-1], out.shape[-1])
