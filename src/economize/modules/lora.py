"""Low-rank adaptation of a model's linear maps that is active only inside a context.

Techniques such as jump drafting adapt some decoder layers for their own computation path while the exact path
must keep using the original weights. :class:`LayerLoRA` therefore never changes a weight: it adds its low-rank
update through forward hooks (installed by :meth:`LayerLoRA.install`), and the hooks contribute only inside
:meth:`LayerLoRA.active`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import partial
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.hooks import RemovableHandle

from ..errors import ConfigError
from .sharding import is_sharded, sharded_dim, tensor_parallel_group

__all__ = ["LayerLoRA", "LoRADelta"]


class LoRADelta(nn.Module):
    """The update ``scale * B A x`` of one linear map ``in_features -> out_features``.

    ``A`` gets the default ``nn.Linear`` (Kaiming uniform) init and ``B`` starts at zero, so the update starts at zero.

    On a linear map split by tensor parallelism (see :meth:`shard_like`) the factor that shares the split dimension
    is sharded like the map's weight: ``B`` for a map split along its outputs (column-parallel), whose update is the
    local slice of the outputs, and ``A`` for a map split along its inputs (row-parallel), whose rank-``r``
    intermediate is summed over the processes before ``B`` maps it to the whole output.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, scale: float) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = scale
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self._reduce_group: object | None = None  # the processes of a row-parallel split, which sum ``A x``

    def shard_like(self, weight: torch.Tensor) -> None:
        """Shard the factors like ``weight``, the ``[out, in]`` weight of the adapted map, if it is a ``DTensor``
        split along one dimension (a collective call: the first process's factors are scattered, so every process
        starts from the same update). Factors that are sharded already stay as they are."""
        dim = sharded_dim(weight)
        if dim is None or is_sharded(self.lora_A) or is_sharded(self.lora_B):
            return
        from torch.distributed.tensor import distribute_tensor

        name = "lora_B" if dim == 0 else "lora_A"
        factor = getattr(self, name)
        sharded = distribute_tensor(factor.detach(), weight.device_mesh, weight.placements,  # type: ignore[attr-defined]
                                    src_data_rank=0)
        setattr(self, name, nn.Parameter(sharded, requires_grad=factor.requires_grad))
        if dim == 1:
            self._reduce_group = tensor_parallel_group(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The update for input ``x``, in the dtype of the LoRA parameters (on a split map: this process's part
        of the update for its part of the input, see the class docstring)."""
        a, b = (_local(factor) for factor in (self.lora_A, self.lora_B))
        low = F.linear(x.to(a.dtype), a)
        if self._reduce_group is not None:
            low = _all_reduce(low, self._reduce_group)
        return F.linear(low, b) * self.scale


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if is_sharded(tensor) else tensor  # type: ignore[attr-defined]


def _all_reduce(tensor: torch.Tensor, group: object) -> torch.Tensor:
    """Sum ``tensor`` over ``group``; differentiable when gradients are recorded."""
    if torch.is_grad_enabled() and tensor.requires_grad:
        from torch.distributed.nn.functional import all_reduce

        return all_reduce(tensor, group=group)
    torch.distributed.all_reduce(tensor, group=group)  # type: ignore[call-overload]
    return tensor


class LayerLoRA(nn.Module):
    """LoRA updates for the linear maps of several decoder layers.

    Args:
        targets: ``{layer index: {name: linear map}}``, e.g. built from
            :meth:`economize.arch.ArchAdapter.lora_targets`. A map is any module with ``in_features`` and
            ``out_features`` whose output the update is added to (``nn.Linear``, a low-rank factorization).
        rank: Rank of every update.
        alpha: Scaling numerator; the update is multiplied by ``alpha / rank``.

    Parameters are stored as ``layers.<layer>.<name>.lora_A`` / ``lora_B``. Construction leaves the adapted maps
    untouched; :meth:`install` adds the hooks and :meth:`remove` takes them off again.
    """

    def __init__(self, targets: Mapping[int, Mapping[str, nn.Module]], rank: int, alpha: float) -> None:
        super().__init__()
        self.rank, self.alpha = rank, alpha
        self.layers = nn.ModuleDict()
        self._targets: list[tuple[nn.Module, LoRADelta]] = []  # plain list: the maps belong to the base model
        self._handles: list[RemovableHandle] = []
        self._depth = 0
        for layer, linears in targets.items():
            deltas = nn.ModuleDict()
            for name, linear in linears.items():
                deltas[name] = LoRADelta(cast(int, linear.in_features), cast(int, linear.out_features), rank,
                                         alpha / rank)
            self.layers[str(layer)] = deltas
        self._bind(targets)

    @property
    def installed(self) -> bool:
        """Whether the hooks are on the adapted maps."""
        return bool(self._handles)

    def install(self, targets: Mapping[int, Mapping[str, nn.Module]] | None = None) -> None:
        """Hook the updates into the adapted linear maps (idempotent).

        Args:
            targets: The maps to adapt, in the layout given at construction. Pass them when the base model's modules
                were replaced after construction (e.g. by a low-rank factorization); ``None`` keeps the maps given
                at construction.

        Raises:
            ConfigError: ``targets`` lacks a map that has an update, or a map's shape differs from its update.
        """
        if targets is not None:
            self.remove()
            self._bind(targets)
        if not self._handles:
            self._handles = [linear.register_forward_hook(partial(self._add_update, delta))
                             for linear, delta in self._targets]

    @contextmanager
    def active(self) -> Iterator[None]:
        """Apply the updates to every call of the adapted linear maps inside the block."""
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1

    def remove(self) -> None:
        """Remove the hooks from the adapted linear maps (the parameters stay)."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _bind(self, targets: Mapping[int, Mapping[str, nn.Module]]) -> None:
        bound: list[tuple[nn.Module, LoRADelta]] = []
        for layer, deltas in self.layers.items():
            linears = targets.get(int(layer), {})
            for name, delta in cast(nn.ModuleDict, deltas).items():
                delta = cast(LoRADelta, delta)
                linear = linears.get(name)
                if linear is None:
                    raise ConfigError(f"LoRA update of layer {layer} has no target {name!r}")
                shape = (linear.in_features, linear.out_features)
                if shape != (delta.lora_A.shape[1], delta.lora_B.shape[0]):
                    raise ConfigError(f"LoRA target {name!r} of layer {layer} is {shape[0]} -> {shape[1]}, the "
                                     f"update {delta.lora_A.shape[1]} -> {delta.lora_B.shape[0]}")
                weight = getattr(linear, "weight", None)
                if isinstance(weight, torch.Tensor):
                    delta.shard_like(weight)
                bound.append((linear, delta))
        self._targets = bound

    def _add_update(
        self, delta: LoRADelta, module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> torch.Tensor | None:
        if not self._depth:
            return None
        return output + delta(args[0]).to(output.dtype)

