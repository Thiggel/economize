"""Technique parameters on a model split by tensor parallelism (transformers' ``tp_plan``).

A split model holds its sharded weights as ``DTensor`` parameters, and transformers replaces the ``forward`` of every
sharded module by a closure that redistributes inputs and reduces outputs. Technique parameters fall into two kinds:

* Parameters of new modules (predictors, mixers, readout adapters, draft heads) are replicated: every process holds
  all of them, and :func:`replicate_from_first_rank` makes them equal to those of the first process, so that every
  process computes the same drafts.
* Parameters tied to a sharded weight (trainable layer copies, LoRA factors) are sharded like that weight, as
  ``DTensor`` parameters with the weight's placements. :func:`copy_tensor_parallel_forwards` gives a layer copy the
  sharded forwards of its original, and :class:`~economize.modules.LayerLoRA` shards its factors itself.

Checkpoints hold whole tensors in either case: :func:`full_tensor` gathers a shard for saving and :func:`shard_like`
cuts a whole tensor into the local shard of a parameter for loading, so a checkpoint saved on one GPU loads on a
split model and back.
"""

from __future__ import annotations

import functools
from typing import Any

import torch
from torch import nn

from ..errors import ConfigError

__all__ = [
    "check_adaptable_split",
    "clone_sharded",
    "copy_tensor_parallel_forwards",
    "full_tensor",
    "is_sharded",
    "model_tensor_parallel_group",
    "replicate_from_first_rank",
    "shard_like",
    "shard_state_like",
    "sharded_dim",
    "tensor_parallel_group",
]


def _dtensor_class() -> type | None:
    if not torch.distributed.is_available():
        return None
    from torch.distributed.tensor import DTensor

    return DTensor


def is_sharded(tensor: object) -> bool:
    """Whether ``tensor`` is a ``DTensor`` (a parameter of a split model)."""
    cls = _dtensor_class()
    return cls is not None and isinstance(tensor, cls)


def full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """The whole tensor of a ``DTensor`` (gathered from all processes, a collective call), else ``tensor``."""
    return tensor.full_tensor() if is_sharded(tensor) else tensor  # type: ignore[attr-defined]


def clone_sharded(tensor: torch.Tensor) -> torch.Tensor:
    """A copy of the ``DTensor`` ``tensor`` with a clone of its local shard (no communication).

    ``copy.deepcopy`` of a loaded ``DTensor`` can fail: transformers cuts the local shard of a process other than
    the first as a view into a storage that cannot be copied as a whole.
    """
    from torch.distributed.tensor import DTensor

    return DTensor.from_local(tensor.to_local().detach().clone(), tensor.device_mesh,  # type: ignore[attr-defined]
                              tensor.placements, run_check=False, shape=tensor.shape,  # type: ignore[attr-defined]
                              stride=tensor.stride())


def shard_like(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """``value`` (a whole tensor) as a ``DTensor`` with the mesh and placements of the ``DTensor`` ``like``.

    Every process cuts its own shard out of its copy of ``value`` (no communication), so ``value`` must be the same
    on every process, as a tensor read from a checkpoint is.
    """
    from torch.distributed.tensor import distribute_tensor

    return distribute_tensor(value.to(device=like.device), like.device_mesh, like.placements,  # type: ignore[attr-defined]
                             src_data_rank=None)


def shard_state_like(state: dict[str, torch.Tensor], module: nn.Module) -> dict[str, torch.Tensor]:
    """``state`` (whole tensors, e.g. from a checkpoint) with every entry whose parameter or buffer in ``module`` is a
    ``DTensor`` cut into that tensor's local shard (see :func:`shard_like`), ready for ``module.load_state_dict``."""
    own = module.state_dict()
    return {key: shard_like(value, own[key]) if key in own and is_sharded(own[key]) and not is_sharded(value)
            else value for key, value in state.items()}


def model_tensor_parallel_group(model: nn.Module) -> Any:
    """The tensor-parallel process group of a split model (that of its first ``DTensor`` parameter), ``None`` for a
    model without ``DTensor`` parameters."""
    for param in model.parameters():
        if is_sharded(param):
            return tensor_parallel_group(param)
    return None


def tensor_parallel_group(tensor: torch.Tensor) -> Any:
    """The process group a ``DTensor`` is sharded over (the ``"tp"`` dimension of a multi-dimensional mesh)."""
    mesh = tensor.device_mesh  # type: ignore[attr-defined]
    return mesh.get_group() if mesh.ndim == 1 else mesh.get_group("tp")


def sharded_dim(tensor: torch.Tensor) -> int | None:
    """The dimension (``0 .. ndim-1``) a ``DTensor`` is sharded along over its tensor-parallel group, or ``None`` for
    a replicated or plain tensor."""
    if not is_sharded(tensor):
        return None
    mesh = tensor.device_mesh  # type: ignore[attr-defined]
    index = 0 if mesh.ndim == 1 else list(mesh.mesh_dim_names or ()).index("tp")
    placement = tensor.placements[index]  # type: ignore[attr-defined]
    if not placement.is_shard():
        return None
    return int(placement.dim) % tensor.ndim


def check_adaptable_split(model: nn.Module, tensor_parallel_size: int, adaptation: str) -> None:
    """Refuse ``adaptation="lora"`` or ``"full"`` on a split model whose weights are no ``DTensor`` parameters.

    Older transformers releases split a model with forward hooks over plain local tensors; LoRA factors and layer
    copies follow the ``DTensor`` placements, so only frozen paths run on such a split.

    Raises:
        ConfigError: The model is split, holds no ``DTensor`` parameter and ``adaptation`` is not ``"frozen"``.
    """
    if adaptation == "frozen" or tensor_parallel_size <= 1 or model_tensor_parallel_group(model) is not None:
        return
    raise ConfigError(
        f"adaptation={adaptation!r} on a tensor-parallel split model needs DTensor parameters (transformers' DTensor "
        "tensor parallelism); this split holds plain tensors. Use adaptation='frozen' or a newer transformers"
    )


def replicate_from_first_rank(module: nn.Module, group: Any) -> None:
    """Overwrite the plain (unsharded) parameters and buffers of ``module`` with those of the first process of
    ``group`` (a collective call). ``DTensor`` parameters are left alone: they are sharded consistently already."""
    first = torch.distributed.get_global_rank(group, 0)
    with torch.no_grad():
        for tensor in [*module.parameters(), *module.buffers()]:
            if not is_sharded(tensor):
                torch.distributed.broadcast(tensor.data, src=first, group=group)


def _tensor_parallel_forward(module: nn.Module) -> tuple[Any, Any, dict[str, Any]] | None:
    """The transformers parallel style, mesh and further arguments of ``module``'s sharded forward, ``None`` if the
    module keeps its class's forward or its forward is bound to the module itself.

    ``accelerate``'s device hooks (``device_map``) install ``functools.partial(new_forward, module)``; a deep copy
    rebinds such a partial to the copied module, so it needs no new forward.
    """
    forward = module.__dict__.get("forward")
    if forward is None or (isinstance(forward, functools.partial) and forward.args[:1] == (module,)):
        return None
    from transformers.distributed.tensor_parallel import TensorParallelLayer

    closure = getattr(forward, "__closure__", None) or ()
    cells = dict(zip(getattr(getattr(forward, "__code__", None), "co_freevars", ()), closure))
    values = {name: cell.cell_contents for name, cell in cells.items()}
    style, mesh = values.get("self"), values.get("mesh")
    if not isinstance(style, TensorParallelLayer) or mesh is None:
        raise ConfigError(
            f"{type(module).__name__} has a replaced forward that is no transformers tensor-parallel forward, so a "
            "copy of it cannot be given its own"
        )
    extra = {name: value for name, value in values.items() if name == "is_expert_parallel"}
    return style, mesh, extra


def copy_tensor_parallel_forwards(original: nn.Module, copy: nn.Module) -> None:
    """Give every module of ``copy`` (a deep copy of ``original``) the sharded forward of its original.

    transformers installs a split module's forward as a closure over the module, which a deep copy shares with the
    original: the copy would run the original's weights. Each such closure of the copy is replaced by a new one that
    the same parallel style installs over the copied module.

    Raises:
        ConfigError: A module has a replaced forward that is neither a transformers tensor-parallel forward nor bound
            to the module itself.
    """
    for source, target in zip(original.modules(), copy.modules(), strict=True):
        found = _tensor_parallel_forward(source)
        if found is None:
            continue
        style, mesh, extra = found
        del target.__dict__["forward"]
        style.install_forward(target, mesh, **extra)
