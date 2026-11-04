"""Tensor parallelism through the plans transformers ships with its models (``tp_plan="auto"``).

transformers shards a model over the processes of a ``torchrun`` launch when it is loaded with ``tp_plan="auto"``:
every architecture with a ``base_model_tp_plan`` in its config splits the query, key, value, gate and up projections
by columns and the output and down projections by rows, so every process holds ``1 / world_size`` of each layer and
the model's own ``forward`` returns full logits on every rank. These helpers check that a model can be split that way
before anything is loaded and do the loading::

    # torchrun --nproc-per-node 2 serve.py
    from economize.distributed.tp import load_tensor_parallel

    model = load_tensor_parallel("Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16)
    logits = model(input_ids.to(model.device)).logits  # identical on every rank

A split model works wherever the model's own forward runs: ``generate``, the vLLM and TRL routes of
:mod:`economize.integrations` (vLLM has its own ``tensor_parallel_size``) and lm-eval. economize's layer runner sizes
its caches from the key/value states it stores, so they hold each process's local heads, and economize's own decoding
(jump and early-exit drafters with any adaptation, chained verification, CUDA graphs) runs on a split model; see
:mod:`economize.modules.sharding` for how technique weights are replicated or sharded. :mod:`economize.distributed.ssd`
rejects split models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from ..arch.config import config_value, layer_overrides, text_config
from ..errors import ConfigError
from ..validation import check_positive

__all__ = [
    "TensorParallelCheck",
    "check_tensor_parallel",
    "init_distributed",
    "is_tensor_parallel",
    "load_tensor_parallel",
    "tensor_parallel_plan",
]

#: Config fields that a plan splits evenly over the processes, when present.
_SPLIT_FIELDS = ("num_attention_heads", "num_key_value_heads", "intermediate_size", "moe_intermediate_size")


def tensor_parallel_plan(config_or_model: Any) -> dict[str, str]:
    """The tensor-parallel plan of a model or config (module pattern to ``"colwise"``, ``"rowwise"``, ...); empty
    when the architecture has none."""
    if isinstance(config_or_model, nn.Module):
        plan = getattr(config_or_model, "tp_plan", None)
        if plan:
            return dict(plan)
        config_or_model = config_or_model.config
    return dict(getattr(text_config(config_or_model), "base_model_tp_plan", None) or {})


@dataclass(frozen=True)
class TensorParallelCheck:
    """Whether a config can be split over ``world_size`` processes.

    Attributes:
        world_size: Processes the model would be split over.
        plan: The architecture's tensor-parallel plan.
        problems: Everything that prevents the split (empty when it works).
    """

    world_size: int
    plan: dict[str, str] = field(default_factory=dict)
    problems: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.problems

    def raise_for_problems(self) -> None:
        """Raise :class:`~economize.errors.ConfigError` listing the problems, if any."""
        if self.problems:
            raise ConfigError(f"cannot split the model over {self.world_size} processes: " + "; ".join(self.problems))


def check_tensor_parallel(config: Any, world_size: int) -> TensorParallelCheck:
    """Check that ``config`` has a tensor-parallel plan and that its heads and MLP widths divide by ``world_size``."""
    check_positive("world_size", world_size)
    text = text_config(config)
    plan = tensor_parallel_plan(text)
    problems = []
    if not plan:
        problems.append(f"{type(text).__name__} has no base_model_tp_plan")
    overrides = layer_overrides(text).values()
    for name in _SPLIT_FIELDS:
        values = {config_value(text, name)} | {layer.get(name) for layer in overrides}
        for value in sorted(v for v in values if isinstance(v, int) and v % world_size):
            problems.append(f"{name}={value} is not divisible by {world_size}")
    return TensorParallelCheck(world_size, plan, tuple(problems))


def init_distributed() -> tuple[int, int, int]:
    """Join the process group of a ``torchrun`` launch (NCCL on GPUs, Gloo otherwise) and select this process's GPU.

    Returns:
        ``(rank, local_rank, world_size)``; ``(0, 0, 1)`` outside a launch.
    """
    if "WORLD_SIZE" not in os.environ:
        return 0, 0, 1
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, local_rank, world_size


def load_tensor_parallel(name_or_path: str, **kwargs: Any) -> nn.Module:
    """Load a causal LM split over the processes of the current ``torchrun`` launch.

    Args:
        name_or_path: Hub id or local folder.
        **kwargs: Passed to ``AutoModelForCausalLM.from_pretrained`` (``dtype``, ``revision``, ...).

    Raises:
        ConfigError: Not inside a launch, or the architecture cannot be split over its processes.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    if "device_map" in kwargs:
        raise ConfigError("tensor parallelism places the shards itself; drop device_map")
    _, _, world_size = init_distributed()
    if world_size == 1:
        raise ConfigError("load_tensor_parallel needs a torchrun launch with several processes")
    config_keys = ("revision", "trust_remote_code", "token", "cache_dir", "local_files_only", "subfolder")
    config_kwargs = {key: kwargs[key] for key in config_keys if key in kwargs}
    check_tensor_parallel(AutoConfig.from_pretrained(name_or_path, **config_kwargs), world_size).raise_for_problems()
    model = AutoModelForCausalLM.from_pretrained(name_or_path, tp_plan="auto", **kwargs)
    return model.eval()


def is_tensor_parallel(model: nn.Module) -> bool:
    """Whether ``model`` was loaded split over several processes."""
    size = getattr(model, "_tp_size", None)
    return isinstance(size, int) and size > 1
