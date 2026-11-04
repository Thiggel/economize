"""Whole-model rotation (a pure function on an adapter; the technique delegates to it)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from ...core.calibration import resolve_layers, select_modules
from ...errors import ConfigError
from .._weights.transaction import WeightTransaction
from .config import RotateConfig
from .hadamard import hadamard_block, hadamard_transform
from .methods import ROTATIONS, RotationMethod

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["ROTATED_ATTR", "InputHadamard", "input_hadamard_of", "rotate_model"]

#: Attribute holding the online Hadamard block of a linear map whose input is rotated (``int``).
ROTATED_ATTR = "economize_input_hadamard"
_HOOK_ATTR = "_economize_input_hadamard_hook"


@dataclass(frozen=True)
class InputHadamard:
    """The forward pre-hook of an online rotation: ``x -> x H`` with ``H`` block-diagonal of size ``block``.

    A plain callable (no reference to its module), so a deep copy of the module copies a working hook.

    Attributes:
        block: The Hadamard block size.
        kernel: The kernel name (see :data:`~.hadamard.HADAMARD_KERNELS`), or ``"auto"``.
    """

    block: int
    kernel: str = "auto"

    def __call__(self, module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        return (hadamard_transform(args[0], self.block, self.kernel), *args[1:])


def input_hadamard_of(module: nn.Module) -> int | None:
    """The online Hadamard block of ``module``'s input, ``None`` if its input is not rotated."""
    block = getattr(module, ROTATED_ATTR, None)
    return block if isinstance(block, int) else None


def _weight_uniform(norm: nn.Module | None) -> bool:
    """Whether a per-head norm commutes with an orthogonal rotation of the head: no norm, or one without a weight
    or with the same scale on every channel."""
    weight = getattr(norm, "weight", None) if norm is not None else None
    if not isinstance(weight, Tensor) or weight.numel() <= 1:
        return True
    return bool(torch.all(weight == weight.reshape(-1)[0]))


def _value_maps(adapter: ArchAdapter, index: int) -> tuple[nn.Linear, list[nn.Linear], int] | str:
    """``(v_proj, o_proj of every layer reading its values, head_dim)`` of the source layer ``index``, or why its
    value rotation is skipped."""
    attention = adapter.attention_module(index)
    if attention is None:
        return "no softmax attention"
    if adapter.kv_source(index) != index:
        return f"reads the values of layer {adapter.kv_source(index)}"
    v_proj, q_proj = getattr(attention, "v_proj", None), getattr(attention, "q_proj", None)
    if type(v_proj) is not nn.Linear:
        return "no plain nn.Linear v_proj (shared K/V projection or a replaced map)"
    if not _weight_uniform(getattr(attention, "v_norm", None)):
        return "a per-channel value norm"
    geometry = adapter.head_geometry(index)
    if v_proj.out_features != geometry.kv_heads * geometry.head_dim:
        return f"v_proj has {v_proj.out_features} outputs, not kv_heads * head_dim"
    if isinstance(q_proj, nn.Linear) and q_proj.out_features != geometry.heads * geometry.head_dim:
        return "q_proj packs more than the queries (gated attention output)"
    outputs = []
    for reader in range(1, adapter.num_layers + 1):
        if adapter.kv_source(reader) != index:
            continue
        module = adapter.attention_module(reader)
        o_proj = getattr(module, "o_proj", None) if module is not None else None
        reader_geometry = adapter.head_geometry(reader)
        if type(o_proj) is not nn.Linear or reader_geometry.head_dim != geometry.head_dim:
            return f"layer {reader} reads its values without a plain nn.Linear o_proj of the same head width"
        if o_proj.in_features != reader_geometry.heads * geometry.head_dim:
            return f"layer {reader}'s o_proj has {o_proj.in_features} inputs, not heads * head_dim"
        outputs.append(o_proj)
    return v_proj, outputs, geometry.head_dim


def _rotate_values(v_proj: nn.Linear, outputs: list[nn.Linear], rotation: Tensor) -> None:
    """Fuse ``v -> v R`` into the rows of ``v_proj`` (per KV head) and ``R^T`` into the columns of each ``o_proj``."""
    head_dim = rotation.shape[0]
    weight = v_proj.weight.data
    heads = weight.view(-1, head_dim, weight.shape[1]).double()
    weight.copy_(torch.einsum("dr,kdi->kri", rotation, heads).reshape(weight.shape).to(weight.dtype))
    if v_proj.bias is not None:
        bias = v_proj.bias.data
        bias.copy_((bias.view(-1, head_dim).double() @ rotation).reshape(bias.shape).to(bias.dtype))
    for o_proj in outputs:
        out = o_proj.weight.data
        rotated = out.view(out.shape[0], -1, head_dim).double() @ rotation
        out.copy_(rotated.reshape(out.shape).to(out.dtype))


def _install(module: nn.Linear, block: int, kernel: str) -> None:
    setattr(module, _HOOK_ATTR, module.register_forward_pre_hook(InputHadamard(block, kernel)))
    setattr(module, ROTATED_ATTR, block)


@torch.no_grad()
def rotate_model(adapter: ArchAdapter, config: RotateConfig) -> dict[str, Any]:
    """Apply the rotations of ``config`` to the model of ``adapter`` in place and return a summary.

    The value rotation of a layer is fused into its ``v_proj`` and into the ``o_proj`` of every layer that reads
    its cache entries (KV sharing); a layer whose values pass through a per-channel operation is skipped and listed
    under ``"skipped"``. The weight of each online site becomes ``W H`` and a forward pre-hook rotates its input.

    All or nothing: if a rotation fails, every changed weight is restored before the error propagates.

    Args:
        adapter: The model's architecture adapter.
        config: What to rotate.

    Returns:
        ``{"method", "values": [rotated source layers], "online": {"<layer>.<map>": block}, "skipped": {...}}``.

    Raises:
        ConfigError: The config matches nothing in the model.
    """
    method: RotationMethod = ROTATIONS.get(config.method)()
    layers = resolve_layers(config.layers, adapter.num_layers)
    values: list[int] = []
    online: dict[str, int] = {}
    skipped: dict[str, str] = {}
    pending: list[tuple[nn.Linear, int]] = []
    with WeightTransaction() as transaction:
        for index in layers:
            if config.values:
                found = _value_maps(adapter, index)
                if isinstance(found, str):
                    skipped[f"{index}.values"] = found
                else:
                    v_proj, outputs, head_dim = found
                    for value_map in (v_proj, *outputs):
                        transaction.take(value_map)
                    rotation = method.value_rotation(head_dim, index, config.seed).to(v_proj.weight.device)
                    _rotate_values(v_proj, outputs, rotation)
                    values.append(index)
            sites = select_modules(adapter.linear_modules(adapter.layer(index)), config.online)
            for name, module in sites.items():
                if not isinstance(module, nn.Linear):
                    skipped[f"{index}.{name}"] = f"not an nn.Linear ({type(module).__name__})"
                    continue
                if input_hadamard_of(module) is not None:
                    raise ConfigError(f"layer {index} {name} is already rotated")
                block = hadamard_block(module.in_features, config.block)
                if block < 2:
                    skipped[f"{index}.{name}"] = f"input width {module.in_features} has no power-of-two factor"
                    continue
                transaction.take(module)
                weight = module.weight.data
                weight.copy_(hadamard_transform(weight.float(), block, "torch").to(weight.dtype))
                pending.append((module, block))
                online[f"{index}.{name}"] = block
        if not values and not online:
            raise ConfigError(f"rotate found nothing to rotate (skipped: {skipped or 'no map matches online'})")
    for module, block in pending:
        _install(module, block, config.kernel)
    return {"method": config.method, "values": values, "online": online, "skipped": skipped}
