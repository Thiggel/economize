"""LLM-Pruner's recovery stage: a LoRA update of the pruned model's maps that is always applied.

Ma et al. tune a rank-8 LoRA (``alpha = 16``) of the query, key, value, output, gate, up and down maps of every layer
on Alpaca after pruning. :class:`RecoveryLoRA` is that adapter: a :class:`~economize.modules.LayerLoRA` whose
updates are active on every call (the base model *is* the pruned model, so there is no exact path to protect).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch import Tensor, nn

from ...modules import LayerLoRA, LoRADelta
from .._structured.layout import DecoderLayout

__all__ = ["RecoveryLoRA", "merge_lora", "recovery_targets", "stub_targets"]


class RecoveryLoRA(LayerLoRA):
    """A :class:`~economize.modules.LayerLoRA` that adds its updates on every call once installed."""

    def __init__(self, targets: Mapping[int, Mapping[str, nn.Module]], rank: int, alpha: float) -> None:
        super().__init__(targets, rank, alpha)
        self._depth = 1

    def order_like(self, targets: Mapping[int, Mapping[str, nn.Module]]) -> None:
        """List the updates in the order of ``targets``, updates without a target last.

        A LoRA loaded from a checkpoint is created from the saved keys, which come back sorted (layer ``"10"``
        before ``"2"``, ``down_proj`` before ``q_proj``). A fresh one follows the targets. Putting a loaded LoRA in
        the targets' order gives both the same parameter order, which the optimizer state of a resumed run is
        matched by.
        """
        saved: dict[str, dict[str, nn.Module]] = {
            str(key): dict(deltas.items())  # type: ignore[operator]
            for key, deltas in self.layers.items()
        }
        layers = nn.ModuleDict()
        for index, linears in targets.items():
            key = str(index)
            deltas = saved.pop(key, {})
            layers[key] = nn.ModuleDict({**{n: deltas.pop(n) for n in linears if n in deltas}, **deltas})
        for key, rest in saved.items():
            layers[key] = nn.ModuleDict(rest)
        self.layers = layers

    @torch.no_grad()
    def merged(self) -> Iterator[tuple[nn.Module, Tensor]]:
        """The merged weight ``W + scale * B A`` of every map the adapter is bound to, one map at a time so that
        a caller writing them back never holds a second copy of the model (nothing is changed)."""
        for linear, delta in self._targets:
            weight = linear.weight
            assert isinstance(weight, Tensor)
            update = (delta.lora_B.double() @ delta.lora_A.double()) * delta.scale
            yield linear, (weight.double() + update.to(weight.device)).to(weight.dtype)


def recovery_targets(adapter: Any, layout: DecoderLayout) -> dict[int, dict[str, nn.Module]]:
    """The seven maps of every decoder layer (six without a value map, see :attr:`DecoderLayout.optional_maps`),
    keyed by their attribute names (``q_proj``, ..., ``down_proj``)."""
    names = (*layout.attn_inputs, layout.attn_output, *layout.mlp_inputs, layout.mlp_output)
    return {i: {name.rsplit(".", 1)[-1]: linear for name, linear in layout.maps(adapter.layer(i), names).items()}
            for i in range(1, adapter.num_layers + 1)}


class _Shape(nn.Module):
    """Stands in for a map of known size while a saved LoRA is loaded before the maps are pruned."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features, self.out_features = in_features, out_features


def stub_targets(state: Mapping[str, Tensor], prefix: str) -> dict[int, dict[str, nn.Module]]:
    """Targets shaped like the updates stored under ``prefix`` (``layers.<layer>.<name>.lora_A`` / ``lora_B``)."""
    shapes: dict[int, dict[str, list[int]]] = {}
    for key, value in state.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].split(".")
        if len(parts) != 4 or parts[0] != "layers":
            continue
        entry = shapes.setdefault(int(parts[1]), {}).setdefault(parts[2], [0, 0])
        if parts[3] == "lora_A":
            entry[0] = int(value.shape[1])
        elif parts[3] == "lora_B":
            entry[1] = int(value.shape[0])
    return {layer: {name: _Shape(*size) for name, size in maps.items()} for layer, maps in shapes.items()}


@torch.no_grad()
def merge_lora(lora: LayerLoRA, targets: Mapping[int, Mapping[str, nn.Module]]) -> dict[nn.Module, Tensor]:
    """The merged weight ``W + scale * B A`` of every adapted map (in the map's dtype; nothing is changed)."""
    merged: dict[nn.Module, Tensor] = {}
    for layer, deltas in lora.layers.items():
        for name, delta in deltas.items():  # type: ignore[operator]
            assert isinstance(delta, LoRADelta)
            linear = targets[int(layer)][name]
            weight = linear.weight
            assert isinstance(weight, Tensor)
            update = (delta.lora_B.double() @ delta.lora_A.double()) * delta.scale
            merged[linear] = (weight.double() + update.to(weight.device)).to(weight.dtype)
    return merged
