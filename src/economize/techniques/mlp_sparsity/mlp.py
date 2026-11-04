"""The gated MLPs a channel-sparsity technique acts on, and the patch that replaces their forward.

Both techniques of this package rewrite the forward of a gated MLP ``down(act(gate(x)) * up(x))`` (Llama, Qwen,
Mistral, Gemma and the other dense decoders of the ecosystem). The submodules are looked up when the MLP runs, so
a weight transform that replaces a map (quantization, a LoRA wrapper) is picked up without reinstalling.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from ...core.calibration import resolve_layers
from ...errors import UnsupportedModelError
from ...runtime.phase import current_phase
from ..sparsity.sparsify import PhasePolicy, layer_tokens

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from .backends import MlpOps
    from .config import MlpSparsityConfig

__all__ = [
    "GatedMlp",
    "MlpPatch",
    "MlpRule",
    "call_mode",
    "dense_first_half",
    "ffn_width",
    "gated_mlps",
    "masked",
    "topk_mask",
]


class GatedMlp:
    """The four parts of one gated MLP, looked up by attribute name at every access.

    Attributes:
        module: The MLP module.
    """

    def __init__(self, module: nn.Module, config: MlpSparsityConfig) -> None:
        self.module = module
        self._names = (config.gate, config.up, config.down, config.act)

    @property
    def gate(self) -> nn.Module:
        """The gate map."""
        return getattr(self.module, self._names[0])

    @property
    def up(self) -> nn.Module:
        """The up map."""
        return getattr(self.module, self._names[1])

    @property
    def down(self) -> nn.Module:
        """The down map."""
        return getattr(self.module, self._names[2])

    @property
    def act(self) -> Callable[[Tensor], Tensor]:
        """The activation function."""
        return getattr(self.module, self._names[3])


#: The sparse forward of one MLP: ``rule(mlp, x, ops, mode)`` with ``mode`` ``"all"`` or ``"second_half"``.
MlpRule = Callable[[GatedMlp, Tensor, "MlpOps", str], Tensor]


def ffn_width(mlp: GatedMlp) -> int:
    """The number of FFN channels (the output width of the gate map).

    Raises:
        UnsupportedModelError: The gate map has no readable output width.
    """
    gate = mlp.gate
    width = getattr(gate, "out_features", None)
    if isinstance(width, int):
        return width
    weight = getattr(gate, "weight", None)
    if isinstance(weight, Tensor) and weight.dim() == 2:
        return int(weight.shape[0])
    raise UnsupportedModelError(f"cannot read the FFN width of the gate map {type(gate).__name__}")


def gated_mlps(adapter: ArchAdapter, config: MlpSparsityConfig) -> dict[int, GatedMlp]:
    """The gated MLP of every selected layer, keyed by the 1-indexed layer.

    Raises:
        UnsupportedModelError: A layer has no MLP under ``config.mlp`` or the MLP lacks one of the gate, up, down and
            activation attributes (a fused ``gate_up_proj`` as in Phi-3, a mixture of experts).
    """
    found: dict[int, GatedMlp] = {}
    for index in sorted(resolve_layers(config.layers, adapter.num_layers)):
        module = getattr(adapter.layer(index), config.mlp, None)
        if not isinstance(module, nn.Module):
            raise UnsupportedModelError(f"layer {index} has no MLP module {config.mlp!r}")
        missing = [n for n in (config.gate, config.up, config.down, config.act) if not hasattr(module, n)]
        if missing:
            raise UnsupportedModelError(
                f"{type(module).__name__} of layer {index} is not a gated MLP with separate maps (missing {missing}); "
                "fused gate/up maps and mixtures of experts are not supported"
            )
        found[index] = GatedMlp(module, config)
    return found


def call_mode(x: Tensor, policy: PhasePolicy) -> str:
    """The mode (:data:`..sparsity.sparsify.PREFILL_MODES`) of a call with input ``x`` in the current phase, read
    like TEAL's: positions per sequence from the tracked decoder layer, else from ``x.shape[-2]``."""
    layout = layer_tokens()
    if layout is None and x.dim() < 2:
        return "all"
    tokens = layout[1] if layout is not None else x.shape[-2]
    return policy.mode(current_phase(), tokens)


def dense_first_half(keep: Tensor) -> Tensor:
    """``keep`` with every channel kept on the first positions of every sequence (all but the last ``max(1, S // 2)``),
    the ``"second_half"`` protocol of :func:`..sparsity.sparsify.sparsify_tokens`."""
    shape = keep.shape
    layout = layer_tokens()
    if layout is not None and keep.shape[:-1].numel() == layout[0] * layout[1]:
        keep = keep.reshape(*layout, shape[-1])
    elif keep.dim() < 2:
        return keep
    half = max(1, keep.shape[-2] // 2)
    keep = keep.clone()
    keep[..., : keep.shape[-2] - half, :] = True
    return keep.reshape(shape)


class MlpPatch:
    """The replaced ``forward`` of one MLP; :meth:`remove` restores it.

    Calls in a ``"dense"`` mode of ``policy`` run the original forward; the others run ``rule``.
    """

    def __init__(self, mlp: GatedMlp, rule: MlpRule, ops: MlpOps, policy: PhasePolicy) -> None:
        module = mlp.module
        self.module = module
        self.ops = ops
        self._had_forward = "forward" in module.__dict__
        original = module.__dict__.get("forward") or type(module).forward.__get__(module)

        def forward(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
            mode = call_mode(x, policy)
            if mode == "dense":
                return original(x, *args, **kwargs)
            return rule(mlp, x, ops, mode)

        self._previous = module.__dict__.get("forward")
        module.forward = forward

    def remove(self) -> None:
        module = self.module
        if self._had_forward:
            module.forward = self._previous  # type: ignore[assignment]
        else:
            module.__dict__.pop("forward", None)
        self.ops.remove()


def masked(x: Tensor, keep: Tensor) -> Tensor:
    """``x`` with the dropped entries set to exactly 0."""
    return x.masked_fill(~keep, 0)


def topk_mask(scores: Tensor, k: int) -> Tensor:
    """Boolean mask of the ``k`` largest entries of ``scores`` along the last dimension."""
    index = scores.topk(k, dim=-1).indices
    return torch.zeros(scores.shape, dtype=torch.bool, device=scores.device).scatter_(-1, index, True)
