"""The draft model class vLLM instantiates for an exported KnapSpec drafter (architecture ``EconomizeKnapSpecEagle3``).

A draft step is the jump drafter's (:class:`~economize.backends.vllm.jump.model.EconomizeJumpEagle3ForCausalLM`)
with ``m = u = 0``, the embedding predictor and the whole-skipped layers as holes, on the target's own layers. The
layers of ``skip_attention`` and ``skip_mlp`` (:class:`~economize.techniques.knapspec.export.KnapSpecDrafterSpec`)
run partially: their attention or their MLP-like modules are replaced by stand-ins that return zeros for the
duration of the draft call (:class:`~economize.techniques.knapspec.sublayer.PartialLayer`), so the target's forward
and the exact pass see the layers intact. A vLLM decoder layer adds both sublayer outputs to its residual stream
through (fused) RMSNorm adds, and every norm maps zeros to zeros, so a zeroed sublayer leaves the stream unchanged.
A skipped attention writes no K/V.

Which modules form the sublayers of a vLLM layer is a :class:`~economize.techniques.knapspec.sublayer.SublayerLayout`
found in :data:`VLLM_SUBLAYER_LAYOUTS` by the layer's class name; unregistered classes use :func:`vllm_default_layout`
(``self_attn``, and the first of ``mlp`` / ``feed_forward`` / ``block_sparse_moe``). Gemma-4's layout adds the MoE
block and the per-layer-input projection to the MLP and resets ``layer_scalar`` with it, as the Hugging Face layout.

The path is frozen (the target's layers); the drafter owns at most an identity readout.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
from torch import nn

from ....errors import ConfigError
from ....registry import Registry
from ....techniques.jump.export import JumpDrafterSpec
from ....techniques.knapspec.export import KnapSpecDrafterSpec
from ....techniques.knapspec.sublayer import PartialLayer, SublayerLayout, _ZeroModule
from ..jump.model import EconomizeJumpEagle3ForCausalLM, TargetModules, _draft_hf_config, _is_attention

__all__ = ["DRAFTER_ARCHITECTURE", "VLLM_SUBLAYER_LAYOUTS", "EconomizeKnapSpecEagle3ForCausalLM",
           "vllm_default_layout"]

#: Architecture name of exported KnapSpec drafters (``method="eagle3"`` keeps names that end in ``Eagle3``).
DRAFTER_ARCHITECTURE = "EconomizeKnapSpecEagle3"

_MLPS = ("mlp", "feed_forward", "block_sparse_moe")

#: Sublayer layouts of vLLM decoder layers by class name: ``fn(layer) -> SublayerLayout``.
VLLM_SUBLAYER_LAYOUTS: Registry[Callable[[nn.Module], SublayerLayout]] = Registry(
    "vLLM sublayer layout", "economize.vllm_sublayer_layouts"
)


def vllm_default_layout(layer: nn.Module) -> SublayerLayout:
    """The child that holds the layer's vLLM ``Attention``, and the layer's first MLP-like child."""
    attention = next((name for name, child in layer.named_children()
                      if any(_is_attention(module) for module in child.modules())), None)
    mlp = next((name for name in _MLPS if isinstance(getattr(layer, name, None), nn.Module)), None)
    return SublayerLayout(attention, () if mlp is None else (mlp,))


@VLLM_SUBLAYER_LAYOUTS.register("Gemma4DecoderLayer")
def _gemma4(layer: nn.Module) -> SublayerLayout:
    # The MoE block runs next to the MLP (its router output only feeds it), the per-layer input follows the MLP.
    base = vllm_default_layout(layer)
    extra = tuple(name for name in ("moe", "per_layer_projection") if isinstance(getattr(layer, name, None), nn.Module))
    return SublayerLayout(base.attention, base.mlp + extra, "layer_scalar" if hasattr(layer, "layer_scalar") else None)


class _ZeroAttention(nn.Module):
    """Stands in for a vLLM attention module (``self_attn(positions, hidden_states)``): zeros, and no K/V written."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[1 if len(args) > 1 else 0]
        return cast(torch.Tensor, hidden.new_zeros(*hidden.shape[:-1], self.width))


def _layout(layer: nn.Module) -> SublayerLayout:
    name = type(layer).__name__
    return VLLM_SUBLAYER_LAYOUTS.get(name)(layer) if name in VLLM_SUBLAYER_LAYOUTS else vllm_default_layout(layer)


class EconomizeKnapSpecEagle3ForCausalLM(EconomizeJumpEagle3ForCausalLM):
    """A KnapSpec drafter in vLLM's EAGLE-3 slot (see the module docstring).

    Args:
        vllm_config: The draft ``VllmConfig``; its draft ``hf_config`` holds the ``economize_knapspec`` block.
        prefix: Module prefix of the draft model (unused).
    """

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.knapspec_spec = KnapSpecDrafterSpec.from_hf_config(_draft_hf_config(vllm_config))
        if self.spec.config.adaptation != "frozen":
            raise ConfigError("a KnapSpec drafter runs the target's own layers: its path must be frozen")

    @classmethod
    def read_spec(cls, hf_config: Any) -> JumpDrafterSpec:
        return KnapSpecDrafterSpec.from_hf_config(hf_config).path

    def tie_to_target(self, target: nn.Module) -> TargetModules:
        """Tie the path (as the jump drafter does), then run the partial layers through sublayer stand-ins.

        Raises:
            ConfigError: As :meth:`EconomizeJumpEagle3ForCausalLM.tie_to_target`, or a partial layer has no module
                for the sublayer it skips.
        """
        tied = super().tie_to_target(target)
        width = self.spec.hidden_size
        attention, mlp = set(self.knapspec_spec.skip_attention), set(self.knapspec_spec.skip_mlp)
        path: list[Any] = []
        for index, layer in zip(self.spec.jump_layers, self._jump_path):
            layout = _layout(layer)
            if index in attention:
                if layout.attention is None:
                    raise ConfigError(f"layer {index} of the target ({type(layer).__name__}) has no attention module "
                                      f"to skip: register its layout in VLLM_SUBLAYER_LAYOUTS")
                path.append(PartialLayer(layer, {layout.attention: _ZeroAttention(width)}))
            elif index in mlp:
                if not layout.mlp:
                    raise ConfigError(f"layer {index} of the target ({type(layer).__name__}) has no MLP module to "
                                      f"skip: register its layout in VLLM_SUBLAYER_LAYOUTS")
                path.append(PartialLayer(layer, {name: _ZeroModule(width) for name in layout.mlp}, layout.scale))
            else:
                path.append(layer)
        self._jump_path = path
        return tied

    @property
    def attention_layer_names(self) -> list[str]:
        """The attention layers the draft step runs (layers of ``skip_attention`` write no K/V and are left out;
        when every layer of the path skips its attention, the path's layers keep the proposer's metadata builders)."""
        skipped = set(self.knapspec_spec.skip_attention)
        names = [cast(str, module.layer_name)
                 for index, layer in zip(self.spec.jump_layers, self.target.layers) if index not in skipped
                 for module in layer.modules() if _is_attention(module)]
        return names or super().attention_layer_names
