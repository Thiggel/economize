"""The EAGLE-1/2 technique: a one-layer draft decoder that extrapolates the model's last hidden state."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import register_technique
from .._draft_layer import DraftLayerTechnique
from .config import EagleConfig
from .drafter import EagleDrafter
from .layer import EagleLayer

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["EagleTechnique"]


@register_technique
class EagleTechnique(DraftLayerTechnique):
    """EAGLE-1 and EAGLE-2: one draft decoder layer on the model's features, with a static or dynamic draft tree.

    EAGLE-1 (Li et al., 2024) and, with a dynamic draft tree, EAGLE-2: one draft decoder layer with its own
    attention and KV cache (see :class:`~economize.techniques.eagle.EagleConfig`).

    Draft step at position ``p``: the layer reads ``fc([E[x_p]; f])``, where ``f`` is the model's final norm of its
    last block's output at ``p - 1`` when token ``p - 1`` was processed exactly and the layer's own output at
    ``p - 1`` otherwise. The model's LM head (and its ``transform_logits``) reads the output without the final norm
    and drafts token ``p + 1``.

    Parameters (float32, or the model's dtype with ``param_dtype="model"``; on the device of the LM head):
    ``layer.*``, ``fc`` among them (``fc.*`` and ``layers.0.*`` in the official checkpoints).
    """

    name = EagleConfig.technique
    config_class = EagleConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True, trainable=("layer",))
    drafter_class = EagleDrafter
    config: EagleConfig
    layer: EagleLayer

    def __init__(self, config: EagleConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)
        shape = self._shape()
        self.layer = EagleLayer(adapter.hidden_size, shape.intermediate_size, shape.num_heads, shape.num_kv_heads,
                                shape.head_dim, shape.rms_norm_eps, shape.attention_bias, config.fc_bias)
        self._init_rotary(shape)
        self._init_linears(self.layer)
        self._place()

    @property
    def source_layers(self) -> tuple[int, ...]:
        """The model's last block."""
        return (self.adapter.num_layers,)

    def fuse(self, states: Sequence[torch.Tensor]) -> torch.Tensor:
        """The feature ``f``: the model's final norm of its last block's output ``[..., d]``, in :attr:`dtype`."""
        adapter = self.adapter
        return adapter.final_norm(states[0].to(adapter.head_device)).to(self.device, self.dtype)

    def head_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Float logits ``[..., V]`` of the model's LM head on the draft layer's outputs ``[..., d]`` (no final
        norm)."""
        adapter = self.adapter
        logits = adapter.lm_head(hidden.to(adapter.head_device, adapter.dtype))
        return adapter.transform_logits(logits).float().to(self.device)

    def extra_repr(self) -> str:
        layer = self.layer
        return (f"fc_bias={self.config.fc_bias}, "
                f"heads={layer.num_heads}/{layer.num_kv_heads}x{layer.head_dim}")
