"""The ``"vllm/layer_skip"`` exporter: a layer-skip drafter as a jump drafter folder with ``m = u = 0`` and holes."""

from __future__ import annotations

import torch
from torch import nn

from ....techniques.jump.export import JumpDrafterSpec
from ....techniques.layer_skip.export import jump_spec, jump_state_dict
from ...exporters import register_exporter
from ..jump.export import JumpVllmExporter

__all__ = ["LayerSkipVllmExporter"]


@register_exporter("vllm", "layer_skip")
class LayerSkipVllmExporter(JumpVllmExporter):
    """Writes the ``layer_skip`` technique in the jump format (both layouts of :class:`JumpVllmExporter`).

    The draft step of the export runs the target's own layers outside the skip set on the token's embedding, then
    the final norm and the LM head, as :class:`~economize.techniques.layer_skip.LayerSkipTechnique` drafts.
    """

    def spec(self, technique: nn.Module) -> JumpDrafterSpec:
        return jump_spec(technique)

    def drafter_state(self, technique: nn.Module) -> dict[str, torch.Tensor]:
        return jump_state_dict(technique)
