"""The ``"vllm/dmtd"`` exporter: a DMTD drafter as a jump drafter folder with ``m = u = L - L_d``."""

from __future__ import annotations

from torch import nn

from ....techniques.dynamic_depth.dmtd_export import jump_spec
from ....techniques.jump.export import JumpDrafterSpec
from ...exporters import register_exporter
from ..jump.export import JumpVllmExporter

__all__ = ["DmtdVllmExporter"]


@register_exporter("vllm", "dmtd")
class DmtdVllmExporter(JumpVllmExporter):
    """Writes the ``dmtd`` technique in the jump format (both layouts of :class:`JumpVllmExporter`).

    The draft step of the export runs the late layers ``L - L_d + 1..L`` on the next token's embedding (layer
    copies, LoRA updates or the target's own layers), then the final norm and the LM head, as
    :class:`~economize.techniques.dynamic_depth.DmtdTechnique` drafts. The tensors are the jump technique's.
    """

    def spec(self, technique: nn.Module) -> JumpDrafterSpec:
        return jump_spec(technique)
