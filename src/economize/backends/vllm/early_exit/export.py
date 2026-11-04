"""The ``"vllm/early_exit"`` exporter: an early-exit drafter as a jump drafter folder with ``m = u = 0``."""

from __future__ import annotations

import torch
from torch import nn

from ....techniques.early_exit.export import jump_spec, jump_state_dict
from ....techniques.jump.export import JumpDrafterSpec
from ...exporters import register_exporter
from ..jump.export import JumpVllmExporter

__all__ = ["EarlyExitVllmExporter"]


@register_exporter("vllm", "early_exit")
class EarlyExitVllmExporter(JumpVllmExporter):
    """Writes the ``early_exit`` technique in the jump format (both layouts of :class:`JumpVllmExporter`).

    The draft step of the export runs layers ``1..r`` on the token's embedding through the draft path's LoRA
    updates or layer copies, then the readout adapter (the identity for ``head="shared"``), the final norm and the
    LM head, as :class:`~economize.techniques.early_exit.EarlyExitTechnique` drafts.
    """

    def spec(self, technique: nn.Module) -> JumpDrafterSpec:
        return jump_spec(technique)

    def drafter_state(self, technique: nn.Module) -> dict[str, torch.Tensor]:
        return jump_state_dict(technique)
