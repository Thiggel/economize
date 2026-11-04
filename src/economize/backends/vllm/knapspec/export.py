"""The ``"vllm/knapspec"`` exporter: the current KnapSpec skip set as a vLLM EAGLE-3 draft model folder.

The folder is the jump drafter's (:class:`~economize.backends.vllm.jump.export.JumpVllmExporter`, both layouts) with
the architecture ``EconomizeKnapSpecEagle3``, the ``economize_knapspec`` block
(:class:`~economize.techniques.knapspec.export.KnapSpecDrafterSpec`, which nests the path as a jump block) and at most
an identity readout.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ....techniques.jump.export import JumpDrafterSpec
from ....techniques.knapspec.export import CONFIG_KEY, KnapSpecDrafterSpec, drafter_state
from ...exporters import register_exporter
from ..jump.export import JumpVllmExporter
from .model import DRAFTER_ARCHITECTURE

__all__ = ["KnapSpecVllmExporter"]


@register_exporter("vllm", "knapspec")
class KnapSpecVllmExporter(JumpVllmExporter):
    """Writes the ``knapspec`` technique for
    :class:`~economize.backends.vllm.knapspec.model.EconomizeKnapSpecEagle3ForCausalLM`."""

    architecture = DRAFTER_ARCHITECTURE
    config_key = CONFIG_KEY

    def spec(self, technique: nn.Module) -> JumpDrafterSpec:
        return KnapSpecDrafterSpec.of(technique).path

    def drafter_state(self, technique: nn.Module) -> dict[str, torch.Tensor]:
        return drafter_state(technique)

    def config_block(self, technique: nn.Module, spec: JumpDrafterSpec) -> dict[str, Any]:
        shared = super().config_block(technique, spec)["shared_with_target"]
        return KnapSpecDrafterSpec.of(technique).to_dict() | {"shared_with_target": shared}
