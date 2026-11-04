"""The ``"vllm/hidden_transfer"`` exporter: Hidden Transfer maps as a vLLM EAGLE-3 draft model folder.

The folder follows the jump drafter's layouts (:class:`~economize.backends.vllm.jump.export.JumpVllmExporter`,
``eagle3`` or ``speculators``) with the architecture ``EconomizeHiddenTransferEagle3``, the
``economize_hidden_transfer`` block
(:class:`~economize.techniques.dynamic_depth.hidden_transfer_export.HiddenTransferDrafterSpec`,
which nests the path ``t_1 + 1..L`` as a jump block), the states after the distinct transfer layers as auxiliary
states and the maps keyed ``hidden_transfer.transfers.{i}.weight``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ....techniques.dynamic_depth.hidden_transfer_export import (
    CONFIG_KEY,
    WEIGHTS_PREFIX,
    HiddenTransferDrafterSpec,
    drafter_state,
)
from ....techniques.jump.export import JumpDrafterSpec
from ...exporters import register_exporter
from ..jump.export import JumpVllmExporter
from .model import DRAFTER_ARCHITECTURE

__all__ = ["HiddenTransferVllmExporter"]


@register_exporter("vllm", "hidden_transfer")
class HiddenTransferVllmExporter(JumpVllmExporter):
    """Writes the ``hidden_transfer`` technique for
    :class:`~economize.backends.vllm.hidden_transfer.model.EconomizeHiddenTransferEagle3ForCausalLM`."""

    architecture = DRAFTER_ARCHITECTURE
    config_key = CONFIG_KEY

    def spec(self, technique: nn.Module) -> JumpDrafterSpec:
        return HiddenTransferDrafterSpec.of(technique).path

    def drafter_state(self, technique: nn.Module) -> dict[str, torch.Tensor]:
        return drafter_state(technique)

    def config_block(self, technique: nn.Module, spec: JumpDrafterSpec) -> dict[str, Any]:
        adapter: Any = technique.adapter  # the ArchAdapter of the technique (an nn.Module attribute)
        shared = [*(adapter.layer(index) for index in spec.jump_layers), adapter.final_norm, adapter.embed_tokens,
                  adapter.lm_head]
        return HiddenTransferDrafterSpec.of(technique).to_dict() | {
            "shared_with_target": [adapter.module_path(module) for module in shared],
        }

    def aux_layers(self, technique: nn.Module, spec: JumpDrafterSpec) -> list[int]:
        """The states after the distinct transfer layers, ascending (a round reads them at ``P`` only)."""
        return list(HiddenTransferDrafterSpec.of(technique).source_layers)

    def live_tensors(self, model: Any) -> dict[str, torch.Tensor]:
        """The maps as they live in the model (detached), keyed ``hidden_transfer.<name>``."""
        return {WEIGHTS_PREFIX + name: tensor for name, tensor in self.drafter_state(self.technique_of(model)).items()}
