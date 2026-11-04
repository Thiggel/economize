"""The ``"vllm/hydra"`` exporter: a Hydra drafter as a vLLM EAGLE-3 draft model folder.

The folder follows the jump drafter's layouts (:class:`~economize.backends.vllm.jump.export.JumpVllmExporter`,
``eagle3`` or ``speculators``) with the architecture ``EconomizeHydraEagle3``, the ``economize_hydra`` block
(:class:`~economize.techniques.hydra.export.HydraDrafterSpec`), the auxiliary state ``[L]`` (the target's last
hidden state) and the technique's tensors keyed ``hydra.*``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ....techniques.hydra.export import CONFIG_KEY, WEIGHTS_PREFIX, HydraDrafterSpec, drafter_state
from ...exporters import register_exporter
from ..jump.export import JumpVllmExporter
from .model import DRAFTER_ARCHITECTURE

__all__ = ["HydraVllmExporter"]


@register_exporter("vllm", "hydra")
class HydraVllmExporter(JumpVllmExporter):
    """Writes the ``hydra`` technique for
    :class:`~economize.backends.vllm.hydra.model.EconomizeHydraEagle3ForCausalLM`."""

    architecture = DRAFTER_ARCHITECTURE
    config_key = CONFIG_KEY

    def spec(self, technique: nn.Module) -> HydraDrafterSpec:  # type: ignore[override]
        return HydraDrafterSpec.of(technique)

    def drafter_state(self, technique: nn.Module) -> dict[str, torch.Tensor]:
        return drafter_state(technique)

    def config_block(self, technique: nn.Module, spec: HydraDrafterSpec) -> dict[str, Any]:  # type: ignore[override]
        adapter: Any = technique.adapter  # the ArchAdapter of the technique (an nn.Module attribute)
        shared = [adapter.module_path(module) for module in (adapter.final_norm, adapter.embed_tokens,
                                                              adapter.lm_head)]
        return spec.to_dict() | {"shared_with_target": shared}

    def aux_layers(self, technique: nn.Module, spec: HydraDrafterSpec) -> list[int]:  # type: ignore[override]
        """The target's last hidden state (layer ``L``, before the final norm)."""
        return [spec.num_layers]

    def live_tensors(self, model: Any) -> dict[str, torch.Tensor]:
        """The technique's state dict as it lives in the model (detached), keyed ``hydra.<name>``."""
        return {WEIGHTS_PREFIX + name: tensor for name, tensor in self.drafter_state(self.technique_of(model)).items()}
