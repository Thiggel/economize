"""The ``"vllm/kangaroo"`` exporter: a trained Kangaroo adapter as a vLLM EAGLE-3 draft model folder.

The folder is the jump drafter's (:class:`~economize.backends.vllm.jump.export.JumpVllmExporter`, both layouts) with
the architecture ``EconomizeKangarooEagle3``, the ``economize_kangaroo`` block
(:class:`~economize.techniques.kangaroo.export.KangarooDrafterSpec`, which nests the path ``1..l`` as a jump block),
the state after layer ``l`` as the only auxiliary state, and the adapter's tensors (``kangaroo.*``).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ....techniques.jump.export import JumpDrafterSpec
from ....techniques.kangaroo.export import CONFIG_KEY, WEIGHTS_PREFIX, KangarooDrafterSpec, adapter_state
from ...exporters import register_exporter
from ..jump.export import JumpVllmExporter
from .model import DRAFTER_ARCHITECTURE

__all__ = ["KangarooVllmExporter"]


@register_exporter("vllm", "kangaroo")
class KangarooVllmExporter(JumpVllmExporter):
    """Writes the ``kangaroo`` technique for
    :class:`~economize.backends.vllm.kangaroo.model.EconomizeKangarooEagle3ForCausalLM`."""

    architecture = DRAFTER_ARCHITECTURE
    config_key = CONFIG_KEY

    def spec(self, technique: nn.Module) -> JumpDrafterSpec:
        return KangarooDrafterSpec.of(technique).path

    def drafter_state(self, technique: nn.Module) -> dict[str, torch.Tensor]:
        return adapter_state(technique)

    def config_block(self, technique: nn.Module, spec: JumpDrafterSpec) -> dict[str, Any]:
        adapter: Any = technique.adapter  # the ArchAdapter of the technique (an nn.Module attribute)
        shared = [*(adapter.layer(index) for index in spec.jump_layers), adapter.embed_tokens, adapter.lm_head]
        return KangarooDrafterSpec.of(technique).to_dict() | {
            "shared_with_target": [adapter.module_path(module) for module in shared],
        }

    def aux_layers(self, technique: nn.Module, spec: JumpDrafterSpec) -> list[int]:
        """The state after layer ``l``: the adapter's K/V of verified tokens is computed from it."""
        return [spec.exit_layer]

    def live_tensors(self, model: Any) -> dict[str, torch.Tensor]:
        """The adapter's tensors as they live in the model (detached), keyed ``kangaroo.<name>``."""
        return {WEIGHTS_PREFIX + name: tensor for name, tensor in self.drafter_state(self.technique_of(model)).items()}
