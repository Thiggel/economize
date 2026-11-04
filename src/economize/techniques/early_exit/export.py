"""An early-exit drafter in the jump drafter's export format (:mod:`economize.techniques.jump.export`).

Early exit after layer ``r`` is the jump with ``m = u = 0`` and the parameter-free ``"embedding"`` predictor
(:class:`~economize.techniques.dynamic_depth.EmbeddingPredictor`, DMTD's): the draft step runs layers ``1..r`` on
the token's embedding and reads logits through the readout adapter and the model's norm and head. Writing it in the
jump format lets every serving backend of jump drafters serve early exit too, without a draft model of its own.

The tensors keep their names with the ``path.`` prefix dropped (``readout.*``, ``lora.layers.*``, ``blocks.*``).
``head="shared"`` has no readout of its own; the export writes an untrained readout adapter, which is the identity.
"""

from __future__ import annotations

from typing import Any

import torch

from ...modules import ReadoutAdapter
from ..jump.config import JumpConfig
from ..jump.export import JumpDrafterSpec
from .config import EarlyExitConfig

__all__ = ["jump_config", "jump_spec", "jump_state_dict"]


def jump_config(config: EarlyExitConfig) -> JumpConfig:
    """The jump configuration that drafts like ``config`` (``m = u = 0``, ``r = exit_layer``, embedding predictor)."""
    return JumpConfig(
        target_layer=0, source_layer=0, exit_layer=config.exit_layer, predictor="embedding",
        adaptation=config.adaptation, lora_rank=config.lora_rank, lora_alpha=config.resolved_lora_alpha,
        copy_dtype=config.copy_dtype,
    )


def jump_spec(technique: Any) -> JumpDrafterSpec:
    """The exported specification of an attached :class:`~economize.techniques.early_exit.EarlyExitTechnique`."""
    adapter = technique.adapter
    return JumpDrafterSpec(jump_config(technique.config), adapter.num_layers, adapter.hidden_size)


def jump_state_dict(technique: Any) -> dict[str, torch.Tensor]:
    """The technique's tensors under the jump technique's names (detached, where they live).

    ``readout.*`` is the trained readout adapter, or an identity adapter for ``head="shared"``; ``lora.*`` and
    ``blocks.*`` are the draft path's LoRA factors or layer copies.
    """
    state = {name.removeprefix("path."): tensor.detach() for name, tensor in technique.state_dict().items()}
    if technique.readout is None:
        identity = ReadoutAdapter(technique.adapter.hidden_size)
        state.update({"readout." + name: tensor for name, tensor in identity.state_dict().items()})
    return state
