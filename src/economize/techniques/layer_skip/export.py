"""A layer-skip drafter in the jump drafter's export format (:mod:`economize.techniques.jump.export`).

A draft step of :class:`~economize.techniques.layer_skip.LayerSkipTechnique` embeds the token, runs every layer
outside the skip set and reads logits with the model's norm and head. That is the jump with ``m = u = 0``, the
parameter-free ``"embedding"`` predictor, a frozen path and the skip set as holes in it
(:attr:`~economize.techniques.jump.export.JumpDrafterSpec.skip_layers`), so every serving backend of jump drafters
serves layer skipping too.

The path ends at the last layer the draft step runs (``r``). When the skip set holds the model's last layers, ``r``
is below the last layer and the export writes an untrained readout adapter, which is the identity: the state after
the last run layer goes straight into the final norm, as in the technique.
"""

from __future__ import annotations

from typing import Any

import torch

from ...modules import ReadoutAdapter
from ..jump.config import JumpConfig
from ..jump.export import JumpDrafterSpec

__all__ = ["jump_config", "jump_spec", "jump_state_dict"]


def jump_config(run_layers: tuple[int, ...]) -> JumpConfig:
    """The jump configuration of a draft path that runs ``run_layers`` (``m = u = 0``, ``r`` the last of them)."""
    return JumpConfig(target_layer=0, source_layer=0, exit_layer=max(run_layers), predictor="embedding",
                      adaptation="frozen")


def jump_spec(technique: Any) -> JumpDrafterSpec:
    """The exported specification of an attached :class:`~economize.techniques.layer_skip.LayerSkipTechnique`."""
    adapter = technique.adapter
    run = technique.run_layers
    holes = tuple(layer for layer in technique.skip_layers if layer < max(run))
    return JumpDrafterSpec(jump_config(run), adapter.num_layers, adapter.hidden_size, skip_layers=holes)


def jump_state_dict(technique: Any) -> dict[str, torch.Tensor]:
    """The drafter's tensors under the jump technique's names: an identity ``readout.*`` when ``r < L``, else none."""
    if max(technique.run_layers) == technique.adapter.num_layers:
        return {}
    identity = ReadoutAdapter(technique.adapter.hidden_size)
    return {"readout." + name: tensor.detach() for name, tensor in identity.state_dict().items()}
