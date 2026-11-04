"""A DMTD drafter in the jump drafter's export format (:mod:`economize.techniques.jump.export`).

DMTD's draft step is the jump with ``m = u = L - L_d``, ``r = L`` and the parameter-free ``"embedding"`` predictor
(:class:`~economize.techniques.dynamic_depth.EmbeddingPredictor`): the late layers run on the next token's
embedding through the draft path's layer copies (``adaptation="full"``), LoRA updates (``"lora"``) or the model's
own layers (``"frozen"``), then the model's norm and head. The export writes the plain
:class:`~economize.techniques.jump.JumpConfig` part of :class:`~economize.techniques.dynamic_depth.DmtdConfig`; the
cycle fields (``decoding_layers``, ``cycle_length``) describe DMTD's lossy decoding, which a verifying backend does
not run. The tensors keep the jump technique's names (``blocks.*``, ``lora.layers.*``).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from ..jump.config import JumpConfig
from ..jump.export import JumpDrafterSpec

__all__ = ["jump_config", "jump_spec"]


def jump_config(config: Any) -> JumpConfig:
    """The jump configuration that drafts like the resolved :class:`~economize.techniques.dynamic_depth.DmtdConfig`
    ``config`` (its :class:`~economize.techniques.jump.JumpConfig` fields)."""
    return JumpConfig(**{field.name: getattr(config, field.name) for field in dataclasses.fields(JumpConfig)
                         if field.init})


def jump_spec(technique: Any) -> JumpDrafterSpec:
    """The exported specification of an attached :class:`~economize.techniques.dynamic_depth.DmtdTechnique`."""
    adapter = technique.adapter
    return JumpDrafterSpec(jump_config(technique.config), adapter.num_layers, adapter.hidden_size)
