"""Configuration of EAGLE-1/2 drafting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .._draft_layer import DraftLayerConfig

__all__ = ["EagleConfig"]


@dataclass(frozen=True)
class EagleConfig(DraftLayerConfig):
    """Hyperparameters of an EAGLE-1 draft head (Li et al., 2024, arXiv 2401.15077), which EAGLE-2 (arXiv
    2406.16858) drafts with a dynamic tree.

    At position ``t`` the drafter runs one decoder layer with its own attention and KV cache on
    ``fc([E[x_t]; f_{t-1}])``, where ``f_{t-1}`` is the model's last hidden state of token ``t - 1`` after the final
    norm (the feature EAGLE extrapolates) when that token was processed exactly, and the layer's own output at
    ``t - 1`` otherwise. The model's LM head reads the layer's output directly (without the final norm) and drafts
    token ``t + 1``. The layer has no input norm. Its shape, rotary table and training options are those of
    :class:`~economize.techniques._draft_layer.DraftLayerConfig`.

    EAGLE-2 is this head with ``DecodeConfig(draft_policy="dynamic_tree")``; the official heads
    (``yuhuili/EAGLE-*``) serve both.

    Attributes:
        fc_bias: A bias on ``fc`` (the official default; ``yuhuili/EAGLE-LLaMA3-Instruct-8B`` has none).
    """

    technique: ClassVar[str] = "eagle"

    fc_bias: bool = True
