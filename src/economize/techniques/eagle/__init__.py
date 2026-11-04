"""EAGLE-1 (Li et al., 2024) and EAGLE-2 drafting: one draft decoder layer with its own KV cache extrapolates the
model's last hidden state and drafts through the model's LM head.

    import economize
    from economize import DecodeConfig

    model = economize.load(hf_model, "yuhuili/EAGLE-Vicuna-7B-v1.3")  # the official heads ("eagle" format)
    chain = model.decode(prompt_ids, 256, decoding=DecodeConfig(cycle_length=6))  # EAGLE-1: chains of 5 drafts
    tree = model.decode(prompt_ids, 256, decoding=DecodeConfig(  # EAGLE-2: a dynamic draft tree
        cycle_length=7, draft_policy="dynamic_tree", draft_policy_options={"depth": 6, "topk": 10, "total_tokens": 60}))

A new head (``economize.apply(hf_model, EagleConfig())``) trains with the ``hass`` objective by default (HASS, the
context-aligned EAGLE-2 recipe); any step-distillation objective of :mod:`economize.train` applies.
"""

from __future__ import annotations

from ...train.objective import register_default_objective
from . import formats as _formats  # noqa: F401  (registers the "eagle" checkpoint format)
from .config import EagleConfig
from .drafter import EagleDrafter
from .formats import EagleFormat
from .layer import EagleLayer
from .technique import EagleTechnique

register_default_objective(EagleTechnique.name, "hass")

__all__ = ["EagleConfig", "EagleDrafter", "EagleFormat", "EagleLayer", "EagleTechnique"]
