"""DFlash block drafting (Chen et al., 2026): a few draft layers conditioned on fused features of the model draft a
whole block of tokens in one forward from ``[anchor, mask, ..., mask]``, through the model's own LM head.

    import economize
    from economize.techniques.dflash import DFlashConfig

    model = economize.apply(hf_model, DFlashConfig(block_size=16, mask_token_id=4))

Official checkpoints load with ``economize.load(hf_model, path)`` (z-lab/dflash, e.g.
``z-lab/gemma-4-31B-it-DFlash``) or ``economize.load(hf_model, path, format="speculators_dflash")`` (vLLM
``speculators``, e.g. ``RedHatAI/gemma-4-31B-it-speculator.dflash``).
"""

from __future__ import annotations

from . import formats as _formats  # noqa: F401  (registers the "dflash" and "speculators_dflash" checkpoint formats)
from .config import DFLASH_LAYER_TYPES, DFlashConfig, default_target_layers
from .drafter import DFlashDrafter
from .layer import DFlashLayer
from .technique import DFlashTechnique

__all__ = [
    "DFLASH_LAYER_TYPES",
    "DFlashConfig",
    "DFlashDrafter",
    "DFlashLayer",
    "DFlashTechnique",
    "default_target_layers",
]
