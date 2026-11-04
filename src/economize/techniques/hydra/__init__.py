"""Hydra drafting (Ankner et al., 2024): Medusa-like heads where head ``j`` also reads the embeddings of the tokens
drafted by heads ``1..j - 1``, optionally over a prefix attention layer of its own (Hydra++, the default).

    import economize
    from economize.techniques.hydra import HydraConfig

    model = economize.apply(hf_model, HydraConfig(num_heads=4, tree="mc_sim_7b_63"))

The heads train with the ``step_distill`` objective on a frozen model (Hydra++'s teacher distillation).
"""

from __future__ import annotations

from . import objective as _objective  # noqa: F401  (names "step_distill" the default objective of "hydra")
from .config import HYDRA_HEADS, HydraConfig
from .drafter import HydraDrafter
from .heads import HydraHead, InputBlock
from .prefix import PrefixLayer
from .technique import HydraTechnique

__all__ = [
    "HYDRA_HEADS",
    "HydraConfig",
    "HydraDrafter",
    "HydraHead",
    "HydraTechnique",
    "InputBlock",
    "PrefixLayer",
]
