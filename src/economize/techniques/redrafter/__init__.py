"""ReDrafter (Cheng et al., 2024): a recurrent draft head on the model's last hidden state, drafting a chain or
running beam search whose beams the verifier checks at once as a tree.

    import economize
    from economize.techniques.redrafter import RedrafterConfig

    model = economize.apply(hf_model, RedrafterConfig(beam_width=8, beam_length=5))

The head trains with the ``step_distill`` objective on a frozen model (ReDrafter's knowledge distillation).
"""

from __future__ import annotations

from . import objective as _objective  # noqa: F401  (names "step_distill" the default objective of "redrafter")
from .config import REDRAFTER_HEADS, REDRAFTER_INPUT_INITS, RedrafterConfig
from .drafter import RedrafterDrafter, beam_tree
from .technique import LOG_0, SELECT_EXTRA_SCALE, RedrafterTechnique

__all__ = [
    "LOG_0",
    "REDRAFTER_HEADS",
    "REDRAFTER_INPUT_INITS",
    "SELECT_EXTRA_SCALE",
    "RedrafterConfig",
    "RedrafterDrafter",
    "RedrafterTechnique",
    "beam_tree",
]
