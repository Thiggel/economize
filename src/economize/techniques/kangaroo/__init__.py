"""Kangaroo (Liu et al., NeurIPS 2024): self-speculative drafting through the model's first ``l`` layers, a trained
one-layer attention adapter and the model's LM head, with a double early exit.

    import economize
    from economize import DecodeConfig
    from economize.techniques.kangaroo import KangarooConfig

    model = economize.apply(hf_model, KangarooConfig(exit_layer=2),
                            decoding=DecodeConfig(cycle_length=7, min_confidence=0.6))
    economize.train.fit(model, data, "step_distill", steps=...)  # trains the adapter only

The second exit (stop drafting once the top-1 probability falls below 0.6) is the decoder's dynamic draft length.
"""

from __future__ import annotations

from .adapter import KangarooAdapter
from .config import KangarooConfig
from .drafter import KangarooDrafter
from .technique import KangarooTechnique

__all__ = ["KangarooAdapter", "KangarooConfig", "KangarooDrafter", "KangarooTechnique"]
