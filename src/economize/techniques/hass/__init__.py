"""HASS (Zhang et al., ICLR 2025): an EAGLE-3 drafter trained with harmonized objective distillation (a top-K
term) and harmonized context alignment (unrolled steps that read the earlier steps' outputs without gradients).

    import economize
    from economize.techniques.eagle3 import Eagle3Config
    from economize.techniques.hass import HassDistill

    model = economize.apply(hf_model, Eagle3Config(draft_vocab=32000))
    economize.train.fit(model, data, HassDistill(technique="eagle3"), steps=...)

The drafter is :class:`~economize.techniques.eagle3.Eagle3Technique` itself, so HASS checkpoints save, load and
decode as EAGLE-3 ones. This package registers only the ``hass`` objective; another drafter becomes trainable with
it by implementing :class:`ContextAlignedDrafter`.
"""

from __future__ import annotations

from .objective import ContextAlignedDrafter, HassDistill, top_k_cross_entropy

__all__ = ["ContextAlignedDrafter", "HassDistill", "top_k_cross_entropy"]
