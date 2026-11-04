"""Two-model speculative decoding with the draft LM on a device of its own (e.g. a second GPU).

    import economize
    from economize.techniques.placed_draft import PlacedDraftModelConfig

    model = economize.apply(target, PlacedDraftModelConfig(model="google/gemma-4-12B-it", device="cuda:1"))
    model.decode(prompt_ids, 128, decoding=economize.DecodeConfig(cycle_length=5))

The drafter is :class:`~economize.techniques.draft_model.DraftModelDrafter` with every draft-LM call on the draft
device: the committed tokens and draft tokens go there, and the float logits of each draft come back to the model's
device, where the verifier reads them. Use it when the draft LM does not fit next to the model.
"""

from __future__ import annotations

from .config import PlacedDraftModelConfig
from .drafter import PlacedDraftModelDrafter
from .technique import PlacedDraftModelTechnique

__all__ = ["PlacedDraftModelConfig", "PlacedDraftModelDrafter", "PlacedDraftModelTechnique"]
