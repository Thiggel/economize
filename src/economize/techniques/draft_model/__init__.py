"""Two-model speculative decoding: a separate, smaller LM of the same tokenizer drafts for the model.

    import economize
    from economize.techniques.draft_model import DraftModelConfig

    model = economize.apply(target, DraftModelConfig(model="Qwen/Qwen2.5-0.5B"))
    model.decode(prompt_ids, 128, decoding=economize.DecodeConfig(cycle_length=5))
"""

from __future__ import annotations

from .config import DraftModelConfig
from .drafter import CATCHUP, DraftModelDrafter
from .technique import DraftModelTechnique, load_draft_model

__all__ = ["CATCHUP", "DraftModelConfig", "DraftModelDrafter", "DraftModelTechnique", "load_draft_model"]
