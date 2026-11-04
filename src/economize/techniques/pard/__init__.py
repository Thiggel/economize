"""PARD: a draft LM that proposes ``k`` tokens in one forward pass (An et al., 2025, arXiv 2504.18583).

    import economize
    from economize.techniques.pard import PardConfig

    model = economize.apply(target, PardConfig(model="path/to/pard-draft"))
    model.decode(prompt_ids, 128, decoding=economize.DecodeConfig(cycle_length=9))

A PARD draft LM is a small causal LM of the model's tokenizer, fine-tuned so that a run of mask tokens after the
context predicts the tokens that follow it: in ``[x_1 .. x_t, m, m, .., m]`` the logits at ``x_t`` give the first
draft and the logits at the ``i``-th mask the draft ``i + 1``. The drafter is
:class:`~economize.techniques.draft_model.DraftModelDrafter` with every cycle's catch-up and all its drafts in one
draft-LM call. The mask positions are not kept in the draft LM's cache: the next cycle's catch-up overwrites them.
"""

from __future__ import annotations

from .config import PardConfig
from .drafter import PardDrafter
from .technique import PardTechnique

__all__ = ["PardConfig", "PardDrafter", "PardTechnique"]
