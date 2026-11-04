"""EAGLE-3 drafting (Li et al., 2025): one draft decoder layer with its own KV cache reads the fused low, middle
and high features of the model and drafts a chain of tokens through its own head over a reduced vocabulary.

    import economize
    from economize.techniques.eagle3 import Eagle3Config

    model = economize.apply(hf_model, Eagle3Config(draft_vocab=32000))
    model.techniques["eagle3"].set_draft_vocab(frequent_ids)  # the 32000 most frequent tokens

The ``eagle3_ttt`` objective trains the draft layer with the training-time test on a frozen model. Official
checkpoints (``yuhuili/EAGLE3-*``, SpecForge, AngelSlim) load with ``economize.load(hf_model, path)``, EAGLE-3
heads saved as ``jump.pt`` with ``economize.load(hf_model, run_dir, format="tskip_eagle3")``.
"""

from __future__ import annotations

from . import formats as _formats  # noqa: F401  (registers the "eagle3" and "tskip_eagle3" checkpoint formats)
from . import objective as _objective  # noqa: F401  (registers the "eagle3_ttt" objective)
from .config import EAGLE3_HEADS, Eagle3Config, default_aux_layers
from .drafter import Eagle3Drafter
from .layer import Eagle3Layer
from .objective import Eagle3TTT
from .technique import Eagle3Technique

__all__ = [
    "EAGLE3_HEADS",
    "Eagle3Config",
    "Eagle3Drafter",
    "Eagle3Layer",
    "Eagle3TTT",
    "Eagle3Technique",
    "default_aux_layers",
]
