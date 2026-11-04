"""Medusa drafting (Cai et al., 2024): ``K`` heads on the model's last hidden state draft the ``K`` tokens after the
next one in a single step, as a chain or as a static sparse tree.

    import economize
    from economize.techniques.medusa import MedusaConfig

    model = economize.apply(hf_model, MedusaConfig(num_heads=4, tree="mc_sim_7b_63"))

The ``medusa_ce`` objective trains the heads on a frozen model (Medusa-1). Checkpoints of FasterDecoding/Medusa
(``medusa_lm_head.safetensors`` with a ``config.json`` naming ``medusa_num_heads``) load with ``economize.load``;
:func:`keep_heads` drops trailing heads of a read checkpoint (``economize.attach(model,
keep_heads(read_checkpoint(path), 4))``).
"""

from __future__ import annotations

from . import formats as _formats  # noqa: F401  (registers the "medusa" and "tskip_medusa" checkpoint formats)
from . import objective as _objective  # noqa: F401  (registers the "medusa_ce" objective)
from .config import MEDUSA_HEADS, MedusaConfig
from .drafter import MedusaDrafter
from .formats import keep_heads
from .heads import MedusaHead, ResBlock
from .objective import MedusaCE
from .technique import MedusaTechnique
from .trees import MEDUSA_TREES, TreePlan

__all__ = [
    "MEDUSA_HEADS",
    "MEDUSA_TREES",
    "MedusaCE",
    "MedusaConfig",
    "MedusaDrafter",
    "MedusaHead",
    "MedusaTechnique",
    "ResBlock",
    "TreePlan",
    "keep_heads",
]
