"""Self-speculative early-exit drafting (LayerSkip, Kangaroo-style readout adapters).

Draft tokens run the model's first ``r`` layers and exit through the model's own head or a trained readout
adapter; the exact pass verifies them with the whole model.

    import economize
    from economize.techniques.early_exit import EarlyExitConfig

    model = economize.apply(hf_model, EarlyExitConfig(exit_layer=8, head="shared"))

The ``early_exit_kl`` objective trains the readout and the draft path's LoRA updates or layer copies. The
``layerskip`` objective (:class:`LayerSkip`, also importable from :mod:`economize.train`) instead trains the base
model so that its own head can exit early.
"""

from __future__ import annotations

from . import objective as _objective  # noqa: F401  (registers the "early_exit_kl" objective)
from .config import EXIT_ADAPTATIONS, EXIT_HEADS, EarlyExitConfig
from .drafter import EarlyExitDrafter, SelfDrafter, SelfDraftStep
from .layerskip import LayerSkip, exit_weights, layer_dropout_rates
from .objective import EarlyExitKL, path_parameter_groups
from .path import DraftPath
from .rollout import self_draft_rollout
from .technique import EarlyExitTechnique

__all__ = [
    "EXIT_ADAPTATIONS",
    "EXIT_HEADS",
    "DraftPath",
    "EarlyExitConfig",
    "EarlyExitDrafter",
    "EarlyExitKL",
    "EarlyExitTechnique",
    "LayerSkip",
    "SelfDraftStep",
    "SelfDrafter",
    "exit_weights",
    "layer_dropout_rates",
    "path_parameter_groups",
    "self_draft_rollout",
]
