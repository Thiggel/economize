"""The cross-layer self-drafter that DMTD specializes (private to :mod:`economize.techniques.dynamic_depth`).

DMTD's draft steps enter the model after a later layer from a predicted state and run the remaining layers, which is
economize's cross-layer self-drafter with its source and target layer both at ``L - L_d``. This module names that
drafter's configuration, technique, predictor registry and rollout objective for the DMTD module.
"""

from __future__ import annotations

from ..jump.config import JumpConfig as CrossLayerConfig
from ..jump.objective import JumpRollout as CrossLayerRollout
from ..jump.predictor import PREDICTORS, Predictor
from ..jump.technique import JumpTechnique as CrossLayerTechnique

__all__ = ["PREDICTORS", "CrossLayerConfig", "CrossLayerRollout", "CrossLayerTechnique", "Predictor"]
