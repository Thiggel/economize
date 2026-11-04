"""Cross-layer jump drafting.

Between two exact passes, each draft token skips layers ``1..m`` (and ``r+1..L``): a small predictor maps the
previous token's state after layer ``u`` and the new token's embedding to the new token's state after layer ``m``,
and only layers ``m+1..r`` run.

    import economize
    from economize.techniques.jump import JumpConfig

    model = economize.apply(hf_model, JumpConfig(target_layer=12, source_layer=18))

Predictors and source mixers are extension points: register new ones in :data:`PREDICTORS` / :data:`MIXERS`.

Training lives here too: :func:`jump_rollout` (parallel open-loop rollouts), the ``jump_rollout`` and ``jump_aware``
objectives (:class:`JumpRollout`, :class:`JumpAware`, registered in :data:`economize.train.OBJECTIVES`) and the RL
pieces :func:`hybrid_log_probs` and :func:`jump_acceptance`. :mod:`.export` defines the backend-neutral folder
format a serving backend reads (:class:`JumpDrafterSpec`, :class:`ExportedJumpDrafter`).
"""

from __future__ import annotations

from .aware import JumpAware, shared_parameters
from .config import ADAPTATIONS, JumpConfig
from .drafter import JumpDrafter
from .export import ExportedJumpDrafter, JumpDrafterSpec
from .mixer import MIXERS, ConcatMixer, FusedLinearMixer, LayerAttentionMixer, SourceMixer
from .objective import CURVE_METRICS, JumpRollout
from .predictor import (
    PREDICTORS,
    BlockPredictor,
    CachedPredictor,
    LinearBlockPredictor,
    LinearPredictor,
    MLPPredictor,
    Predictor,
    PredictorMemory,
    SequencePredictor,
)
from .predictor_flow import FlowLinearPredictor, FlowPredictor
from .rl import hybrid_log_probs, jump_acceptance
from .rollout import StepFn, jump_rollout
from .technique import JumpTechnique

__all__ = [
    "ADAPTATIONS",
    "CURVE_METRICS",
    "MIXERS",
    "PREDICTORS",
    "BlockPredictor",
    "CachedPredictor",
    "ConcatMixer",
    "ExportedJumpDrafter",
    "FlowLinearPredictor",
    "FlowPredictor",
    "FusedLinearMixer",
    "JumpAware",
    "JumpConfig",
    "JumpDrafter",
    "JumpDrafterSpec",
    "JumpRollout",
    "JumpTechnique",
    "LayerAttentionMixer",
    "LinearBlockPredictor",
    "LinearPredictor",
    "MLPPredictor",
    "Predictor",
    "PredictorMemory",
    "SequencePredictor",
    "SourceMixer",
    "StepFn",
    "hybrid_log_probs",
    "jump_acceptance",
    "jump_rollout",
    "shared_parameters",
]
