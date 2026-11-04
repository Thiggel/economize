"""Dynamic depth: per-token layer skipping and early exit on the exact path (GateSkip, Mixture-of-Depths,
FlexiDepth, SkipDecode, LoRA-Drop, CALM), and the dynamic-depth methods that decode through the drafting cycle
(FREE, DMTD, Hidden Transfer).

    import economize
    from economize.techniques.dynamic_depth import CalmConfig, GateSkipConfig

    model = economize.apply(hf_model, CalmConfig(confidence="softmax", threshold=0.9))
    result = model.decode(prompt, 64)
    model.techniques["calm"].cost_summary(result)  # layer applications that ran and that were skipped

    model = economize.apply(hf_model, GateSkipConfig(rule="budget", budget=0.8))
    economize.train.fit(model, data, objective="gateskip")  # trains the gates only; the base model stays frozen

Each technique gives the decoder a fresh :class:`~.base.DepthPolicy` per generation. The policy decides before every
layer which tokens of an exact pass run it; tokens that skip keep their hidden state and, except for SkipDecode
(whose schedule never needs them), get the K/V the layer computes from that state. Drafters are unaffected: they
draft with their own path and the exact passes that verify them follow the policy. All six techniques are lossy.

FREE (shallow exit with a synchronized parallel pass of the deep layers), DMTD (late layers only for the tokens
after a cycle's first, cyclical refilling) and Hidden Transfer (pseudo states of future tokens at intermediate
layers) skip layers for whole tokens and fill them later in one parallel pass, which is economize's drafting cycle:
they are drafters, FREE and DMTD decoded with every draft kept (lossy) and Hidden Transfer verified (lossless, the
paper's) or kept. Their ``cost_summary`` (:func:`~.drafted.drafted_cost_summary`) subtracts what the exact passes
recompute and the methods reuse.
A new technique subclasses :class:`~.base.DepthTechnique` and :class:`~.base.DepthPolicy` and registers itself
with :func:`economize.core.register_technique`; objectives register with :func:`economize.train.register_objective`.
"""

from __future__ import annotations

from . import objective as _objective  # noqa: F401  (registers the objectives)
from .base import (
    ALL_PHASES,
    GENERATION_PHASES,
    DepthMeter,
    DepthPolicy,
    DepthTechnique,
    LayeredDepthConfig,
    RoutedDepthPolicy,
    ScalarScore,
    ScoredDepthPolicy,
    gated_residual,
    layer_modules,
    module_at,
    real_tokens,
    runtime_supports_routed_kv,
    second_half,
    unit_rms,
    valid_mask,
)
from .calm import (
    CONFIDENCE_MEASURES,
    CalmConfig,
    CalmPolicy,
    CalmTechnique,
    ClassifierConfidence,
    ConfidenceMeasure,
    ExitClassifier,
    SoftmaxConfidence,
    StateConfidence,
    calibrate_threshold,
    hoeffding_bentkus_pvalue,
    ltt_threshold,
    token_mismatch,
)
from .dmtd import DmtdConfig, DmtdRollout, DmtdTechnique, EmbeddingPredictor, percentage_layers_per_token
from .drafted import drafted_cost_summary
from .flexidepth import (
    FLEXI_ROUTERS,
    BottleneckRouter,
    FlexiDepthConfig,
    FlexiDepthPolicy,
    FlexiDepthTechnique,
    FlexiRouter,
    SkipAdapter,
)
from .free import (
    KD_MAPPINGS,
    FreeConfig,
    FreeObjective,
    FreeTechnique,
    ShallowExitStop,
    beta_moments,
    bmm_threshold,
    free_calibration_pairs,
    kd_layer_map,
    monotone_layer_map,
    softmax_gap,
)
from .gateskip import GATESKIP_RULES, GateSkipConfig, GateSkipPolicy, GateSkipTechnique, ResidualGate, top_fraction
from .hidden_transfer import (
    TRAIN_CONTEXTS,
    HiddenTransferConfig,
    HiddenTransferDistill,
    HiddenTransferDrafter,
    HiddenTransferTechnique,
)
from .loradrop import AnchorStore, LoRADropConfig, LoRADropPolicy, LoRADropTechnique, LowRankUpdate, redundant_layers
from .mod import MOD_KV_MODES, MOD_ROUTER_INPUTS, MoDConfig, MoDPolicy, MoDTechnique, TokenRouter
from .objective import (
    CalmObjective,
    DepthLMObjective,
    FlexiDepthObjective,
    GateSkipObjective,
    LoRADropObjective,
    MoDObjective,
    student_pass,
    student_states,
)
from .skipdecode import SKIPDECODE_SIDES, SkipDecodeConfig, SkipDecodePolicy, SkipDecodeTechnique

__all__ = [
    "ALL_PHASES",
    "CONFIDENCE_MEASURES",
    "FLEXI_ROUTERS",
    "GATESKIP_RULES",
    "GENERATION_PHASES",
    "KD_MAPPINGS",
    "MOD_KV_MODES",
    "MOD_ROUTER_INPUTS",
    "SKIPDECODE_SIDES",
    "TRAIN_CONTEXTS",
    "AnchorStore",
    "BottleneckRouter",
    "CalmConfig",
    "CalmObjective",
    "CalmPolicy",
    "CalmTechnique",
    "ClassifierConfidence",
    "ConfidenceMeasure",
    "DepthLMObjective",
    "DepthMeter",
    "DepthPolicy",
    "DepthTechnique",
    "DmtdConfig",
    "DmtdRollout",
    "DmtdTechnique",
    "EmbeddingPredictor",
    "ExitClassifier",
    "FlexiDepthConfig",
    "FlexiDepthObjective",
    "FlexiDepthPolicy",
    "FlexiDepthTechnique",
    "FlexiRouter",
    "FreeConfig",
    "FreeObjective",
    "FreeTechnique",
    "GateSkipConfig",
    "GateSkipObjective",
    "GateSkipPolicy",
    "GateSkipTechnique",
    "HiddenTransferConfig",
    "HiddenTransferDistill",
    "HiddenTransferDrafter",
    "HiddenTransferTechnique",
    "LayeredDepthConfig",
    "LoRADropConfig",
    "LoRADropObjective",
    "LoRADropPolicy",
    "LoRADropTechnique",
    "LowRankUpdate",
    "MoDConfig",
    "MoDObjective",
    "MoDPolicy",
    "MoDTechnique",
    "ResidualGate",
    "RoutedDepthPolicy",
    "ScalarScore",
    "ScoredDepthPolicy",
    "ShallowExitStop",
    "SkipAdapter",
    "SkipDecodeConfig",
    "SkipDecodePolicy",
    "SkipDecodeTechnique",
    "SoftmaxConfidence",
    "StateConfidence",
    "TokenRouter",
    "beta_moments",
    "bmm_threshold",
    "calibrate_threshold",
    "drafted_cost_summary",
    "free_calibration_pairs",
    "gated_residual",
    "hoeffding_bentkus_pvalue",
    "kd_layer_map",
    "layer_modules",
    "ltt_threshold",
    "module_at",
    "monotone_layer_map",
    "percentage_layers_per_token",
    "real_tokens",
    "redundant_layers",
    "runtime_supports_routed_kv",
    "second_half",
    "softmax_gap",
    "student_pass",
    "student_states",
    "token_mismatch",
    "top_fraction",
    "unit_rms",
    "valid_mask",
]
