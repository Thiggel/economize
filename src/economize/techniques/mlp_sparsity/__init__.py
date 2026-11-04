"""Gated-MLP channel sparsity: CATS thresholds (Lee et al., COLM 2024) and Deja Vu predictors (Liu et al., ICML 2023).

Both techniques skip FFN channels of a gated MLP ``down(act(gate(x)) * up(x))`` per token. CATS drops the channels
whose gate activation is below a calibrated threshold; Deja Vu computes the channels a trained low-rank predictor
scores highest. The ``dense`` backend masks around the unchanged dense maps (the numerical reference, declared as
a simulation); the ``triton`` backend reads only the kept weight rows of the up (and gate) maps
(:func:`.kernel.row_masked_gemv`) and the kept weight columns of the down map (TEAL's column-sparse GEMV) for calls
of up to 16 rows on 16-bit CUDA weights.
"""

from .backends import (
    MLP_SPARSITY_BACKENDS,
    DenseMlpBackend,
    DenseOps,
    MlpOps,
    MlpSparsityBackend,
    TritonMlpBackend,
    TritonOps,
)
from .cats import CatsTechnique, GateMagnitudeObserver, calibrate_cats, cats_mlp
from .config import PREDICTOR_DTYPES, PREDICTOR_INPUTS, CatsConfig, DejaVuConfig, MlpSparsityConfig
from .dejavu import (
    DejaVuObjective,
    DejaVuTechnique,
    dejavu_labels,
    dejavu_mlp,
    fit_predictor,
    kept_channels,
    predictor_recall,
)
from .kernel import RowGemvConfig, default_row_config, row_masked_gemv
from .mlp import GatedMlp, MlpPatch, MlpRule, call_mode, dense_first_half, ffn_width, gated_mlps, masked, topk_mask

__all__ = [
    "MLP_SPARSITY_BACKENDS",
    "PREDICTOR_DTYPES",
    "PREDICTOR_INPUTS",
    "CatsConfig",
    "CatsTechnique",
    "DejaVuConfig",
    "DejaVuObjective",
    "DejaVuTechnique",
    "DenseMlpBackend",
    "DenseOps",
    "GateMagnitudeObserver",
    "GatedMlp",
    "MlpOps",
    "MlpPatch",
    "MlpRule",
    "MlpSparsityBackend",
    "MlpSparsityConfig",
    "RowGemvConfig",
    "TritonMlpBackend",
    "TritonOps",
    "calibrate_cats",
    "call_mode",
    "cats_mlp",
    "default_row_config",
    "dejavu_labels",
    "dejavu_mlp",
    "dense_first_half",
    "ffn_width",
    "fit_predictor",
    "gated_mlps",
    "kept_channels",
    "masked",
    "predictor_recall",
    "row_masked_gemv",
    "topk_mask",
]
