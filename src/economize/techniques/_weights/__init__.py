"""Helpers shared by the techniques that rewrite base weights once at attach time (``prune``, ``quant``,
``lowrank``).

* :mod:`.hessian`: the damped inverse Hessian of the Optimal Brain Surgeon updates (SparseGPT, GPTQ).
* :mod:`.layerwise`: the layer-by-layer rewrite driver (:func:`rewrite_layerwise`) and its selection helpers
  (:func:`resolve_layers`, :func:`target_linears`, :func:`observe_step`, :func:`require_calibration`).
* :mod:`.obs`: the blocked Optimal Brain Surgeon column sweep of SparseGPT and GPTQ (:func:`obs_sweep`).
* :mod:`.snapshot`: :class:`ParameterSnapshot`, which makes an in-place transform all or nothing.
* :mod:`.technique`: :class:`WeightTransform`, the base of the one-shot weight techniques, and
  :class:`CalibratedWeightTransform`, its attach-with-calibration-data template (``prune``, ``quant``, ``lowrank``).
* :mod:`.transaction`: :class:`WeightTransaction` (snapshots plus module swaps, undone together) and
  :func:`replace_module`.

The config field checks these techniques use (``check_layers``, ``check_globs``, ...) live in
:mod:`economize.validation` and are re-exported here.
"""

from __future__ import annotations

from ...validation import check_choice, check_globs, check_layers, check_options, check_positive, check_unit_interval
from .hessian import damped_inverse_cholesky, fix_dead_features, inverse_hessian_cholesky
from .layerwise import (
    LayerRewrite,
    TargetSelection,
    observe_step,
    require_calibration,
    resolve_layers,
    rewrite_layerwise,
    target_linears,
)
from .obs import ColumnRule, ObsBlock, obs_sweep
from .snapshot import ParameterSnapshot
from .technique import CalibratedWeightTransform, WeightTransform
from .transaction import WeightTransaction, replace_module

__all__ = [
    "CalibratedWeightTransform",
    "ColumnRule",
    "LayerRewrite",
    "ObsBlock",
    "ParameterSnapshot",
    "TargetSelection",
    "WeightTransaction",
    "WeightTransform",
    "check_choice",
    "check_globs",
    "check_layers",
    "check_options",
    "check_positive",
    "check_unit_interval",
    "damped_inverse_cholesky",
    "fix_dead_features",
    "inverse_hessian_cholesky",
    "obs_sweep",
    "observe_step",
    "replace_module",
    "require_calibration",
    "resolve_layers",
    "rewrite_layerwise",
    "target_linears",
]
