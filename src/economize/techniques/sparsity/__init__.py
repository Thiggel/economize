"""Activation sparsity: TEAL magnitude thresholds on the inputs of the linear maps (Liu et al., ICLR 2025)."""

from .backends import TEAL_BACKENDS, DenseBackend, Installed, TealBackend
from .config import TEAL_ALLOCATIONS, TealConfig
from .model import SparsityMeter, TealCalibration, calibrate_teal, install_teal, measure_sparsity, teal_modules
from .sparsify import (
    PREFILL_MODES,
    GreedyStep,
    PhasePolicy,
    block_sparsity,
    greedy_allocation,
    greedy_error,
    layer_tokens,
    magnitude_quantile,
    sparsify,
    sparsify_tokens,
    track_layer_tokens,
    uniform_allocation,
)
from .technique import TealTechnique
from .triton_autotune import AutotuneTritonBackend, Tuning, clear_tuning_cache
from .triton_backend import TritonBackend

__all__ = [
    "PREFILL_MODES",
    "TEAL_ALLOCATIONS",
    "TEAL_BACKENDS",
    "AutotuneTritonBackend",
    "DenseBackend",
    "GreedyStep",
    "Installed",
    "PhasePolicy",
    "SparsityMeter",
    "TealBackend",
    "TealCalibration",
    "TealConfig",
    "TealTechnique",
    "TritonBackend",
    "Tuning",
    "block_sparsity",
    "calibrate_teal",
    "clear_tuning_cache",
    "greedy_allocation",
    "greedy_error",
    "install_teal",
    "layer_tokens",
    "magnitude_quantile",
    "measure_sparsity",
    "sparsify",
    "sparsify_tokens",
    "teal_modules",
    "track_layer_tokens",
    "uniform_allocation",
]
