"""SliceGPT (Ashkboos et al., ICLR 2024): delete rows and columns of the weights after rotating the residual stream
onto its principal components, which gives a dense model of a smaller hidden width."""

from .config import ROTATIONS, SliceGPTConfig
from .pca import CovarianceAccumulator, principal_basis
from .rotate import RotationStore, rotate_global, rotate_per_block
from .shortcut import ATTN_SHORTCUT, MLP_SHORTCUT, install_shortcuts, set_shortcuts, shortcut_widths
from .technique import SliceGPTTechnique

__all__ = [
    "ATTN_SHORTCUT",
    "MLP_SHORTCUT",
    "ROTATIONS",
    "CovarianceAccumulator",
    "RotationStore",
    "SliceGPTConfig",
    "SliceGPTTechnique",
    "install_shortcuts",
    "principal_basis",
    "rotate_global",
    "rotate_per_block",
    "set_shortcuts",
    "shortcut_widths",
]
