"""Rotations that remove outliers before weight quantization: QuaRot's value (R2) and online (R4) Hadamard
rotations, fused into the weights."""

from .config import RotateConfig
from .hadamard import HADAMARD_KERNELS, HadamardKernel, block_hadamard, hadamard_block, hadamard_transform, sylvester
from .methods import ROTATIONS, HadamardRotation, RotationMethod
from .model import ROTATED_ATTR, InputHadamard, input_hadamard_of, rotate_model
from .technique import RotateTechnique

__all__ = [
    "HADAMARD_KERNELS",
    "ROTATED_ATTR",
    "ROTATIONS",
    "HadamardKernel",
    "HadamardRotation",
    "InputHadamard",
    "RotateConfig",
    "RotateTechnique",
    "RotationMethod",
    "block_hadamard",
    "hadamard_block",
    "hadamard_transform",
    "input_hadamard_of",
    "rotate_model",
    "sylvester",
]
