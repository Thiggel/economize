"""Rotation methods: how the value rotation of each layer is chosen.

A method is registered in :data:`ROTATIONS` and returns one orthogonal ``head_dim x head_dim`` matrix per layer.
QuaRot's random Hadamard matrices need no data; a learned method (SpinQuant's Cayley-optimized rotations) would set
``needs_calibration`` and fit its matrices on the calibration batches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import torch
from torch import Tensor

from ...registry import Registry
from .hadamard import sylvester

__all__ = ["ROTATIONS", "HadamardRotation", "RotationMethod", "random_signs"]


class RotationMethod(ABC):
    """One way of choosing the value rotation of a layer."""

    #: Whether the method reads calibration data.
    needs_calibration: ClassVar[bool] = False

    @abstractmethod
    def value_rotation(self, head_dim: int, layer: int, seed: int | None) -> Tensor:
        """The orthogonal ``head_dim x head_dim`` value rotation ``R`` of layer ``layer`` (1-indexed), in fp64.

        Args:
            head_dim: The layer's head width.
            layer: The layer's 1-indexed index.
            seed: The config's seed, ``None`` for a deterministic matrix.

        Returns:
            ``R``; the values of every head become ``v @ R``.
        """


def random_signs(n: int, seed: int) -> Tensor:
    """``n`` random signs (fp64, ``+-1``) drawn from a CPU generator seeded with ``seed``."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (n,), generator=generator).to(torch.float64) * 2 - 1


ROTATIONS: Registry[type[RotationMethod]] = Registry("rotation method", "economize.rotation_methods")


@ROTATIONS.register("hadamard")
class HadamardRotation(RotationMethod):
    """QuaRot's randomized Hadamard matrix ``diag(s) H`` (signs ``s`` drawn per layer from ``seed + layer``), or the
    plain Sylvester-Hadamard matrix ``H`` when the seed is ``None``. Needs a power-of-two head width."""

    def value_rotation(self, head_dim: int, layer: int, seed: int | None) -> Tensor:
        matrix = sylvester(head_dim, dtype=torch.float64)
        if seed is None:
            return matrix
        return random_signs(head_dim, seed + layer)[:, None] * matrix
