"""Configuration of LittleBit binary low-rank compression (``littlebit``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ...core.config import TechniqueConfig
from ...validation import check_globs, check_layers, check_number, check_positive, check_registered

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["LittleBitConfig"]


@dataclass(frozen=True)
class LittleBitConfig(TechniqueConfig):
    """Replace linear maps of the decoder layers by LittleBit binary factorizations (Lee et al., NeurIPS 2025).

    Every selected map becomes ``sum_p diag(h_p) sign(U_p) diag(l_p) sign(V_p) diag(g_p)`` stored with bit-packed
    signs and 16-bit scales (:class:`~economize.techniques.littlebit.LittleBitLinear`); the embedding and the LM
    head stay in their dtype, as in the paper.

    Args:
        method: Registered method: ``"dual_svid"`` (the paper's data-free initialization, the PTQ row) or
            ``"blockwise"`` (Dual-SVID, then quantization-aware training of each decoder layer's factors against
            the original layer's outputs on calibration data).
        bpw: Target bits per weight of every map, signs and scales included (the paper's 1.0, 0.55, 0.3, 0.1).
        residual: Two paths with residual compensation (the paper's default) instead of one.
        targets: Globs over linear-module names inside a layer.
        exclude: Globs removed from ``targets``.
        layers: 1-indexed layers to compress; empty means all.
        rank_multiple: Round every rank down to a multiple of this (32 fills whole sign words).
        backend: Registered :data:`~economize.techniques.littlebit.LITTLEBIT_BACKENDS` entry for the forward.
        factors: Path of a file written by :func:`~economize.techniques.littlebit.save_factors`. When set, the
            stored factors are loaded instead of running ``method`` (no calibration needed).
        epochs: ``blockwise``: passes over the calibration batches per layer.
        lr: ``blockwise``: Adam learning rate of the latent factors and scales.
        accumulate: ``blockwise``: calibration batches per optimizer step.
        smooth_k: ``blockwise``: slope of the SmoothSign surrogate gradient (the paper's 100).
        sequential: ``blockwise``: train each layer on the inputs that the already compressed layers produce
            (the targets stay the original layer's outputs on those inputs) instead of the original model's.
        offload: Keep calibration activations on the CPU between layers.
    """

    technique: ClassVar[str] = "littlebit"

    method: str = "dual_svid"
    bpw: float = 1.0
    residual: bool = True
    targets: tuple[str, ...] = ("*",)
    exclude: tuple[str, ...] = ()
    layers: tuple[int, ...] = ()
    rank_multiple: int = 32
    backend: str = "triton"
    factors: str = ""
    epochs: int = 2
    lr: float = 1e-4
    accumulate: int = 1
    smooth_k: float = 100.0
    sequential: bool = False
    offload: bool = False

    def __post_init__(self) -> None:
        from .backends import LITTLEBIT_BACKENDS
        from .methods import LITTLEBIT_METHODS

        check_registered("LittleBit method", self.method, LITTLEBIT_METHODS)
        check_registered("LittleBit backend", self.backend, LITTLEBIT_BACKENDS)
        check_number("bpw", self.bpw, low=0, high=16, closed_low=False, closed_high=False)
        check_globs("targets", self.targets)
        check_globs("exclude", self.exclude, allow_empty=True)
        check_layers(self.layers)
        check_positive("rank_multiple", self.rank_multiple)
        check_positive("epochs", self.epochs)
        check_number("lr", self.lr, low=0, closed_low=False)
        check_positive("accumulate", self.accumulate)
        check_number("smooth_k", self.smooth_k, low=0, closed_low=False)

    @property
    def paths(self) -> int:
        """Number of binary paths (2 with residual compensation)."""
        return 2 if self.residual else 1

    def validate(self, adapter: ArchAdapter) -> None:
        check_layers(self.layers, adapter)
