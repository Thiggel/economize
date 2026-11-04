"""Configuration of low-rank weight compression (``lowrank``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ...core.config import TechniqueConfig
from ...validation import check_globs, check_layers, check_positive, check_registered, check_unit_interval

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["LowRankConfig"]


@dataclass(frozen=True)
class LowRankConfig(TechniqueConfig):
    """Replace linear maps of the decoder layers by two low-rank factors.

    Args:
        method: Registered factorization: ``"svd_llm"`` (default, needs calibration data) or ``"svd"``.
        compression: Fraction of each matrix's parameters removed (SVD-LLM's "compression ratio"; 0.2 keeps 80%).
        targets: Globs over linear-module names inside a layer.
        exclude: Globs removed from ``targets``.
        layers: 1-indexed layers to compress; empty means all.
        rank_multiple: Round every rank down to a multiple of this (e.g. 128 so that a following ``quant`` with
            ``group_size=128`` can quantize the ``up`` factor, or 8/16 for tensor-core friendly shapes).
        sequential: Profile each layer on the compressed model's activations instead of the original model's (the
            paper profiles the original model).
        offload: Keep calibration activations on the CPU between layers.
    """

    technique: ClassVar[str] = "lowrank"

    method: str = "svd_llm"
    compression: float = 0.2
    targets: tuple[str, ...] = ("*",)
    exclude: tuple[str, ...] = ()
    layers: tuple[int, ...] = ()
    rank_multiple: int = 1
    sequential: bool = False
    offload: bool = False

    def __post_init__(self) -> None:
        from .factorize import FACTORIZERS

        check_registered("low-rank method", self.method, FACTORIZERS)
        check_unit_interval("compression", self.compression, closed_low=False)
        check_globs("targets", self.targets)
        check_globs("exclude", self.exclude, allow_empty=True)
        check_layers(self.layers)
        check_positive("rank_multiple", self.rank_multiple)

    def validate(self, adapter: ArchAdapter) -> None:
        check_layers(self.layers, adapter)
