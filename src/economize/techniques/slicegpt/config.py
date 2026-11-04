"""Configuration of SliceGPT (``slicegpt``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...validation import check_choice, check_positive, check_unit_interval

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["ROTATIONS", "SliceGPTConfig"]

#: ``"per_block"``: the paper's rotation, one PCA basis per residual point with a shortcut matrix on every residual
#: connection; ``"global"``: one basis for the whole residual stream and no shortcuts, a plain smaller checkpoint.
ROTATIONS = ("per_block", "global")


@dataclass(frozen=True)
class SliceGPTConfig(TechniqueConfig):
    """Rotate the residual stream onto its principal components and delete the trailing ones (SliceGPT).

    Args:
        sparsity: Fraction of the hidden width removed; the kept width is ``(1 - sparsity) * hidden_size`` rounded
            down to a multiple of ``round_to`` (as the official ``run_slicegpt.py``).
        rotation: ``"per_block"`` (the paper, Ashkboos et al. 2024): a PCA basis at the input of every block, fitted
            on the outputs of the already sliced layers, with a ``d x d`` shortcut matrix on each residual
            connection; the result runs in transformers only. ``"global"`` (an economize extension): one basis of
            the residual stream of the dense model for every point, so no shortcut is needed and the sliced model
            is a plain checkpoint of the same architecture with a smaller ``hidden_size``, served by vLLM and
            SGLang. ``"global"`` loses more quality at equal sparsity.
        slice_head: Also slice the input of the LM head (``None``: as the official code, ``False`` for
            ``"per_block"``; ``"global"`` always slices it).
        round_to: The kept width is a multiple of this.
        offload: Keep calibration activations on the CPU between layers.
    """

    technique: ClassVar[str] = "slicegpt"

    sparsity: float = 0.25
    rotation: str = "per_block"
    slice_head: bool | None = None
    round_to: int = 8
    offload: bool = False

    def __post_init__(self) -> None:
        check_unit_interval("sparsity", self.sparsity)
        check_choice("rotation", self.rotation, ROTATIONS)
        check_positive("round_to", self.round_to)
        if self.rotation == "global" and self.slice_head is False:
            raise ConfigError("rotation='global' slices the head input; slice_head=False needs rotation='per_block'")

    @property
    def slices_head(self) -> bool:
        """Whether the LM head reads the sliced residual stream."""
        return self.rotation == "global" or bool(self.slice_head)

    def width(self, hidden_size: int) -> int:
        """The kept width of a residual stream of ``hidden_size`` channels."""
        kept = int((1 - self.sparsity) * hidden_size)
        return kept - kept % self.round_to

    def validate(self, adapter: ArchAdapter) -> None:
        from .._structured.layout import layout_for

        layout_for(adapter, self.technique, residual=True)
        if adapter.tensor_parallel_size > 1:
            raise ConfigError("slicegpt rewrites the residual width and needs the whole model on one process")
        if self.width(adapter.hidden_size) < 1:
            raise ConfigError(
                f"sparsity={self.sparsity} with round_to={self.round_to} keeps no channel of {adapter.hidden_size}"
            )
