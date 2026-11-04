"""Configuration of the rotation technique (``rotate``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...validation import check_globs, check_layers, check_non_negative, check_registered

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["RotateConfig"]


@dataclass(frozen=True)
class RotateConfig(TechniqueConfig):
    """Rotate the inputs of linear maps by Hadamard matrices so that a following ``quant`` sees fewer outliers.

    The rotations of QuaRot (Ashkboos et al., NeurIPS 2024) that a model with any normalization layout admits:

    * the value rotation (QuaRot's R2): an orthogonal ``head_dim x head_dim`` matrix ``R`` fused into the rows of
      ``v_proj`` (per KV head) and the columns of ``o_proj`` (per query head). Offline and exact: the attention
      output of every head is rotated by ``R`` and ``o_proj`` undoes it.
    * online input rotations (QuaRot's R4 on ``mlp.down_proj``): the weight becomes ``W H`` and a forward pre-hook
      turns the input ``x`` into ``x H``, with ``H`` the block-diagonal normalized Sylvester-Hadamard matrix of the
      largest power-of-two block that divides the input width. vLLM runs the same transform with its
      ``hadacore_transform`` kernel.

    The residual-stream rotation R1 is not applied: it needs every norm in front of a linear map to be fused into it,
    which the sandwich norms of Gemma 2/3/4 rule out.

    Args:
        method: Registered rotation method (:data:`~.methods.ROTATIONS`): ``"hadamard"`` (QuaRot: random-sign
            Hadamard value rotations, Sylvester-Hadamard online rotations).
        online: Globs over linear-map names inside a layer whose inputs are rotated online; empty for none.
        values: Apply the value rotation.
        layers: 1-indexed layers to rotate; empty means all.
        block: Largest online Hadamard block, a power of two; ``0`` for the largest that divides the width.
        seed: Seed of the random signs of the value rotations; ``None`` for the plain Sylvester-Hadamard matrix.
        kernel: Kernel of the online transform on the Hugging Face path (a name in
            :data:`~.hadamard.HADAMARD_KERNELS`, or ``"auto"``).

    Raises:
        ConfigError: A setting is invalid, or the config rotates nothing.
    """

    technique: ClassVar[str] = "rotate"

    method: str = "hadamard"
    online: tuple[str, ...] = ("mlp.down_proj",)
    values: bool = True
    layers: tuple[int, ...] = ()
    block: int = 0
    seed: int | None = 0
    kernel: str = "auto"

    def __post_init__(self) -> None:
        from .hadamard import HADAMARD_KERNELS, is_power_of_two
        from .methods import ROTATIONS

        check_registered("rotation method", self.method, ROTATIONS)
        check_globs("online", self.online, allow_empty=True)
        check_layers(self.layers)
        check_non_negative("block", self.block)
        if self.block and not is_power_of_two(self.block):
            raise ConfigError(f"block must be 0 or a power of two, got {self.block}")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise ConfigError(f"seed must be an int or None, got {self.seed!r}")
        if self.kernel != "auto":
            check_registered("Hadamard kernel", self.kernel, HADAMARD_KERNELS)
        if not self.online and not self.values:
            raise ConfigError("rotate with online=() and values=False rotates nothing")

    def validate(self, adapter: ArchAdapter) -> None:
        check_layers(self.layers, adapter)
