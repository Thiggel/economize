"""The registered ``slicegpt`` technique (a thin wrapper over :mod:`.rotate`)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import torch
from torch import Tensor

from ...core.config import TechniqueConfig
from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import Capability, Technique, register_technique
from .._structured.layout import layout_for
from .._weights.technique import CalibratedWeightTransform
from .config import SliceGPTConfig
from .rotate import RotationStore, rotate_global, rotate_per_block

__all__ = ["SliceGPTTechnique"]

#: Families whose techniques run a subset of the layers or read the residual stream between layers, which the
#: per-block rotation expresses in a different basis (and width) at every point.
_PER_BLOCK_EXCLUDES = (Family.DRAFTING, Family.DYNAMIC_DEPTH)


def _parameters(model: Any) -> int:
    return sum(p.numel() for p in model.parameters())


@register_technique
class SliceGPTTechnique(CalibratedWeightTransform):
    """SliceGPT: rotate the residual stream onto its principal components and delete the trailing ones.

    The fitted bases are buffers (``rotations``, ``head_rotation``, ``fitted``), so a saved technique replays the
    same rotation on the base model without calibration data. ``rotations`` holds ``2L + 1`` matrices of shape
    ``[D, d]`` for ``rotation="per_block"`` (about 3.3 GB in float32 for a 7B Llama at 25% sparsity) and one for
    ``rotation="global"``.
    """

    name: ClassVar[str] = SliceGPTConfig.technique
    config_class: ClassVar[type[SliceGPTConfig]] = SliceGPTConfig
    info = TechniqueInfo(
        family=Family.WEIGHT_COMPRESSION,
        saves=Saving.WEIGHT_MEMORY | Saving.COMPUTE | Saving.LATENCY,
        lossless=False,
    )
    capabilities: ClassVar[Capability] = Capability.WEIGHTS
    #: After layer removal (``depth_prune``, 5), before every transform of single maps (``prune``, ``quant``, ...).
    order: ClassVar[int] = 6
    config: SliceGPTConfig
    rotations: Tensor
    head_rotation: Tensor
    fitted: Tensor

    def __init__(self, config: SliceGPTConfig, adapter: Any) -> None:
        super().__init__(config, adapter)
        hidden, width = adapter.hidden_size, config.width(adapter.hidden_size)
        points = 2 * adapter.num_layers + 1 if config.rotation == "per_block" else 1
        head = hidden if config.rotation == "per_block" and not config.slices_head else 0
        self.register_buffer("rotations", torch.zeros(points, hidden, width))
        self.register_buffer("head_rotation", torch.zeros(head, head))
        self.register_buffer("fitted", torch.tensor(False))

    def needs_calibration(self) -> bool:
        """Calibration data is read unless the bases were loaded from a checkpoint."""
        return not bool(self.fitted)

    def transform(self, adapter: Any, batches: Sequence[Tensor] | None) -> dict[str, Any]:
        config, layout = self.config, layout_for(adapter, self.name, residual=True)
        hidden, width, before = adapter.hidden_size, config.width(adapter.hidden_size), _parameters(adapter.model)
        store = RotationStore(self.rotations, self.head_rotation, fitting=not bool(self.fitted))
        if config.rotation == "per_block":
            rotate_per_block(
                adapter, layout, store, width, slice_head=config.slices_head, batches=batches, offload=config.offload
            )
        else:
            rotate_global(adapter, layout, store, batches=batches)
        self.fitted.fill_(True)
        after = _parameters(adapter.model)
        return {
            "rotation": config.rotation,
            "hidden_size": hidden,
            "width": width,
            "sliced_head": config.slices_head,
            "parameters": after,
            "removed_parameters": before - after,
        }

    @classmethod
    def conflict(
        cls, config: TechniqueConfig | None, other: type[Technique], other_config: TechniqueConfig | None
    ) -> str | None:
        """The per-block rotation changes the basis and width of the residual stream from point to point, so layer
        skipping and drafters that read the base model's hidden states do not compose with it (the global rotation
        does)."""
        if (isinstance(config, SliceGPTConfig) and config.rotation == "per_block"
                and other.info.family in _PER_BLOCK_EXCLUDES):
            return (
                f"slicegpt(rotation='per_block') gives every residual point its own basis, which {other.name!r} "
                f"({other.info.family}) cannot read across layers; use slicegpt(rotation='global')"
            )
        return super().conflict(config, other, other_config)
