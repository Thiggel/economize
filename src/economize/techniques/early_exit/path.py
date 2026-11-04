"""The adapted layers of a self-drafting path: LoRA updates active only on the path, or trainable layer copies.

Self-drafters (early exit, layer skipping) run some of the model's own layers while drafting. Their layers can be
adapted for that path without touching the exact path: LoRA updates hook into the layers' linear maps and add their
update only inside :meth:`DraftPath.active`; trainable copies replace the layers through the runner's ``overrides``.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING

import torch
from torch import nn

from ...errors import ConfigError
from ...modules import InModelDtype, LayerLoRA, trainable_copy
from ...modules.sharding import check_adaptable_split

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...runtime import LayerCall

__all__ = ["DraftPath"]


class DraftPath(nn.Module):
    """Adaptation of the decoder layers ``layers`` on a drafting path.

    Args:
        adapter: The model's architecture adapter.
        layers: The adapted layers (1-indexed).
        adaptation: ``"frozen"`` (no parameters), ``"lora"`` or ``"full"``.
        lora_rank: Rank of the LoRA updates.
        lora_alpha: LoRA scaling numerator.
        copy_dtype: Storage dtype of the trainable copies (``adaptation="full"``).

    Parameters, on the model's device: ``lora.layers.<l>.<map>.lora_A|B`` (float32) or ``blocks.<l>.*``
    (``copy_dtype``).
    """

    def __init__(
        self, adapter: ArchAdapter, layers: Iterable[int], adaptation: str, lora_rank: int = 16,
        lora_alpha: float = 32.0, copy_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.adapter = adapter  # plain attribute: the base model is not part of the path's state
        self.layers = tuple(sorted(set(layers)))
        self.lora: LayerLoRA | None = None
        self.blocks: nn.ModuleDict | None = None
        check_adaptable_split(adapter.model, adapter.tensor_parallel_size, adaptation)
        if adaptation == "lora":
            self.lora = LayerLoRA(self._targets(), lora_rank, lora_alpha).to(adapter.device, torch.float32)
        elif adaptation == "full":
            copies = {layer: trainable_copy(adapter.layer(layer), copy_dtype) for layer in self.layers}
            self.blocks = nn.ModuleDict({str(layer): block.to(adapter.device) for layer, block in copies.items()})
        elif adaptation != "frozen":
            raise ConfigError(f"unknown adaptation {adaptation!r}")

    def _targets(self) -> dict[int, dict[str, nn.Module]]:
        adapter = self.adapter
        return {layer: adapter.lora_targets(adapter.layer(layer)) for layer in self.layers}

    def bind(self) -> None:
        """Hook the LoRA updates into the adapted layers as they are now (after any weight transform)."""
        if self.lora is not None:
            self.lora.install(self._targets())

    def refresh(self) -> None:
        """Re-hook the LoRA updates when the layers' linear maps were replaced since :meth:`bind`."""
        if self.lora is not None and self.lora.installed:
            self.bind()

    def unbind(self) -> None:
        """Take the LoRA hooks off the model."""
        if self.lora is not None:
            self.lora.remove()

    def active(self) -> AbstractContextManager[None]:
        """Context in which the adapted layers behave as on the drafting path (LoRA updates active)."""
        return self.lora.active() if self.lora is not None else nullcontext()

    def copies(self) -> dict[int, nn.Module]:
        """The trainable copies by layer (``adaptation="full"``), else none."""
        return {} if self.blocks is None else {int(layer): block for layer, block in self.blocks.items()}

    def overrides(self) -> dict[int, LayerCall]:
        """Replacements of model layers on the drafting path (the trainable copies with ``adaptation="full"``)."""
        if self.blocks is None:
            return {}
        dtype = self.adapter.dtype
        return {int(layer): InModelDtype(block, dtype) for layer, block in self.blocks.items()}

    def backbone_parameters(self) -> dict[str, list[nn.Parameter]]:
        """The path's parameters by kind (``"lora"``, ``"blocks"``), for optimizer groups."""
        out = {}
        if self.lora is not None:
            out["lora"] = list(self.lora.parameters())
        if self.blocks is not None:
            out["blocks"] = list(self.blocks.parameters())
        return out
