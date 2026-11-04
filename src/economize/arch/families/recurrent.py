"""Families economize rejects with a clear error instead of running them wrongly.

Pure state-space and RNN models have no attention layer to skip, draft into or verify with masks. Hybrid models
whose recurrent layers keep states other than the conv window and recurrent state of Qwen3.5 and Nemotron-H are not
supported yet: they need a cache class registered in :data:`~economize.runtime.CACHES` first.
"""

from __future__ import annotations

from torch import nn

from ...errors import UnsupportedModelError
from ..base import register_adapter
from ..standard import StandardDecoderAdapter

__all__ = ["UnsupportedFamilyAdapter"]

RECURRENT_ONLY = ("mamba", "mamba2", "falcon_mamba", "rwkv", "xlstm")
MAMBA_HYBRIDS = (
    "jamba", "bamba", "falcon_h1", "granitemoehybrid", "lfm2", "lfm2_moe", "zamba", "zamba2",
)


@register_adapter(*RECURRENT_ONLY, *MAMBA_HYBRIDS)
class UnsupportedFamilyAdapter(StandardDecoderAdapter):
    """Raises :class:`UnsupportedModelError` on construction."""

    def __init__(self, model: nn.Module) -> None:
        model_type = getattr(model.config, "model_type", type(model).__name__)
        if model_type in RECURRENT_ONLY:
            reason = "it is a recurrent model without attention layers"
        else:
            reason = "its Mamba layers keep states that economize's hybrid cache does not implement yet"
        raise UnsupportedModelError(f"economize cannot run {model_type!r}: {reason}")
