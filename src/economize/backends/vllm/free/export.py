"""The ``"vllm/free"`` exporter: FREE's shallow model as an early-exit drafter (a jump drafter folder with
``m = u = 0`` and ``r = L_S``)."""

from __future__ import annotations

from ...exporters import register_exporter
from ..early_exit.export import EarlyExitVllmExporter

__all__ = ["FreeVllmExporter"]


@register_exporter("vllm", "free")
class FreeVllmExporter(EarlyExitVllmExporter):
    """Writes the ``free`` technique as :class:`EarlyExitVllmExporter` writes ``early_exit``: the target's own layers
    ``1..L_S`` (FREE's shallow model is always frozen), then the readout adapter (the identity for
    ``head="shared"``), the final norm and the LM head."""
