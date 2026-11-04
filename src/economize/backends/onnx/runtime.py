"""The ONNX Runtime export: its technique adapters and the support level of every technique.

The export is one ONNX graph of the plain architecture with KV-cache inputs. What it realizes of economize's
techniques:

* weights: ``int2`` / ``int4`` / ``int8`` codes in contiguous groups of a power of two (at least 16) are stored as
  ONNX Runtime's ``com.microsoft.MatMulNBits`` (exact: float scales, integer zero points); other codes and the
  ``fake`` backend are written dequantized; pruned, low-rank and LittleBit weights are written dense, and a
  depth-pruned model is a plain model;
* everything else is refused with the reason: ONNX Runtime runs one graph per call (no speculative decoding loop,
  no drafter, no dynamic depth) and its cache is the graph's own float tensors.

A technique without an adapter takes its family's refusal (:data:`ONNX`'s ``defaults``). Packages add adapters
through the entry-point group ``economize.onnx_techniques`` or with ``@ONNX.register``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.config import TechniqueConfig
from ...registry import Registry
from ..export_adapters import QuantExport, register_weight_adapters
from ..runtime_export import ExportRuntime, ExportTechnique
from ..vllm.serving.base import Role
from .graph import nbits_reason

if TYPE_CHECKING:
    from ...techniques.quant.grid import QuantizedWeight

__all__ = ["ONNX", "ONNX_TECHNIQUES", "OnnxQuant"]

#: The ONNX Runtime adapters by technique name (the :attr:`ExportRuntime.adapters` of :data:`ONNX`).
ONNX_TECHNIQUES: Registry[type[ExportTechnique]] = Registry("ONNX Runtime technique adapter",
                                                    "economize.onnx_techniques")

_NO_LOOP = "ONNX Runtime runs one graph per call; the export has no speculative decoding loop"

ONNX = ExportRuntime(
    name="onnx",
    title="ONNX Runtime",
    adapters=ONNX_TECHNIQUES,
    defaults={
        "dynamic-depth": (Role.FORWARD, "the ONNX graph runs every layer for every token (no data-dependent depth)"),
        "drafting": (Role.DRAFTER, _NO_LOOP),
        "draft-compression": (Role.DRAFT_OPTION, _NO_LOOP),
        "KV-quantization": (Role.KV_CACHE, "the exported cache is the graph's float tensors (no quantized cache)"),
        "KV-eviction": (Role.KV_EVICTION, "the exported cache keeps every token (no eviction in the graph)"),
        "activation-sparsity": (Role.FORWARD, "ONNX Runtime has no activation-sparse kernels"),
    },
    fallback="the ONNX export has no path for this technique",
)


class OnnxQuant(QuantExport):
    """``quant``: ``int2`` / ``int4`` / ``int8`` codes are stored as ``com.microsoft.MatMulNBits``.

    The codes, the float32 scales and the integer zero points are stored as they are, so ONNX Runtime computes
    economize's dequantized map. Weights with ``act_order`` groups, group sizes that are not a power of two of at
    least 16, and other bit widths are written dequantized (the export then reports ``DENSIFIED``).
    """

    reason = ("int2 / int4 / int8 codes of the compressed-tensors, marlin, machete and torchao backends are stored "
              "as ONNX Runtime MatMulNBits (exact); backend='fake' and other formats are written dequantized")
    fp8_activations_reason = (
        "ONNX Runtime's MatMulNBits is weight-only (no fp8 activations); export the weight-only model"
    )

    def stores(self, config: TechniqueConfig) -> bool:
        return getattr(config, "backend", "fake") != "fake" and getattr(config, "format", "").startswith("int")

    def stores_weight(self, name: str, weight: QuantizedWeight) -> str | None:
        return nbits_reason(weight)


register_weight_adapters(ONNX, OnnxQuant)

#: Techniques whose family default would be too vague, with the reason ONNX Runtime cannot run them.
_REFUSALS: dict[str, tuple[Role, str]] = {
    "draft_model": (Role.DRAFTER, f"{_NO_LOOP}; export the draft LM as its own ONNX graph and verify in your loop"),
    "placed_draft_model": (Role.DRAFTER, _NO_LOOP),
    "lookup": (Role.DRAFTER, f"{_NO_LOOP} (prompt lookup needs a verify loop around the graph)"),
    "kv_fp8": (Role.KV_CACHE, "the exported cache is the graph's float tensors (no fp8 cache)"),
    "teal": (Role.FORWARD, "ONNX Runtime has no activation-sparse kernels"),
}

for _name, (_role, _reason) in _REFUSALS.items():
    ONNX.refuse(_name, _role, _reason)
