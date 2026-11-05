"""The ONNX Runtime export: its technique adapters and the support level of every technique.

The export is one ONNX graph of the plain architecture with KV-cache inputs. What it realizes of economize's
techniques:

* weights: ``int2`` / ``int4`` / ``int8`` codes in contiguous groups of a power of two (at least 16) are stored as
  ONNX Runtime's ``com.microsoft.MatMulNBits`` (exact: float scales, integer zero points); other layouts are
  refused and the explicit ``fake`` reference is written dequantized; low-rank maps keep their two factors (two
  ``MatMul`` nodes, :class:`OnnxLowRank`); a depth-pruned model is a plain model; pruning and LittleBit are refused
  (no sparse or binary-factor kernel), except the explicit masked-dense pruning reference
  ``prune(semi_structured=False)``, written dense;
* everything else is refused with the reason: ONNX Runtime runs one graph per call (no speculative decoding loop,
  no drafter, no dynamic depth) and its cache is the graph's own float tensors.

A technique without an adapter takes its family's refusal (:data:`ONNX`'s ``defaults``). Packages add adapters
through the entry-point group ``economize.onnx_techniques`` or with ``@ONNX.register``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.config import TechniqueConfig
from ...registry import Registry
from ..export_adapters import QuantExport, float_zero_points, int_bits, register_weight_adapters
from ..runtime_export import ExportRuntime, ExportTechnique
from ..vllm.serving.base import Role, Support
from .graph import NBITS, nbits_reason

if TYPE_CHECKING:
    from torch import nn

    from ...techniques.quant.grid import QuantizedWeight
    from ..runtime_export import ExportBuilder

__all__ = ["ONNX", "ONNX_TECHNIQUES", "OnnxLowRank", "OnnxQuant"]

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
    economize's dequantized map. A weight with ``act_order`` groups or an input width its group size does not divide
    is written dequantized (the export then reports ``DENSIFIED``). Other bit widths, group sizes that are not a
    power of two of at least 16 and real-valued zero points are refused; ``backend="fake"`` is written dequantized.
    """

    reason = ("int2 / int4 / int8 codes of the auto, compressed-tensors, marlin, machete and torchao backends are "
              "stored as ONNX Runtime MatMulNBits (exact) and run by its quantized matmul kernels")
    fp8_activations_reason = (
        "ONNX Runtime's MatMulNBits is weight-only (no fp8 activations); export the weight-only model"
    )

    def layout_reason(self, config: TechniqueConfig) -> str | None:
        if int_bits(config) not in NBITS:
            return (f"{getattr(config, 'format', '?')} codes have no MatMulNBits form "
                    f"(int{', int'.join(map(str, NBITS))})")
        group = getattr(config, "group_size", None)
        if group is None or group < 16 or group & (group - 1):
            return f"group_size={group} is not a power of two >= 16 (MatMulNBits' block_size)"
        if float_zero_points(config):
            return "MatMulNBits stores integer zero points; these codes have real-valued ones (HQQ)"
        return None

    def stores_weight(self, name: str, weight: QuantizedWeight) -> str | None:
        return nbits_reason(weight)


register_weight_adapters(ONNX, OnnxQuant)


@ONNX.register(override=True)
class OnnxLowRank(ExportTechnique):
    """``lowrank`` on ONNX Runtime: every factored map stays two ``MatMul`` nodes (``down`` then ``up``).

    The graph stores ``rank * (in + out)`` numbers per map and multiplies by the factors, as the HF model does
    (:func:`~economize.backends.onnx.factored.factor_linears`). A quantized factor becomes a ``MatMulNBits`` node.
    """

    technique = "lowrank"
    role = Role.WEIGHTS
    support = Support.NATIVE
    reason = "each factored map is stored as its two factors and runs as two MatMul nodes"

    def contribute(self, technique: nn.Module, builder: ExportBuilder) -> None:
        from ...techniques.lowrank.module import LowRankLinear
        from ..vllm.serving.weights import dense_modules

        for name, module in dense_modules(builder.base).items():
            if isinstance(module, LowRankLinear):
                builder.factored[name] = module

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
