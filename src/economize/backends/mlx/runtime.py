"""The MLX runtime (Apple silicon, ``mlx_lm``): its technique adapters and the support level of every technique.

The export is an ``mlx_lm`` model folder. What ``mlx_lm`` realizes of economize's techniques:

* weights: integer codes of 2 to 8 bits in contiguous groups of 32, 64 or 128 are stored as MLX's affine
  quantization (``QuantizedLinear``: packed codes, scales, biases); other codes and the ``fake`` backend are written
  dequantized; a depth-pruned model is a plain model; pruning, low rank and LittleBit are refused (``mlx_lm``
  builds its models from its own architecture code, one dense weight per linear map, and has no sparse kernel),
  except the explicit masked-dense pruning reference ``prune(semi_structured=False)``, written dense;
* drafting: a separate draft model (``draft_model``, ``placed_draft_model``) becomes ``mlx_lm.generate
  --draft-model`` (``mlx_lm``'s speculative decoding);
* KV eviction: StreamingLLM with 4 sinks (``kv_evict(method="streaming")``) is ``mlx_lm``'s rotating cache
  (``--max-kv-size``, which keeps the first 4 entries);
* everything else is refused with the reason (``mlx_lm`` has no prompt lookup, no hidden-state drafters, no dynamic
  depth, no activation sparsity, no fp8 cache; its ``--kv-bits`` cache quantizes keys per token, KIVI per channel).

A technique without an adapter takes its family's refusal (:data:`MLX`'s ``defaults``). Packages add adapters
through the entry-point group ``economize.mlx_techniques`` or with ``@MLX.register``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from ...core.config import TechniqueConfig
from ...registry import Registry
from ..export_adapters import QuantExport, StreamingWindowExport, int_bits, register_weight_adapters
from ..runtime_export import ExportBuilder, ExportRuntime, ExportTechnique
from ..vllm.serving.base import Role, Support
from .writer import MLX_BITS, MLX_GROUP_SIZES, mlx_reason

if TYPE_CHECKING:
    from ...techniques.quant.grid import QuantizedWeight

__all__ = ["MLX", "MLX_TECHNIQUES", "MlxDraftModel", "MlxQuant", "MlxStreaming"]

#: The MLX adapters by technique name (the :attr:`ExportRuntime.adapters` of :data:`MLX`).
MLX_TECHNIQUES: Registry[type[ExportTechnique]] = Registry("MLX technique adapter", "economize.mlx_techniques")

_DRAFTS = ("mlx_lm drafts with a separate draft model (--draft-model); it has no hook for drafters that read the "
           "target's hidden states")

MLX = ExportRuntime(
    name="mlx",
    title="MLX",
    adapters=MLX_TECHNIQUES,
    defaults={
        "dynamic-depth": (Role.FORWARD, "mlx_lm runs every layer for every token (no per-token depth routing)"),
        "drafting": (Role.DRAFTER, _DRAFTS),
        "draft-compression": (Role.DRAFT_OPTION, "mlx_lm runs the draft model it is given; quantize the draft "
                                                 "model itself"),
        "KV-quantization": (Role.KV_CACHE, "mlx_lm's --kv-bits cache quantizes keys and values per token in groups; "
                                           "it has no fp8 or KIVI (per-channel key) cache"),
        "KV-eviction": (Role.KV_EVICTION, "mlx_lm's rotating cache keeps 4 sinks and a recent window; it has no "
                                          "score-based eviction"),
        "activation-sparsity": (Role.FORWARD, "MLX has no activation-sparse kernels"),
    },
    fallback="the MLX export has no path for this technique",
    exclusive={Role.DRAFTER: "draft model (--draft-model)", Role.KV_EVICTION: "cache size (--max-kv-size)"},
)


class MlxQuant(QuantExport):
    """``quant``: integer codes are stored as MLX affine quantization (``scales * code + biases``).

    The codes are copied as they are and ``biases = -scale * zero``; scales and biases are stored in the export's
    float type, so a float16 or bfloat16 export rounds them. A weight with ``act_order`` groups or an input width
    its group size does not divide is written dequantized (the export then reports ``DENSIFIED``). Other formats
    and group sizes other than 32, 64 or 128 are refused; ``backend="fake"`` is written dequantized.
    """

    reason = ("int2-int8 codes of the auto, compressed-tensors, marlin, machete and torchao backends in groups of 32, "
              "64 or 128 are stored as MLX affine quantization (packed uint32 codes, scales, biases) and run by "
              "MLX's quantized matmul")
    fp8_activations_reason = "MLX's QuantizedLinear is weight-only (no fp8 activations); export the weight-only model"

    def layout_reason(self, config: TechniqueConfig) -> str | None:
        bits = int_bits(config)
        if bits not in MLX_BITS:
            return (f"{getattr(config, 'format', '?')} codes have no MLX affine form "
                    f"(int{', int'.join(map(str, MLX_BITS))})")
        group = getattr(config, "group_size", None)
        if group not in MLX_GROUP_SIZES:
            return f"group_size={group} is not one of MLX's group sizes {', '.join(map(str, MLX_GROUP_SIZES))}"
        return None

    def stores_weight(self, name: str, weight: QuantizedWeight) -> str | None:
        return mlx_reason(weight)


register_weight_adapters(MLX, MlxQuant)


@MLX.register
class MlxDraftModel(ExportTechnique):
    """``draft_model``: the draft LM is written to ``draft/`` and run with ``mlx_lm.generate --draft-model``."""

    technique = "draft_model"
    role = Role.DRAFTER
    support = Support.NATIVE
    reason = "mlx_lm.generate --draft-model draft --num-draft-tokens N (the draft LM written in the same layout)"

    def contribute(self, technique: nn.Module, builder: ExportBuilder) -> None:
        builder.drafter = getattr(technique, "draft_model")  # noqa: B009 (nn.Module types attributes as tensors)
        builder.set("--draft-model", "draft", owner=self.technique)
        builder.set("--num-draft-tokens", builder.num_draft_tokens, owner=self.technique)


@MLX.register
class MlxPlacedDraftModel(MlxDraftModel):
    """``placed_draft_model``: the same draft folder (MLX runs both models on the one Apple GPU)."""

    technique = "placed_draft_model"
    reason = "mlx_lm.generate --draft-model draft; MLX runs both models on the same device (no placement)"


@MLX.register
class MlxStreaming(StreamingWindowExport):
    """``kv_evict(method="streaming", sinks=4)``: ``mlx_lm``'s rotating cache of ``--max-kv-size`` entries."""

    reason = ("mlx_lm's RotatingKVCache (--max-kv-size sinks + recent) for kv_evict(method='streaming') with 4 "
              "sinks, the number mlx_lm keeps; score-based methods are refused")
    fixed_sinks = 4

    def window_settings(self, sinks: int, recent: int) -> dict[str, object]:
        return {"--max-kv-size": sinks + recent}


#: Techniques whose family default would be too vague, with the reason MLX cannot run them.
_REFUSALS: dict[str, tuple[Role, str]] = {
    "lookup": (Role.DRAFTER, "mlx_lm has no prompt lookup decoding (only --draft-model)"),
    "eagle": (Role.DRAFTER, "mlx_lm has no EAGLE draft head (it drafts with a separate model)"),
    "eagle3": (Role.DRAFTER, "mlx_lm has no EAGLE-3 draft head (it drafts with a separate model)"),
    "medusa": (Role.DRAFTER, "mlx_lm has no Medusa heads (it drafts with a separate model)"),
    "vendor_mtp": (Role.DRAFTER, "mlx_lm drops multi-token-prediction layers"),
    "magicdec": (Role.DRAFTER, "mlx_lm drafts with a separate draft model (--draft-model); it has no hook that "
                              "drafts with the target itself on a StreamingLLM view of its KV cache"),
    "triforce": (Role.DRAFTER, "mlx_lm drafts with a separate draft model (--draft-model); it has no hook that "
                              "drafts with the target itself on retrieved chunks of its KV cache"),
    "frspec": (Role.DRAFT_OPTION, "mlx_lm has no reduced draft vocabulary"),
    "draft_quant": (Role.DRAFT_OPTION, "quantize the draft LM itself (economize.apply(draft, QuantConfig(...)))"),
    "kv_fp8": (Role.KV_CACHE, "MLX has no fp8 KV cache (its --kv-bits cache is integer, per token)"),
    "teal": (Role.FORWARD, "MLX has no activation-sparse kernels"),
}

for _name, (_role, _reason) in _REFUSALS.items():
    MLX.refuse(_name, _role, _reason)
