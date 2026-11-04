"""SGLang adapters of the structured weight techniques (the smaller dense checkpoint the vLLM adapters write).

SGLang's ``llama``, ``mistral``, ``qwen2`` and ``qwen3`` implementations read ``head_dim`` from the config, so the
globally sliced SliceGPT model serves on all four (vLLM 0.30 refuses ``qwen2``, see
:mod:`economize.backends.vllm.serving.structured`).
"""

from __future__ import annotations

from typing import ClassVar

from torch import nn

from ....techniques.width_prune.serving import EngineArchitectures, merged_lora_state, serving_obstacle
from ...exporters import UnsupportedBackendError
from .base import Role, SglangPlanBuilder, SglangTechnique, Support, register_sglang_technique

__all__ = ["SliceGPTSglang", "WidthPruneSglang"]


@register_sglang_technique
class SliceGPTSglang(SglangTechnique):
    """``slicegpt``: the globally rotated model is a plain checkpoint of a smaller hidden width."""

    technique = "slicegpt"
    role = Role.WEIGHTS
    support = Support.NATIVE
    reason = ("rotation='global' is a plain checkpoint of the smaller hidden width; rotation='per_block' needs a "
              "shortcut matrix on every residual connection, which no SGLang architecture has")
    #: ``model_type`` values whose SGLang implementation takes the head size from ``config.head_dim``.
    head_dim_types: ClassVar[frozenset[str]] = frozenset({"llama", "mistral", "qwen2", "qwen3"})

    def level(self, technique: nn.Module) -> Support:
        config = getattr(technique, "config", None)
        return Support.NATIVE if getattr(config, "rotation", None) == "global" else Support.UNSUPPORTED

    def check(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        if self.level(technique) is Support.UNSUPPORTED:
            raise UnsupportedBackendError(
                "SGLang cannot serve slicegpt(rotation='per_block'): every residual connection of the sliced model "
                "multiplies by a matrix, which no SGLang architecture has; use rotation='global'"
            )
        model_type = str(getattr(builder.base.config, "model_type", ""))
        if model_type not in self.head_dim_types:
            raise UnsupportedBackendError(
                f"slicegpt serves on SGLang for model types {sorted(self.head_dim_types)}, not {model_type!r}"
            )

    def contribute(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        builder.rewrite_base = True
        builder.note(self.technique, f"hidden width {getattr(builder.base.config, 'hidden_size', '?')} after slicing")


@register_sglang_technique
class WidthPruneSglang(SglangTechnique):
    """``width_prune`` (FLAP, LLM-Pruner, Minitron): the pruned model of uniform widths is a plain checkpoint with
    smaller ``num_attention_heads``, ``num_key_value_heads``, ``intermediate_size`` and ``hidden_size``; a recovery
    LoRA is merged into the weights it adapts."""

    technique = "width_prune"
    role = Role.WEIGHTS
    support = Support.NATIVE
    reason = ("structure='uniform' without protected layers is a plain checkpoint of smaller widths; layers of "
              "different widths need a per-layer config, which no SGLang architecture reads, and SGLang's MLPs have no "
              "bias for FLAP's MLP compensation (use compensate=False)")
    #: What SGLang's implementations read from the config (head size, compensation biases).
    architectures: ClassVar[EngineArchitectures] = EngineArchitectures(
        "SGLang", head_dim=frozenset({"llama", "mistral", "qwen2", "qwen3"}),
        output_bias=frozenset({"llama", "mistral", "qwen3"}), mlp_bias=frozenset())

    def level(self, technique: nn.Module) -> Support:
        config = getattr(technique, "config", None)
        if not getattr(config, "uniform_widths", False):
            return Support.UNSUPPORTED
        mlp_compensated = getattr(config, "compensates", False) and getattr(config, "mlp_ratio", 0) > 0
        return Support.UNSUPPORTED if mlp_compensated and not self.architectures.mlp_bias else Support.NATIVE

    def check(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        obstacle = serving_obstacle(technique, builder.base, self.architectures)
        if obstacle is not None:
            raise UnsupportedBackendError(f"SGLang cannot serve this width_prune: {obstacle}")

    def contribute(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        builder.rewrite_base = True
        builder.dense.update(merged_lora_state(technique, builder.base))
        summary = getattr(technique, "summary", {})
        lora = " with the recovery LoRA merged" if getattr(technique, "lora", None) is not None else ""
        builder.note(self.technique, f"{summary.get('method', '?')}: {summary.get('kv_heads', ['?'])[0]} key/value "
                                     f"heads, {summary.get('intermediate', ['?'])[0]} FFN channels, hidden width "
                                     f"{summary.get('width', '?')}{lora}")
