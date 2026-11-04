"""vLLM adapters of the structured weight techniques, which leave a smaller dense model of the same architecture.

* ``slicegpt`` with ``rotation="global"``: the model of hidden width ``d`` (``hidden_size = d`` and an explicit
  ``head_dim`` in ``config.json``), which vLLM runs natively for the architectures whose vLLM implementation reads
  ``head_dim`` from the config (:attr:`SliceGPTVllm.head_dim_types`). vLLM 0.30's ``Qwen2ForCausalLM`` derives the
  head size from ``hidden_size // num_attention_heads`` and is refused. ``rotation="per_block"`` needs a matrix on
  every residual connection, which no vLLM architecture has.
"""

from __future__ import annotations

from typing import ClassVar

from torch import nn

from ....techniques.width_prune.serving import EngineArchitectures, merged_lora_state, serving_obstacle
from ...exporters import UnsupportedBackendError
from .base import PlanBuilder, Role, Support, VllmTechnique, register_vllm_technique

__all__ = ["SliceGPTVllm", "WidthPruneVllm"]


@register_vllm_technique
class SliceGPTVllm(VllmTechnique):
    """``slicegpt``: the globally rotated model is a plain checkpoint of a smaller hidden width."""

    technique = "slicegpt"
    role = Role.WEIGHTS
    support = Support.NATIVE
    reason = ("rotation='global' is a plain checkpoint of the smaller hidden width; rotation='per_block' needs a "
              "shortcut matrix on every residual connection, which no vLLM architecture has")
    #: ``model_type`` values whose vLLM 0.30 implementation takes the head size from ``config.head_dim`` (so the
    #: hidden width may differ from ``heads * head_dim``).
    head_dim_types: ClassVar[frozenset[str]] = frozenset({"llama", "mistral", "qwen3"})

    def level(self, technique: nn.Module) -> Support:
        config = getattr(technique, "config", None)
        return Support.NATIVE if getattr(config, "rotation", None) == "global" else Support.UNSUPPORTED

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        if self.level(technique) is Support.UNSUPPORTED:
            raise UnsupportedBackendError(
                "vLLM cannot serve slicegpt(rotation='per_block'): every residual connection of the sliced model "
                "multiplies by a matrix, which no vLLM architecture has; use rotation='global'"
            )
        model_type = str(getattr(builder.base.config, "model_type", ""))
        if model_type not in self.head_dim_types:
            raise UnsupportedBackendError(
                f"vLLM 0.30's {model_type!r} implementation derives the head size from hidden_size // "
                "num_attention_heads, which slicing changes; slicegpt serves on vLLM for model types "
                f"{sorted(self.head_dim_types)}"
            )

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        builder.rewrite_base = True
        builder.note(self.technique, f"hidden width {getattr(builder.base.config, 'hidden_size', '?')} after slicing")


@register_vllm_technique
class WidthPruneVllm(VllmTechnique):
    """``width_prune`` (FLAP, LLM-Pruner, Minitron): the pruned model of uniform widths is a plain checkpoint with
    smaller ``num_attention_heads``, ``num_key_value_heads``, ``intermediate_size`` and ``hidden_size``; a recovery
    LoRA is merged into the weights it adapts."""

    technique = "width_prune"
    role = Role.WEIGHTS
    support = Support.NATIVE
    reason = ("structure='uniform' without protected layers is a plain checkpoint of smaller widths; layers of "
              "different widths need a per-layer config, which no vLLM architecture reads")
    #: What vLLM's implementations read from the config (head size, compensation biases).
    architectures: ClassVar[EngineArchitectures] = EngineArchitectures(
        "vLLM 0.30", head_dim=frozenset({"llama", "mistral", "qwen3"}), output_bias=frozenset({"llama", "mistral"}),
        mlp_bias=frozenset({"llama"}))

    def level(self, technique: nn.Module) -> Support:
        config = getattr(technique, "config", None)
        return Support.NATIVE if getattr(config, "uniform_widths", False) else Support.UNSUPPORTED

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        obstacle = serving_obstacle(technique, builder.base, self.architectures)
        if obstacle is not None:
            raise UnsupportedBackendError(f"vLLM cannot serve this width_prune: {obstacle}")

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        builder.rewrite_base = True
        builder.dense.update(merged_lora_state(technique, builder.base))
        summary = getattr(technique, "summary", {})
        lora = " with the recovery LoRA merged" if getattr(technique, "lora", None) is not None else ""
        builder.note(self.technique, f"{summary.get('method', '?')}: {summary.get('kv_heads', ['?'])[0]} key/value "
                                     f"heads, {summary.get('intermediate', ['?'])[0]} FFN channels, hidden width "
                                     f"{summary.get('width', '?')}{lora}")
