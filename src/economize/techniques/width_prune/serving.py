"""What a serving engine needs of a width-pruned model: the refusals shared by the vLLM and SGLang adapters and the
merged recovery LoRA.

An engine builds every layer from the one global config, so it serves the pruned model only when every layer has
the same widths (:attr:`WidthPruneConfig.uniform_widths`) and the rewritten config describes the model
(``summary["plain_checkpoint"]``). Each engine's implementation of an architecture further decides whether it reads
an explicit ``head_dim`` (needed once heads are removed at a fixed hidden width) and whether it builds the output
and down-map biases that FLAP's compensation adds.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any

from torch import Tensor, nn

from ...arch.config import config_value, text_config

__all__ = ["EngineArchitectures", "merged_lora_state", "serving_obstacle"]


@dataclass(frozen=True)
class EngineArchitectures:
    """What one engine's implementations of the supported architectures read from the config.

    Attributes:
        engine: Engine name, for messages.
        head_dim: ``model_type`` values whose implementation takes the head size from ``config.head_dim``.
        output_bias: ``model_type`` values that build a bias on the attention output map (FLAP's attention
            compensation) when ``attention_bias`` is set.
        mlp_bias: ``model_type`` values that build a bias on the down map (FLAP's MLP compensation) when
            ``mlp_bias`` is set.
    """

    engine: str
    head_dim: Collection[str] = field(default_factory=frozenset)
    output_bias: Collection[str] = field(default_factory=frozenset)
    mlp_bias: Collection[str] = field(default_factory=frozenset)


def serving_obstacle(technique: Any, base: nn.Module, engine: EngineArchitectures) -> str | None:
    """Why ``engine`` cannot load the width-pruned ``base`` as a plain checkpoint, or ``None``.

    Args:
        technique: The attached :class:`~economize.techniques.width_prune.WidthPruneTechnique`.
        base: The pruned Hugging Face model.
        engine: The engine's architecture facts.
    """
    config, summary = technique.config, getattr(technique, "summary", {}) or {}
    model_type = str(getattr(base.config, "model_type", ""))
    if not config.uniform_widths or not summary.get("uniform", False):
        return (f"{engine.engine} builds every layer from one config, and this width_prune leaves layers of different "
                "widths (structure='adaptive' or protected layers); use structure='uniform' with "
                "keep_first=keep_last=0")
    if not summary.get("plain_checkpoint", False):
        return (f"the rewritten config does not describe the pruned {model_type!r} model (its Hugging Face "
                "implementation has no bias for FLAP's compensation); use compensate=False or a llama model")
    text = text_config(base.config)  # a multimodal wrapper keeps the decoder's widths in its text config
    heads, head_dim, hidden = (int(config_value(text, name, 0) or 0)
                               for name in ("num_attention_heads", "head_dim", "hidden_size"))
    if heads and head_dim and hidden // heads != head_dim and model_type not in engine.head_dim:
        return (f"{engine.engine}'s {model_type!r} implementation derives the head size from hidden_size // "
                f"num_attention_heads ({hidden} // {heads}), which differs from head_dim={head_dim} after head "
                f"pruning; width_prune with attention pruning serves on {engine.engine} for model types "
                f"{sorted(engine.head_dim)} (or prune the MLP only: attn_sparsity=0)")
    compensated = summary.get("compensated", {})
    if compensated.get("attention") and model_type not in engine.output_bias:
        return (f"{engine.engine}'s {model_type!r} implementation has no attention output bias, which FLAP's "
                "attention compensation needs; use compensate=False")
    if compensated.get("mlp") and model_type not in engine.mlp_bias:
        return (f"{engine.engine}'s {model_type!r} implementation has no down-map bias, which FLAP's MLP "
                "compensation needs; use compensate=False")
    return None


def merged_lora_state(technique: Any, base: nn.Module) -> dict[str, dict[str, Tensor]]:
    """The recovery LoRA multiplied into the weights it adapts, as ``{module name: {"weight": merged}}`` (empty
    without a LoRA)."""
    lora = getattr(technique, "lora", None)
    if lora is None:
        return {}
    names = {id(module): name for name, module in base.named_modules()}
    return {names[id(module)]: {"weight": weight} for module, weight in lora.merged() if id(module) in names}
