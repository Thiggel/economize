"""The ``"vllm/medusa"`` exporter: economize Medusa heads as the draft folder vLLM's Medusa model loads.

vLLM's Medusa (``model_type="medusa"``, ``architectures=["MedusaModel"]``) runs, per head ``h``, the residual blocks
``x + SiLU(W x + b)`` named ``blocks.{h}.layers.{l}`` on the target's last hidden state after the final norm, then
its own vocabulary projection ``lm_heads.{h}``. That is economize's ``head="own"`` Medusa head exactly (its blocks
read the post-norm state, :meth:`economize.techniques.medusa.MedusaTechnique.head_logits`), so the export renames the
tensors and sets ``medusa_fc_bias=True``.

A ``head="shared"`` head runs its blocks on the pre-norm state and then the model's final norm and LM head. vLLM
has no such head (its ``original_lm_head`` shares one projection across heads, after the norm), so the export
refuses it.

vLLM builds ``num_speculative_tokens`` heads and loads the first ones, so a plan may draft fewer tokens per step
than the technique has heads, never more. Static Medusa trees are not used: vLLM drafts one chain (the top-1 token
of each head).
"""

from __future__ import annotations

from typing import Any

import torch

from ....errors import ConfigError
from ...exporters import DrafterExporter, register_exporter

__all__ = ["ARCHITECTURE", "MODEL_TYPE", "MedusaVllmExporter"]

#: ``model_type`` of vLLM's Medusa config.
MODEL_TYPE = "medusa"
#: Architecture of vLLM's Medusa draft model.
ARCHITECTURE = "MedusaModel"


@register_exporter("vllm", "medusa")
class MedusaVllmExporter(DrafterExporter):
    """Medusa heads (``head="own"``) as vLLM's Medusa draft model, see the module docstring."""

    def own_heads(self, model: Any) -> Any:
        """The model's Medusa technique.

        Raises:
            ConfigError: The heads are ``head="shared"``.
        """
        technique = self.technique_of(model)
        if technique.config.head != "own":
            raise ConfigError(
                "vLLM's Medusa reads the post-norm state and projects with each head's own matrix; a head='shared' "
                "Medusa (blocks before the model's final norm and LM head) has no vLLM equivalent. Train with "
                "head='own' to serve on vLLM."
            )
        return technique

    def config(self, model: Any, **options: Any) -> dict[str, Any]:
        """The ``config.json`` vLLM's ``MedusaConfig`` reads (``options`` are accepted and ignored)."""
        technique = self.own_heads(model)
        vocab = int(technique.heads[0].proj.weight.shape[0])
        return {
            "model_type": MODEL_TYPE,
            "architectures": [ARCHITECTURE],
            "hidden_size": int(technique.adapter.hidden_size),
            "vocab_size": vocab,
            "truncated_vocab_size": vocab,
            "num_heads": int(technique.config.num_heads),
            "num_hidden_layers": int(technique.config.num_layers),
            "medusa_fc_bias": True,
        }

    def live_tensors(self, model: Any) -> dict[str, torch.Tensor]:
        """The heads' tensors as they live in the model, under vLLM's parameter names."""
        technique = self.own_heads(model)
        out: dict[str, torch.Tensor] = {}
        for h, head in enumerate(technique.heads):
            for i, block in enumerate(head.blocks):
                out[f"blocks.{h}.layers.{i}.weight"] = block.linear.weight.detach()
                out[f"blocks.{h}.layers.{i}.bias"] = block.linear.bias.detach()
            out[f"lm_heads.{h}.weight"] = head.proj.weight.detach()
        return out

    def tensors(self, model: Any) -> dict[str, torch.Tensor]:
        """:meth:`live_tensors` in float32 on the CPU."""
        return {name: tensor.to("cpu", torch.float32).contiguous()
                for name, tensor in self.live_tensors(model).items()}
