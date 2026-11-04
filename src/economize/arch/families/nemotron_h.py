"""NVIDIA Nemotron-H (``nemotron_h``): Mamba-2, attention, MLP and MoE blocks in one stack.

``config.hybrid_override_pattern`` (``M`` Mamba-2, ``*`` attention, ``-`` MLP, ``E`` MoE) becomes
``config.layers_block_type`` (``linear_attention``, ``full_attention``, ``mlp``, ``moe``). Every block is
``x + mixer(norm(x))``, and only the mixer differs, so one decoder layer of economize is one block of the model.
Attention uses no rotary embedding (the adapter keeps the default ``rotary_emb = None``). Mamba-2 blocks keep a
conv window and an SSM state in the :class:`~economize.runtime.HybridCache` (drafts are verified as chains and undone
by snapshot, restore and replay); MLP and MoE blocks keep no state and may be skipped like any token-wise layer.

The multi-token-prediction layers of the released checkpoints (``num_nextn_predict_layers``) are not part of the
``transformers`` model; a drafter that uses them belongs in the ``vendor_mtp`` drafter slot.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from ..base import ArchAdapter, register_adapter
from ..config import HFConfig
from ..protocols import LayerInputs
from ..spec import PassContext

if TYPE_CHECKING:
    from ...attention.banded import StructuredMask

__all__ = ["NemotronHAdapter"]

#: The parts of the cost model's ``ModelShape.layer_parts`` that each block type holds.
_BLOCK_PARTS = {
    "linear_attention": frozenset({"mixer"}),
    "full_attention": frozenset({"attention"}),
    "mlp": frozenset({"mlp"}),
    "moe": frozenset({"mlp"}),
}


@register_adapter("nemotron_h")
class NemotronHAdapter(ArchAdapter):
    """``model.model.{embeddings, layers, norm_f}`` and ``model.lm_head``; blocks are typed by ``block_type``."""

    @property
    def decoder(self) -> nn.Module:
        """The backbone (``model.model``)."""
        return cast(nn.Module, self.model.model)

    @property
    def layers(self) -> nn.ModuleList:
        return cast(nn.ModuleList, self.decoder.layers)

    @property
    def embed_tokens(self) -> nn.Module:
        return cast(nn.Module, self.decoder.embeddings)

    @property
    def final_norm(self) -> nn.Module:
        return cast(nn.Module, self.decoder.norm_f)

    @property
    def lm_head(self) -> nn.Module:
        return cast(nn.Module, self.model.lm_head)

    @classmethod
    def shape_fields(cls, config: HFConfig, fields: Mapping[str, object]) -> Mapping[str, object]:
        """One part per block, ungated (relu^2) MLPs and experts, and the Mamba-2 mixer's weights.

        A Mamba-2 mixer holds ``in_proj`` (hidden to gate, ``x``, ``B``, ``C`` and one ``dt`` per head), the
        depthwise ``conv1d`` over ``x``, ``B`` and ``C`` and ``out_proj``. The shared expert is one MLP of width
        ``moe_shared_expert_intermediate_size`` without a gate, and ``moe_latent_size`` sets the width the routed
        experts work in.
        """
        types = [str(t) for t in config.layers_block_type]
        inner = config.mamba_num_heads * config.mamba_head_dim
        conv = inner + 2 * config.n_groups * config.ssm_state_size
        hidden = config.hidden_size
        mixer = hidden * (inner + conv + config.mamba_num_heads) + conv * config.conv_kernel + inner * hidden
        out: dict[str, object] = {
            "gated": False,
            "layer_types": tuple(types),
            "layer_parts": tuple(_BLOCK_PARTS[t] for t in types),
            "mixer_params": mixer,
            "dense_layers": frozenset(i for i, t in enumerate(types, start=1) if t == "mlp"),
            "shared_gate": False,
        }
        if "moe" in types:
            out.update(shared_intermediate=config.moe_shared_expert_intermediate_size,
                       expert_hidden=config.moe_latent_size or 0)
        else:
            out.update(experts=0, active_experts=0, expert_intermediate=0, shared_intermediate=0)
        return out

    @property
    def layer_types(self) -> list[str]:
        return [str(t) for t in self.config.layers_block_type][: self.num_layers]

    def layer_inputs(
        self,
        index: int,
        hidden: torch.Tensor,
        ctx: PassContext,
        cache: object,
        mask: torch.Tensor | StructuredMask | None,
    ) -> LayerInputs:
        return (), {
            "past_key_values": cache,
            "attention_mask": mask,
            "position_ids": ctx.position_ids(hidden.device),
            "use_cache": True,
        }
