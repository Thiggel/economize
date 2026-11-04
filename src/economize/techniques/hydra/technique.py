"""The Hydra technique: sequentially dependent draft heads, optionally over a prefix attention layer (Hydra++)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import Capability, Technique, register_technique
from ...errors import UnsupportedModelError
from ...modules.dtype import copy_dtype_of
from .._drafting import StepDrafterDefaults
from ..eagle3.technique import _last_full_attention_layer, _target_bias
from ..medusa.trees import TreePlan
from .config import HydraConfig
from .heads import HydraHead, InputBlock
from .prefix import PrefixLayer

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...core.model import EfficientModel
    from ...runtime import LayerRunner
    from ...train.objective import Batch, ParamGroup, TeacherOutputs, TeacherRequest
    from .drafter import HydraDrafter

__all__ = ["HydraTechnique"]


@register_technique
class HydraTechnique(StepDrafterDefaults, Technique):
    """Hydra heads (Ankner et al., 2024): head ``j`` drafts token ``t + 1 + j`` from the base state ``P_t`` of the last
    processed token ``t`` and the embeddings of tokens ``t + 1..t + j`` (see :class:`HydraConfig`).

    The base state is the model's last hidden state at ``t`` (post-norm with ``head="own"``, pre-norm with
    ``head="shared"``), or with ``prefix=True`` the output of the prefix layer at ``t``, which attends causally over
    the base states of all tokens up to ``t``. The prefix layer takes the geometry and rotary table of the model's
    last full-attention layer.

    Parameters (float32, or the model's dtype with ``param_dtype="model"``; on the device of the LM head):
    ``prefix.*`` (with ``prefix=True``), ``heads.{j}.blocks.{i}.*`` and, with ``head="own"``, ``heads.{j}.proj``
    (initialized from the LM head, zero bias).
    """

    name = HydraConfig.technique
    config_class = HydraConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True, trainable=("prefix", "heads"))
    capabilities = Capability.DRAFTER
    config: HydraConfig

    def __init__(self, config: HydraConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)
        own = config.head == "own"
        d = adapter.hidden_size
        lm_weight = cast(torch.Tensor, adapter.lm_head.weight)
        vocab = lm_weight.shape[0]
        self.prefix: PrefixLayer | None = None
        self.rope_table = "default"
        if config.prefix:
            text = adapter.config
            full = _last_full_attention_layer(adapter)
            geometry = adapter.head_geometry(full) if full is not None else adapter.head_geometry(1)
            width = config.prefix_intermediate_size or int(getattr(text, "intermediate_size", None) or 4 * d)
            eps = float(getattr(text, "rms_norm_eps", None) or 1e-6)
            self.prefix = PrefixLayer(d, width, geometry.heads, geometry.kv_heads, geometry.head_dim, eps,
                                      bias=_target_bias(adapter), output_norm=own)
            for linear in self.prefix.modules():
                if isinstance(linear, nn.Linear):
                    nn.init.normal_(linear.weight, std=0.02)
                    if linear.bias is not None:
                        nn.init.zeros_(linear.bias)
            self.rope_table = adapter.attention_spec(full).rope if full is not None else "default"
        heads = [HydraHead(d, vocab, j + 1, config.num_layers, projection=own, shared=not own)
                 for j in range(1, config.num_heads + 1)]
        self.heads = nn.ModuleList(heads)
        if own:
            with torch.no_grad():
                for head in heads:
                    assert head.proj is not None
                    head.proj.weight.copy_(lm_weight.detach())
                    head.proj.bias.zero_()
        self.plan = TreePlan(config.paths) if config.paths is not None else None
        self.to(device=adapter.head_device, dtype=copy_dtype_of(config.param_dtype, adapter.dtype))

    @property
    def device(self) -> torch.device:
        return cast(InputBlock, cast(HydraHead, self.heads[0]).blocks[0]).linear.weight.device

    @property
    def dtype(self) -> torch.dtype:
        """Storage and compute dtype of the drafter (``HydraConfig.param_dtype``)."""
        return cast(InputBlock, cast(HydraHead, self.heads[0]).blocks[0]).linear.weight.dtype

    @property
    def num_heads(self) -> int:
        return self.config.num_heads

    # ---- the draft computation ---------------------------------------------------------------------------------
    def base(self, hidden: torch.Tensor) -> torch.Tensor:
        """The input ``[..., d]`` of the prefix layer (or the base state without one) for the model's last hidden
        states ``[..., d]`` (before the final norm), in :attr:`dtype`."""
        hidden = hidden.to(self.adapter.head_device)
        if self.config.head == "own":
            hidden = self.adapter.final_norm(hidden)
        return hidden.to(self.device, self.dtype)

    def rotary_table(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(cos, sin)`` ``[B, S, D']`` of the prefix layer for positions ``[B, S]``, in :attr:`dtype`."""
        table = self.adapter.position_embeddings(cast(torch.Tensor, self.heads[0].blocks[0].linear.weight),
                                                 positions.to(self.device), self.rope_table)
        if not (isinstance(table, tuple) and len(table) == 2):
            raise UnsupportedModelError(f"{self.adapter.model_type!r} has no (cos, sin) rotary table for the Hydra "
                                        "prefix layer; use prefix=False")
        cos, sin = table
        return cos.to(self.device, self.dtype), sin.to(self.device, self.dtype)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Token embeddings ``[..., d]`` of the model in :attr:`dtype`, on the drafter's device."""
        return self.adapter.embed(tokens).to(self.device, self.dtype)

    def head_logits(self, state: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Float logits ``[N, V]`` of head ``j = history.shape[1]`` for base states ``[N, d]`` and the tokens
        ``[N, j]`` after them (a pure function of its inputs, so a draft step can be replayed from a CUDA graph)."""
        x = torch.cat([state, self.embed(history).flatten(1)], -1)
        return self.logits_of(x, history.shape[1])

    def logits_of(self, x: torch.Tensor, head: int) -> torch.Tensor:
        """Float logits ``[..., V]`` of head ``head`` (1-based) for its inputs ``[..., (head + 1) d]``."""
        module = cast(HydraHead, self.heads[head - 1])
        if self.config.head == "own":
            return module(x.to(self.dtype)).float()
        adapter = self.adapter
        return adapter.logits(module.hidden(x.to(self.dtype)).to(adapter.dtype)).float().to(self.device)

    def prefix_states(self, hidden: torch.Tensor) -> torch.Tensor:
        """Base states ``[B, T, d]`` of whole sequences of last hidden states ``[B, T, d]`` (training): the prefix
        layer's causal pass, or :meth:`base` without a prefix layer."""
        base = self.base(hidden)
        if self.prefix is None:
            return base
        positions = torch.arange(base.shape[1], device=self.device)[None].expand(base.shape[0], -1)
        return self.prefix(base, self.rotary_table(positions))

    # ---- technique interface ----------------------------------------------------------------------------------
    def drafter(self, runner: LayerRunner) -> HydraDrafter:
        from .drafter import HydraDrafter

        return HydraDrafter(self)

    def extra_repr(self) -> str:
        c = self.config
        tree = "" if c.tree is None else f", tree={c.tree if isinstance(c.tree, str) else len(c.tree)}"
        return f"num_heads={c.num_heads}, num_layers={c.num_layers}, prefix={c.prefix}, head={c.head!r}{tree}"

    # ---- training (the StepDrafter protocol of economize.train, objective step_distill) --------------------------
    def train_request(self, model: EfficientModel) -> TeacherRequest:
        """The model's last hidden state (the input of the prefix layer or the heads)."""
        from ...train.objective import TeacherRequest

        return TeacherRequest(hidden_layers=(self.adapter.num_layers,))

    def step_states(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs, starts: torch.Tensor,
                    steps: int) -> list[torch.Tensor]:
        """Step ``k`` reads ``[P_t; E[x_{t+1}]; ...; E[x_{t+k}]]`` at every rollout start ``t`` with the true tokens
        (teacher forcing, as the reference trains)."""
        states = self.prefix_states(teacher.hidden[self.adapter.num_layers])
        starts = starts.to(self.device)
        ids = batch.input_ids.to(self.device)
        parts, out = [states[:, starts]], []
        for k in range(1, steps + 1):
            parts.append(self.embed(ids[:, starts + k]))
            out.append(torch.cat(parts, -1))
        return out

    def step_logits(self, states: torch.Tensor, step: int) -> torch.Tensor:
        """The logits of head ``step``."""
        return self.logits_of(states, step)

    def step_limit(self) -> int | None:
        """One draft step per head."""
        return self.num_heads

    def step_parameter_groups(self) -> list[ParamGroup]:
        """The prefix layer and all heads, as new modules."""
        from ...train.objective import ParamGroup

        return [ParamGroup(self.name, list(self.parameters()), "head")]
