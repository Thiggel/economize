"""The Kangaroo technique: the model's first ``l`` layers, a trained one-layer attention adapter and the LM head."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from torch import nn

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import Capability, Technique, register_technique
from ...errors import UnsupportedModelError
from ...modules.dtype import copy_dtype_of
from .._drafting import StepDrafterDefaults
from ..eagle3.technique import _last_full_attention_layer, _target_bias
from .adapter import KangarooAdapter
from .config import KangarooConfig

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...core.model import EfficientModel
    from ...runtime import LayerContext, LayerRunner
    from ...train.objective import Batch, ParamGroup, TeacherOutputs, TeacherRequest
    from .drafter import KangarooDrafter

__all__ = ["KangarooTechnique"]


@register_technique
class KangarooTechnique(StepDrafterDefaults, Technique):
    """Kangaroo (Liu et al., 2024): self-speculative drafting through a shallow sub-network and an adapter.

    Draft step at position ``p``: the model's layers ``1..l`` run on the token ``x_p`` (writing their keys and
    values into the model's cache, as early exit does), the adapter (:class:`KangarooAdapter`) reads their output
    ``h^l_p`` and attends to its own keys and values of positions ``<= p``, and the model's LM head reads the
    adapter's output to draft token ``p + 1``. Layers ``1..l`` and the head are the model's, frozen; the next exact
    pass verifies the drafts with the whole model.

    The adapter's entry of a committed token comes from the exact pass's output of layer ``l``, which equals the
    shallow path's (the layers are unmodified), so the drafter never reruns the shallow layers on accepted tokens.

    Parameters (float32, or the model's dtype with ``param_dtype="model"``; on the device of the LM head):
    ``adapter.*``. The adapter's output norm starts with the gain of the model's final norm, so that an untrained
    adapter with a small ``o_proj`` drafts like an early exit at layer ``l``.
    """

    name = KangarooConfig.technique
    config_class = KangarooConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True, trainable=("adapter_net",))
    capabilities = Capability.DRAFTER
    config: KangarooConfig

    def __init__(self, config: KangarooConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)
        self.exit_layer = config.exit_layer
        text, d = adapter.config, adapter.hidden_size
        full = _last_full_attention_layer(adapter)
        geometry = adapter.head_geometry(full) if full is not None else adapter.head_geometry(1)
        heads = config.num_heads or geometry.heads
        kv_heads = config.num_kv_heads or (geometry.kv_heads if config.num_heads is None else heads)
        eps = config.rms_norm_eps or float(getattr(text, "rms_norm_eps", None) or 1e-6)
        bias = config.attention_bias if config.attention_bias is not None else _target_bias(adapter)
        self.adapter_net = KangarooAdapter(d, heads, kv_heads, config.head_dim or geometry.head_dim, eps, bias)
        self.rope_table: str = adapter.attention_spec(full).rope if full is not None else "default"
        for linear in self.adapter_net.modules():
            if isinstance(linear, nn.Linear):
                nn.init.normal_(linear.weight, std=0.02)
                if linear.bias is not None:
                    nn.init.zeros_(linear.bias)
        self._init_norm(eps)
        self.to(device=adapter.head_device, dtype=copy_dtype_of(config.param_dtype, adapter.dtype))

    @torch.no_grad()
    def _init_norm(self, eps: float) -> None:
        """The output norm's gain from the model's final norm, read off the norm of a constant vector (which also
        covers ``(1 + w)`` norms)."""
        adapter = self.adapter
        try:
            probe = torch.ones(1, adapter.hidden_size, dtype=adapter.dtype, device=adapter.head_device)
            gain = adapter.final_norm(probe)[0].float().cpu() * math.sqrt(1.0 + eps)
        except (RuntimeError, TypeError):  # a norm that rejects a lone vector: keep unit gains
            return
        if torch.isfinite(gain).all():
            self.adapter_net.norm.weight.copy_(gain)

    # ---- the pieces of a draft step ----------------------------------------------------------------------------
    @property
    def layer_span(self) -> tuple[int, int]:
        """First and last model layer of a draft step."""
        return 1, self.exit_layer

    @property
    def device(self) -> torch.device:
        return self.adapter_net.q_proj.weight.device

    @property
    def dtype(self) -> torch.dtype:
        """Storage and compute dtype of the adapter (``KangarooConfig.param_dtype``)."""
        return self.adapter_net.q_proj.weight.dtype

    def run(self, runner: LayerRunner, embedding: torch.Tensor, ctx: LayerContext) -> torch.Tensor:
        """Output of layer ``l`` for token embeddings ``[B, S, d]``, writing the K/V of layers ``1..l``."""
        hidden, _ = runner.run(embedding, ctx, 1, self.exit_layer)
        return hidden

    def rotary_table(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(cos, sin)`` ``[B, S, D']`` of the adapter for positions ``[B, S]``, in :attr:`dtype`."""
        weight = self.adapter_net.q_proj.weight
        table = self.adapter.position_embeddings(weight, positions, self.rope_table)
        if not (isinstance(table, tuple) and len(table) == 2):
            raise UnsupportedModelError(f"{self.adapter.model_type!r} has no (cos, sin) rotary table for kangaroo")
        cos, sin = table
        return cos.to(self.device, self.dtype), sin.to(self.device, self.dtype)

    def causal(self, hidden: torch.Tensor) -> torch.Tensor:
        """The adapter's outputs ``[B, T, d]`` over a sequence of shallow states ``[B, T, d]`` (positions ``0..T-1``):
        entry ``t`` drafts token ``t + 1``."""
        batch, length = hidden.shape[:2]
        positions = torch.arange(length, device=self.device).expand(batch, length)
        return self.adapter_net(hidden, self.rotary_table(positions))

    def logits(self, out: torch.Tensor) -> torch.Tensor:
        """Float logits ``[..., V]`` of adapter outputs through the model's LM head (and its logit transform)."""
        adapter = self.adapter
        return adapter.transform_logits(adapter.lm_head(out.to(adapter.head_device, adapter.dtype))).float()

    # ---- technique interface ----------------------------------------------------------------------------------
    def drafter(self, runner: LayerRunner) -> KangarooDrafter:
        from .drafter import KangarooDrafter

        return KangarooDrafter(self, runner)

    def extra_repr(self) -> str:
        net = self.adapter_net
        return f"l={self.exit_layer}, heads={net.num_heads}/{net.num_kv_heads}x{net.head_dim}"

    # ---- training (the TrainableDrafter protocol of economize.train) ----------------------------------------------
    def train_request(self, model: EfficientModel) -> TeacherRequest:
        """The output of layer ``l``."""
        from ...train.objective import TeacherRequest

        return TeacherRequest(hidden_layers=(self.exit_layer,))

    def train_logits(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs) -> Mapping[int, torch.Tensor]:
        """Draft logits ``{1: [B, T, V]}``: position ``t`` drafts token ``t + 1`` (Kangaroo's training, the
        adapter on the teacher-forced shallow states)."""
        return {1: self.logits(self.causal(teacher.hidden[self.exit_layer]))}

    # ---- training (the StepDrafter protocol of economize.train, objective step_distill) --------------------------
    def step_states(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs, starts: torch.Tensor,
                    steps: int) -> list[torch.Tensor]:
        """Adapter outputs of draft steps ``1..steps`` after exact passes that ended at ``starts``. The shallow
        layers are the model's, so a draft step on the true token at ``s + k`` computes the teacher's ``h^l`` there,
        and the adapter's causal pass over the teacher's states gives every step at once."""
        out = self.causal(teacher.hidden[self.exit_layer])
        index = starts.to(out.device)
        return [out[:, index + k] for k in range(1, steps + 1)]

    def step_logits(self, states: torch.Tensor, step: int) -> torch.Tensor:
        """:meth:`logits`."""
        return self.logits(states)

    def step_parameter_groups(self) -> list[ParamGroup]:
        """The adapter, as one new module."""
        from ...train.objective import ParamGroup

        return [ParamGroup(self.name, list(self.parameters()), "head")]
