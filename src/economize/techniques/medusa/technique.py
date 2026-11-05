"""The Medusa technique: K heads on the last hidden state that draft the tokens after the next one."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import Capability, Technique, register_technique
from ...modules.dtype import copy_dtype_of
from .._drafting import StepDrafterDefaults
from .config import MedusaConfig
from .heads import MedusaHead, ResBlock
from .trees import TreePlan

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...core.model import EfficientModel
    from ...runtime import LayerRunner
    from ...train.objective import Batch, ParamGroup, TeacherOutputs, TeacherRequest
    from .drafter import MedusaDrafter

__all__ = ["MedusaTechnique"]


@register_technique
class MedusaTechnique(StepDrafterDefaults, Technique):
    """Medusa: extra heads on the last hidden state draft several future tokens at once.

    Medusa heads (Cai et al., 2024): after an exact pass, head ``j`` drafts token ``t + 1 + j`` from the last
    hidden state at the last processed token ``t``, all heads at once and without running a decoder layer.

    Parameters (float32, or the model's dtype with ``param_dtype="model"``; on the device of the LM head):
    ``heads.{j}.blocks.{i}.linear`` and, with ``head="own"``, ``heads.{j}.proj`` (initialized from the LM head).
    """

    name = MedusaConfig.technique
    config_class = MedusaConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True, trainable=("heads",))
    capabilities = Capability.DRAFTER
    config: MedusaConfig

    def __init__(self, config: MedusaConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)
        own = config.head == "own"
        lm_weight = cast(torch.Tensor, adapter.lm_head.weight)
        vocab = lm_weight.shape[0]
        heads = [
            MedusaHead(adapter.hidden_size, vocab, config.num_layers, projection=own, zero_bias=not own)
            for _ in range(config.num_heads)
        ]
        self.heads = nn.ModuleList(heads)
        if own:
            with torch.no_grad():
                for head in heads:
                    assert head.proj is not None
                    head.proj.weight.copy_(lm_weight.detach())
        self.plan = TreePlan(config.paths) if config.paths is not None else None
        self.to(device=adapter.head_device, dtype=copy_dtype_of(config.param_dtype, adapter.dtype))

    @property
    def dtype(self) -> torch.dtype:
        """Storage and compute dtype of the heads (``MedusaConfig.param_dtype``)."""
        return cast(ResBlock, cast(MedusaHead, self.heads[0]).blocks[0]).linear.weight.dtype

    @property
    def num_heads(self) -> int:
        return self.config.num_heads

    def head_logits(self, hidden: torch.Tensor, count: int | None = None) -> torch.Tensor:
        """Float logits ``[..., count, V]`` of the first ``count`` heads (default all) for the model's last hidden
        states ``[..., d]`` (before the final norm)."""
        return self._logits(hidden, self.heads[: self.num_heads if count is None else count])

    def logits_of(self, hidden: torch.Tensor, head: int) -> torch.Tensor:
        """Float logits ``[..., V]`` of head ``head`` (1-based) alone, for training on long sequences."""
        return self._logits(hidden, [self.heads[head - 1]]).squeeze(-2)

    def _logits(self, hidden: torch.Tensor, heads: Iterable[nn.Module]) -> torch.Tensor:
        """Float logits ``[..., len(heads), V]`` of ``heads`` for last hidden states ``[..., d]``: an own head reads
        the post-norm state and projects with its own matrix, a shared head reads the pre-norm state and the
        model's final norm and LM head read its output."""
        adapter = self.adapter
        hidden = hidden.to(adapter.head_device)
        if self.config.head == "own":
            x = adapter.final_norm(hidden).to(self.dtype)
            return _stack([head(x) for head in heads]).float()
        x = hidden.to(self.dtype)
        states = _stack([cast(MedusaHead, head).hidden(x) for head in heads])
        return adapter.logits(states.to(adapter.dtype)).float()

    # ---- technique interface ----------------------------------------------------------------------------------
    def drafter(self, runner: LayerRunner) -> MedusaDrafter:
        from .drafter import MedusaDrafter

        return MedusaDrafter(self)

    def extra_repr(self) -> str:
        c = self.config
        tree = "" if c.tree is None else f", tree={c.tree if isinstance(c.tree, str) else len(c.tree)}"
        return f"num_heads={c.num_heads}, num_layers={c.num_layers}, head={c.head!r}{tree}"

    # ---- training (the TrainableDrafter protocol of economize.train) ----------------------------------------------
    def train_request(self, model: EfficientModel) -> TeacherRequest:
        """The model's last hidden state (the heads' input)."""
        from ...train.objective import TeacherRequest

        return TeacherRequest(hidden_layers=(self.adapter.num_layers,))

    def train_logits(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs) -> Mapping[int, torch.Tensor]:
        """Draft logits ``{j + 1: [B, T, V]}`` of heads ``j = 1..K``: position ``t`` of key ``k`` drafts token
        ``t + k`` (head ``j`` reads the last hidden state at ``t``, whose LM head predicts ``t + 1``)."""
        hidden = teacher.hidden[self.adapter.num_layers]
        return {j + 1: self.logits_of(hidden, j) for j in range(1, self.num_heads + 1)}

    # ---- training (the StepDrafter protocol of economize.train, objective step_distill) --------------------------
    def step_states(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs, starts: torch.Tensor,
                    steps: int) -> list[torch.Tensor]:
        """Every step reads the last hidden state at the rollout start (step ``k`` is head ``k``)."""
        hidden = teacher.hidden[self.adapter.num_layers]
        state = hidden[:, starts.to(hidden.device)]
        return [state] * steps

    def step_logits(self, states: torch.Tensor, step: int) -> torch.Tensor:
        """The logits of head ``step``."""
        return self.logits_of(states, step)

    def step_limit(self) -> int | None:
        """One draft step per head."""
        return self.num_heads

    def step_parameter_groups(self) -> list[ParamGroup]:
        """All heads, as new modules."""
        from ...train.objective import ParamGroup

        return [ParamGroup(self.name, list(self.parameters()), "head")]


def _stack(outputs: list[torch.Tensor]) -> torch.Tensor:
    """``outputs`` stacked along a new second-to-last axis (a single output without a copy)."""
    return outputs[0].unsqueeze(-2) if len(outputs) == 1 else torch.stack(outputs, -2)
